"""Prepare and run the five-seed fast NEAT study with validation quality gates."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from statistics import mean
from tempfile import TemporaryDirectory
import time
from zipfile import ZIP_DEFLATED, ZipFile

from line_follower.config import EpisodeConfig, RobotConfig, verify_geometry
from line_follower.controllers import read_model_artifact
from line_follower.experiments import TrainConfig, append_csv, environment_record, snapshot_sources, train, write_json
from line_follower.fast_study import FITNESS_VERSION, SCHEDULE, SUITE_SEED, prepare_suite, quality_gate
from line_follower.pid import ResearchPIDConfig
from line_follower.pid_experiments import evaluate_candidate
from line_follower.suites import derived_seed, load_split, read_suite
from line_follower.tracks import TrackSpec
from run_corner_training import ROOT, digest, protected_tracks, read_json, session_lock
from run_pid_gain_tuning import protected_geometry

OUTPUT = ROOT / "runs/neat-fast-study"
PREVIOUS = ROOT / "runs/neat-speed-study"
PID_FOLLOWUP = ROOT / "runs/pid-speed-followup-20261116"
SEEDS = tuple(range(1, 6))
CAPS = (1.50, .50, 1.00)
WORKERS = 8


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalized(value):
    return json.loads(json.dumps(value))


def publish_directory(source, destination, commit_file):
    """Retry Windows folder locks; publish the checksum manifest last when copying."""
    for attempt in range(6):
        try:
            source.rename(destination)
            return
        except PermissionError:
            if attempt < 5:
                time.sleep(.1 * 2**attempt)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(commit_file))
    shutil.copy2(source / commit_file, destination / commit_file)


def settings(seed):
    return TrainConfig(strategy="multi", seed=derived_seed(SUITE_SEED, "evolution", seed),
                       generations=300, population=100, workers=WORKERS, backend="cpu",
                       episodes_per_genome=6, track_profile="fast", aggregation="mean_cvar",
                       cvar_fraction=.5, speed_schedule=SCHEDULE)


def software():
    env = environment_record()
    if env["dependencies"]["mujoco"] != "4.0.0" or env["dependencies"]["neat-python"] != "0.92":
        raise ValueError("Use the study's MuJoCo 4.0.0 and NEAT-Python 0.92 environment")
    neat = importlib.metadata.distribution("neat-python")
    hasher = hashlib.sha256()
    for path in sorted(neat.files, key=str):
        if str(path).startswith("neat/") and str(path).endswith(".py"):
            hasher.update(str(path).encode())
            hasher.update(Path(neat.locate_file(path)).read_bytes())
    return {key: env[key] for key in ("python", "dependencies", "source_sha256")} | {
        "launcher_sha256": digest(Path(__file__)), "neat_source_sha256": hasher.hexdigest(),
        "helper_sha256": {name: digest(ROOT / name) for name in ("run_corner_training.py", "run_pid_gain_tuning.py")}}


def input_sources():
    frozen = PREVIOUS / "assessment/frozen"
    manifest = read_json(frozen / "selection-freeze.json")
    sources = {"geometry-freeze.json": PREVIOUS / "prepared/inputs/geometry-freeze.json"}
    for seed in SEEDS:
        for phase in ("low", "ramp"):
            key = f"seed-{seed:02d}-{phase}"
            source = frozen / manifest["policies"][key]["file"]
            if digest(source) != manifest["policies"][key]["sha256"]:
                raise ValueError(f"The previous frozen policy changed: {key}")
            artifact = read_model_artifact(source)
            metadata = artifact["metadata"]
            if metadata["controller_semantics"] != "mirrored-sensor-history-v1":
                raise ValueError("Expected the existing sensor-history action mapping")
            sources[f"{key}.pkl"] = source
    pid_low = PREVIOUS / "prepared/inputs/pid-050.json"
    if digest(pid_low) != read_json(PREVIOUS / "prepared/protocol.json")["inputs"]["pid-050.json"]:
        raise ValueError("The original low-cap PID configuration changed")
    sources["pid-050.json"] = pid_low
    pid_selection = read_json(PID_FOLLOWUP / "selection-freeze.json")
    for cap in ("1.00", "1.50"):
        record = pid_selection["selections"][cap]
        source = PID_FOLLOWUP / record["config"]
        if digest(source) != record["sha256"]:
            raise ValueError("The saved PID configuration changed")
        sources[f"pid-{int(float(cap)*100):03d}.json"] = source
    for name, path in sources.items():
        if name.startswith("pid-") and ResearchPIDConfig.load(path).corner_supervision:
            raise ValueError("The comparison requires filtered recovery PID")
    return sources


def planned_protocol():
    sources = input_sources()
    prior = read_json(PREVIOUS / "prepared/protocol.json")
    robot = RobotConfig(**prior["robot"])
    verify_geometry(sources["geometry-freeze.json"], robot)
    episode = replace(EpisodeConfig(**prior["episode"]), fitness_version=FITNESS_VERSION,
                      completion_speed_bonus=.50, command_jitter_penalty=.20)
    return normalized({
        "study": "neat-fast-study-v1", "software": software(), "suite_seed": SUITE_SEED,
        "robot": asdict(robot), "episode": asdict(episode),
        "settings": {seed: asdict(settings(seed)) for seed in SEEDS},
        "inputs": {name: digest(path) for name, path in sources.items()},
        "validation_groups": {"mixed": 30, "speed": 20, "corner": 10},
        "test_groups": {"mixed": 90, "speed": 30, "corner": 30},
        "straight_goal_m_s": .95,
        "selection": "Final phase only. Pass quality gate, minimize mean paired time change, then maximize completion and minimize geometric RMSE.",
        "comparison_caps_m_s": CAPS,
    })


def verify(prepared):
    path = prepared / "protocol.json"
    if not path.is_file():
        raise ValueError("Run prepare before training or assessment")
    plan = read_json(path)
    if plan["software"] != software():
        raise ValueError("Source or dependencies changed since preparation")
    for file, sha in plan["prepared_files"].items():
        if digest(prepared / file) != sha:
            raise ValueError(f"Frozen study file changed: {file}")
    for seed in SEEDS:
        if plan["settings"][str(seed)] != normalized(asdict(settings(seed))):
            raise ValueError("Training settings changed since preparation")
    return plan


def evaluation_tasks(prepared, split, policies, caps, plan):
    entries = read_suite(prepared / "suite/suite.json")["splits"][split]
    for index, entry in enumerate(entries):
        for cap in caps:
            for name, path in policies.items():
                controller = ({"neat_model": str(path)} if path.suffix == ".pkl"
                              else {"parameters": read_json(path)})
                yield {"candidate": name, "robot_json": json.dumps({**plan["robot"], "max_speed": cap}),
                       "episode": plan["episode"], "seed": SUITE_SEED, "phase": split,
                       "tracks": [str(prepared / "suite" / entry["path"])], "slot_offset": index,
                       "detailed_metrics": True, "output": str(prepared), **controller}


def collect(tasks, workers, expected):
    results = []
    evolution_seeds = {task["neat_model"]: None for task in tasks if "neat_model" in task}
    for path in evolution_seeds:
        evolution_seeds[path] = read_model_artifact(path)["metadata"]["evolution_seed"]
    with mp.get_context("spawn").Pool(workers) as pool:
        for task, batch in zip(tasks, pool.imap(evaluate_candidate, tasks), strict=True):
            row, = batch
            if row["error"] or row["geometry_lateral_rmse_m"] is None:
                raise RuntimeError(f"Invalid fixed-policy evaluation: {row['error']}")
            name = task["candidate"]
            results.append({**row, "policy_id": name, "candidate": name,
                            "lineage_seed": int(name.rsplit("-", 1)[1]) if "neat_model" in task else None,
                            "evolution_seed": evolution_seeds[task["neat_model"]] if "neat_model" in task else None})
            if len(results) % 60 == 0 or len(results) == expected:
                print(f"Fixed-policy evaluation {len(results)}/{expected}", flush=True)
    if len(results) != expected:
        raise ValueError("Evaluation record count is incomplete")
    return results


def prepare(output):
    prepared = output / "prepared"
    if prepared.exists():
        return verify(prepared)
    plan = planned_protocol()
    sources = input_sources()
    protected = protected_tracks() | protected_geometry()
    long_course = ROOT / "runs/long-course-speed-challenge/track.json"
    if long_course.exists():
        protected.add(TrackSpec.load(long_course).geometry_hash)
    with TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
        ready = Path(temporary) / "prepared"
        (ready / "inputs").mkdir(parents=True)
        for name, source in sources.items():
            shutil.copyfile(source, ready / "inputs" / name)
        print("Generating disjoint training, validation, and test courses.", flush=True)
        suite = prepare_suite(ready / "suite", protected)
        for split, counts in (("validation", plan["validation_groups"]), ("test", plan["test_groups"])):
            actual = Counter(row["group"] for row in read_suite(suite)["splits"][split])
            if actual != counts:
                raise ValueError("Unexpected split group sizes")
        references = {f"seed-{seed:02d}": ready / f"inputs/seed-{seed:02d}-ramp.pkl" for seed in SEEDS}
        tasks = list(evaluation_tasks(ready, "validation", references, (1.50,), plan))
        print("Measuring the five frozen reference policies on 60 validation courses each.", flush=True)
        rows = collect(tasks, WORKERS, 300)
        dedicated = {entry["geometry_hash"]: entry.get("dedicated_straight", False)
                     for entry in read_suite(suite)["splits"]["validation"]}
        for name in references:
            write_json(ready / "references" / f"{name}.json",
                       {row["track_hash"]: {**row, "dedicated_straight": dedicated[row["track_hash"]]}
                        for row in rows if row["policy_id"] == name})
        snapshot_sources(ready / "source.zip")
        with ZipFile(ready / "source.zip", "a", compression=ZIP_DEFLATED) as archive:
            for source in (Path(__file__), ROOT / "run_pid_gain_tuning.py"):
                archive.write(source, source.name)
        plan["prepared_files"] = {path.relative_to(ready).as_posix(): digest(path)
                                  for path in sorted(ready.rglob("*")) if path.is_file()}
        write_json(ready / "protocol.json", plan)
        verify(ready)
        publish_directory(ready, prepared, "protocol.json")
        verify(prepared)
    return plan


def finalize(directory, prepared, plan, seed):
    status = read_json(directory / "status.json")
    if not status["complete"] or status["next_generation"] != 300:
        raise ValueError(f"Seed {seed} training is incomplete")
    generations = read_csv(directory / "generations.csv")
    if [(int(row["generation"]), float(row["wheel_speed_cap_m_s"])) for row in generations] != [
            (g, settings(seed).speed_at(g, 0)) for g in range(300)]:
        raise ValueError("The committed generation schedule is incomplete")
    quality = read_csv(directory / "quality.csv")
    if [int(row["generation"]) for row in quality] != list(range(300)):
        raise ValueError("Quality records are incomplete")
    eligible = [row for row in quality if int(row["generation"]) >= 200 and row["eligible"] == "True"]
    selected = None
    if eligible:
        chosen = max(eligible, key=lambda row: (-float(row["paired_time_change_s"]),
                                               int(row["completed"]), -float(row["geometric_rmse_m"])))
        best = directory / "best.pkl"
        metadata = read_model_artifact(best)["metadata"]
        expected = (-float(chosen["paired_time_change_s"]), int(chosen["completed"]),
                    -float(chosen["geometric_rmse_m"]))
        if (metadata["generation"] != int(chosen["generation"]) or tuple(metadata["validation_key"]) != expected
                or metadata["evolution_seed"] != settings(seed).seed
                or metadata["episode"] != plan["episode"]):
            raise ValueError("Saved winner differs from the quality-gated validation selection")
        selected = {"generation": int(chosen["generation"]), "sha256": digest(best),
                    "paired_time_change_s": float(chosen["paired_time_change_s"]),
                    "straight_goal": chosen["straight_goal"] == "True"}
    elif (directory / "best.pkl").exists():
        raise ValueError("An unqualified best.pkl exists")
    final_rows = [row for row in quality if int(row["generation"]) >= 200]
    closest = min(final_rows, key=lambda row: (len([s for s in row["reasons"].split(";") if s]),
                                              -int(row["completed"]), float(row["geometric_rmse_m"])))
    result = {"complete": True, "selected": selected, "diagnostic_tradeoff": closest}
    write_json(directory / "selection.json", result)
    return result


def train_seed(output, prepared, plan, seed, resume):
    directory = output / f"seed-{seed:02d}/training"
    start_path = directory.parent / "start.json"
    reference = prepared / f"references/seed-{seed:02d}.json"
    start = {"protocol_sha256": digest(prepared / "protocol.json"),
             "parent_sha256": plan["inputs"][f"seed-{seed:02d}-low.pkl"],
             "reference_sha256": digest(reference), "settings": plan["settings"][str(seed)]}
    checkpoint = directory / "checkpoints/latest.pkl.gz"
    if resume:
        if not start_path.exists() or read_json(start_path) != start:
            raise ValueError("Stage inputs changed since the original launch")
        if directory.exists() and any(directory.iterdir()) and not checkpoint.is_file():
            raise ValueError("No committed checkpoint exists. Archive the entire seed directory, including start.json, then launch without --resume.")
    else:
        if start_path.exists() or (directory.exists() and any(directory.iterdir())):
            raise FileExistsError("This seed already exists. Use --resume.")
        write_json(start_path, start)
    status = directory / "status.json"
    if not status.exists() or not read_json(status).get("complete"):
        train(directory, prepared / "suite/suite.json", settings(seed), RobotConfig(**plan["robot"]),
              EpisodeConfig(**plan["episode"]), resume=checkpoint if checkpoint.is_file() else None,
              initial_model=prepared / f"inputs/seed-{seed:02d}-low.pkl" if not checkpoint.is_file() else None,
              quality_reference=reference, geometry_lock=prepared / "inputs/geometry-freeze.json", live_preview=True)
    result = finalize(directory, prepared, plan, seed)
    print(f"Seed {seed} finished. Quality-gated winner: {result['selected']}", flush=True)


def summarize(rows, prepared):
    entries = read_suite(prepared / "suite/suite.json")["splits"]["test"]
    groups = {r["geometry_hash"]: r["group"] for r in entries}
    dedicated = {r["geometry_hash"] for r in entries if r.get("dedicated_straight", False)}
    indexed = defaultdict(dict)
    for row in rows:
        indexed[row["evaluation_speed_limit_m_s"], row["policy_id"]][row["track_hash"]] = row
    if any(set(records) != set(groups) for records in indexed.values()):
        raise ValueError("Assessment track sets disagree")
    summaries, paired = [], []
    for (cap, name), records in indexed.items():
        all_rows = list(records.values())
        for group in ("all", "mixed", "speed", "corner"):
            values = all_rows if group == "all" else [r for r in all_rows if groups[r["track_hash"]] == group]
            straight_rows = [r for r in values if r["track_hash"] in dedicated]
            straights = [r["straight_speed_m_s"] for r in straight_rows if r["straight_speed_m_s"] is not None]
            summaries.append({"cap_m_s": cap, "policy": name, "group": group, "tracks": len(values),
                              "completed": sum(r["completed"] for r in values),
                              "progress": mean(r["progress"] for r in values),
                              "geometric_rmse_m": mean(r["geometry_lateral_rmse_m"] for r in values),
                              "command_jitter": mean(r["command_jitter"] for r in values),
                              "average_forward_m_s": mean(r["average_forward_speed_m_s"] for r in values),
                              "peak_forward_m_s": max(r["peak_forward_speed_m_s"] for r in values),
                              "straight_segments_measured": len(straights), "straight_segments_total": len(straight_rows),
                              "minimum_straight_m_s": min(straights) if straights else None,
                              "straight_goal": bool(straight_rows) and all(r["completed"] and r["straight_speed_m_s"] is not None
                                                                         and r["straight_speed_m_s"] >= .95 for r in straight_rows),
                              "termination_counts": json.dumps(Counter(r["termination"] for r in values), sort_keys=True)})
        if name.startswith("new-"):
            seed = int(name.split("-")[1])
            for ref in (f"reference-{seed:02d}", f"pid-{int(cap*100):03d}"):
                other = indexed[cap, ref]
                shared = [(row, other[key]) for key, row in records.items()
                          if row["completed"] and other[key]["completed"]]
                paired.append({"cap_m_s": cap, "policy": name, "reference": ref, "shared_finishes": len(shared),
                               "new_time_s": mean(a["completion_time_s"] for a, _ in shared) if shared else None,
                               "reference_time_s": mean(b["completion_time_s"] for _, b in shared) if shared else None,
                               "mean_paired_change_s": mean(a["completion_time_s"]-b["completion_time_s"] for a, b in shared) if shared else None})
    return summaries, paired


def assess(output, prepared, plan):
    policies = {f"reference-{seed:02d}": prepared / f"inputs/seed-{seed:02d}-ramp.pkl" for seed in SEEDS}
    outcomes = {}
    for seed in SEEDS:
        directory = output / f"seed-{seed:02d}/training"
        if not (directory / "selection.json").exists():
            raise ValueError(f"Seed {seed} must finish before final assessment")
        result = finalize(directory, prepared, plan, seed)
        outcomes[str(seed)] = result
        if result["selected"]:
            policies[f"new-{seed:02d}"] = directory / "best.pkl"
    for cap in CAPS:
        name = f"pid-{int(cap*100):03d}"
        policies[name] = prepared / "inputs" / f"{name}.json"
    assessment = output / "assessment"
    assessment.mkdir(exist_ok=True)
    freeze_record = {"protocol_sha256": digest(prepared / "protocol.json"), "outcomes": outcomes,
                     "policies": {name: {"file": name + path.suffix, "sha256": digest(path)}
                                  for name, path in policies.items()}, "caps": list(CAPS)}
    frozen = assessment / "frozen"
    if not frozen.exists():
        with TemporaryDirectory(prefix=".freeze-", dir=assessment) as temporary:
            ready = Path(temporary) / "frozen"
            ready.mkdir()
            for name, path in policies.items():
                shutil.copyfile(path, ready / freeze_record["policies"][name]["file"])
            write_json(ready / "selection-freeze.json", freeze_record)
            publish_directory(ready, frozen, "selection-freeze.json")
    if read_json(frozen / "selection-freeze.json") != freeze_record:
        raise ValueError("Frozen assessment inputs changed")
    policies = {name: frozen / record["file"] for name, record in freeze_record["policies"].items()}
    if any(digest(policies[name]) != record["sha256"] for name, record in freeze_record["policies"].items()):
        raise ValueError("A frozen policy file changed")
    destination = assessment / "results"
    if destination.exists():
        status = read_json(destination / "status.json")
        if any(digest(destination / name) != sha for name, sha in status["files"].items()):
            raise ValueError("Saved assessment results changed")
        return
    load_split(prepared / "suite/suite.json", "test")
    tasks = list(evaluation_tasks(prepared, "test", policies, CAPS, plan))
    rows = collect(tasks, WORKERS, 150 * len(policies) * len(CAPS))
    summaries, paired = summarize(rows, prepared)
    with TemporaryDirectory(prefix=".results-", dir=assessment) as temporary:
        ready = Path(temporary) / "results"
        ready.mkdir()
        for name, values in (("test-episodes", rows), ("test-summary", summaries), ("paired-comparisons", paired)):
            if values:
                append_csv(ready / f"{name}.csv", values, list(values[0]))
        write_json(ready / "status.json", {"complete": True, "episodes": len(rows),
                   "files": {path.name: digest(path) for path in ready.glob("*.csv")}})
        publish_directory(ready, destination, "status.json")


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = output / "plots"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for extension in ("png", "svg"):
            fig.savefig(destination / f"{name}.{extension}", dpi=300)
        plt.close(fig)

    logs = [(seed, output / f"seed-{seed:02d}/training") for seed in SEEDS
            if (output / f"seed-{seed:02d}/training/generations.csv").exists()]
    if not logs:
        print("No training logs yet. Run a training command before plotting.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
    for seed, directory in logs:
        rows = read_csv(directory / "generations.csv")
        quality = read_csv(directory / "quality.csv")
        axes[0].plot([int(r["generation"]) for r in rows], [float(r["training_mean"]) for r in rows], label=f"Seed {seed}")
        axes[1].plot([int(r["generation"]) for r in quality], [float(r["geometric_rmse_m"])*1000 for r in quality])
    for axis in axes:
        for generation, _ in SCHEDULE[1:]:
            axis.axvline(generation, color="gray", linestyle=":", linewidth=.7)
        axis.set_xlabel("Generation")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Mean training fitness")
    axes[1].set_ylabel("Validation geometric RMSE (mm)")
    axes[0].legend()
    save(fig, "01-learning-and-tracking")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
    terms = ("progress_term", "completion_term", "elapsed_term", "legacy_term", "speed_term", "jitter_term")
    colors = dict(zip(terms, plt.get_cmap("tab10").colors))
    for seed, directory in logs:
        rows = read_csv(directory / "fitness-components.csv")
        for role, axis in zip(("training", "validation"), axes):
            subset = [row for row in rows if row["role"] == role]
            for term in terms:
                axis.plot([int(r["generation"]) for r in subset], [float(r[term]) for r in subset],
                          color=colors[term], alpha=.6, label=term if seed == logs[0][0] else None)
            axis.set(title=role, xlabel="Generation", ylabel="Mean fitness contribution")
    axes[0].legend(fontsize=7, ncol=3)
    save(fig, "02-fitness-components")
    results = output / "assessment/results/test-summary.csv"
    if results.exists():
        summaries = read_csv(results)
        rows = [row for row in summaries if row["group"] == "all" and float(row["cap_m_s"]) == 1.5
                and (not row["policy"].startswith("pid-") or row["policy"] == "pid-150")]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        for row in rows:
            color = "#0072B2" if row["policy"].startswith("new-") else "#D55E00" if row["policy"].startswith("reference-") else "#009E73"
            axes[0].scatter(float(row["average_forward_m_s"]), 1000*float(row["geometric_rmse_m"]), color=color)
            axes[1].scatter(float(row["command_jitter"]), int(row["completed"]), color=color)
            for axis in axes:
                axis.annotate(row["policy"], axis.collections[-1].get_offsets()[0], xytext=(3, 3),
                              textcoords="offset points", fontsize=6)
        axes[0].set(xlabel="Mean forward speed (m/s)", ylabel="Geometric RMSE (mm)")
        axes[1].set(xlabel="Command jitter", ylabel="Completed test tracks (of 150)")
        save(fig, "03-speed-quality-across-seeds")
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
        for column, cap in enumerate((.5, 1., 1.5)):
            subset = {row["policy"]: row for row in summaries if row["group"] == "all" and float(row["cap_m_s"]) == cap}
            pid = subset[f"pid-{int(cap*100):03d}"]
            for seed in SEEDS:
                reference = subset[f"reference-{seed:02d}"]
                candidate = subset.get(f"new-{seed:02d}")
                for axis, field, factor in ((axes[0, column], "completed", 100/150),
                                            (axes[1, column], "geometric_rmse_m", 1000)):
                    axis.scatter(seed-.1, float(reference[field])*factor, marker="x", color="#D55E00")
                    if candidate:
                        axis.plot([seed-.1, seed+.1], [float(reference[field])*factor, float(candidate[field])*factor],
                                  color="gray", alpha=.5, linewidth=1)
                        axis.scatter(seed+.1, float(candidate[field])*factor, color="#0072B2")
                    if seed == 1:
                        axis.axhline(float(pid[field])*factor, color="#009E73", linestyle=":", linewidth=.6)
                    axis.set(xticks=SEEDS, xlabel="Seed")
                    axis.grid(alpha=.2)
            axes[0, column].set_title(f"{cap:.2f} m/s cap")
            axes[0, column].set(ylim=(0, 102), yticks=(0, 25, 50, 75, 100))
            axes[1, column].set_ylim(bottom=0)
        axes[0, 0].set_ylabel("Completion (%)")
        axes[1, 0].set_ylabel("Geometric RMSE (mm)")
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], color="#0072B2", marker="o", linestyle="", label="New winner"),
                            Line2D([], [], color="#D55E00", marker="x", linestyle="", label="Original ramp"),
                            Line2D([], [], color="#009E73", linestyle=":", label="Cap-specific PID")],
                   loc="outside upper center", ncol=3)
        save(fig, "04-completion-and-tracking-per-seed")
    print(f"Plots saved in {destination}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "prepare", "assess", "plot"):
        commands.add_parser(command)
    training = commands.add_parser("train")
    training.add_argument("--seed", required=True, type=int, choices=SEEDS)
    training.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    prepared = output / "prepared"
    if args.command == "preflight":
        print(json.dumps(verify(prepared) if prepared.exists() else planned_protocol(), indent=2))
        return
    if args.command == "plot":
        plot(output)
        return
    with session_lock(output):
        if args.command == "prepare":
            prepare(output)
            print(f"Prepared {prepared}. Evolution has not started.")
        else:
            plan = verify(prepared)
            if args.command == "train":
                train_seed(output, prepared, plan, args.seed, args.resume)
            else:
                assess(output, prepared, plan)


if __name__ == "__main__":
    mp.freeze_support()
    main()
