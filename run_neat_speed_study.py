"""Prepare, run, and assess the five-seed NEAT speed study one stage at a time."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import csv
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from statistics import mean, stdev
from tempfile import TemporaryDirectory
import zipfile

from line_follower.config import EpisodeConfig, RobotConfig, verify_geometry
from line_follower.controllers import TEMPORAL_INPUT_COUNT, load_neat_config, read_model_artifact
from line_follower.experiments import append_csv, environment_record, snapshot_sources, train, write_json
from line_follower.fitness import FITNESS_VERSION
from line_follower.pid import ResearchPIDConfig
from line_follower.pid_experiments import evaluate_candidate
from line_follower.suites import create_suite, derived_seed, load_split, read_suite
from line_follower.tracks import DIFFICULTIES
from run_corner_training import ROOT, RUNS, SETTINGS as CORNER_SETTINGS
from run_corner_training import digest, protected_tracks, read_json, session_lock
from run_pid_gain_tuning import protected_geometry

OUTPUT = RUNS / "neat-speed-study"
PID_RUN = RUNS / "pid-speed-tuning-20261115"
CORRECTED = ROOT / "research/artifacts/neat-speed-results-corrected/evaluation-freeze.json"
INPUTS = {
    "geometry-freeze.json": RUNS / "28-10-2026-hard/geometry-freeze.json",
    "neat-corner.txt": ROOT / "line_follower/configs/neat-corner.txt",
    "pid-050.json": PID_RUN / "tuned-0.50.json",
    "pid-150.json": PID_RUN / "tuned-1.50.json",
}
CAPS = (0.50, 1.50)
BASELINES = dict(zip(CAPS, ("pid-050", "pid-150"), strict=True))
SEEDS = tuple(range(1, 6))
SUITE_SEED = 2026111502
VALIDATION_COUNT = 30
TEST_COUNT = 90
GENERATIONS = 150
POPULATION = 100
WORKERS = 8
COMPLETION_SPEED_BONUS = 0.40
COMMAND_JITTER_PENALTY = 0.20
LOW_SCHEDULE = ((0, 0.50),)
RAMP_SCHEDULE = ((0, 0.65), (10, 0.80), (20, 1.00), (30, 1.25), (40, 1.50))


def json_values(value):
    return json.loads(json.dumps(value))


def training_settings(seed, phase):
    # The engine uses this root for NEAT and a separate derived track stream.
    phase_seed = derived_seed(SUITE_SEED, "neat-evolution", seed, int(phase == "ramp"))
    settings = replace(CORNER_SETTINGS, seed=phase_seed, generations=GENERATIONS, population=POPULATION,
                       workers=WORKERS, speed_schedule=LOW_SCHEDULE)
    if phase == "ramp":
        return replace(settings, strategy="multi", track_profile="zigzag",
                       speed_schedule=RAMP_SCHEDULE)
    return settings


def software_fingerprints():
    environment = environment_record()
    if environment["dependencies"]["mujoco"] != "4.0.0" or FITNESS_VERSION != "progress-1.2":
        raise ValueError("The study requires MuJoCo 4.0.0 and the progress-1.2 evaluator")
    return {key: environment[key] for key in ("python", "dependencies", "source_sha256", "fitness_version")} | {
        "launcher_sha256": digest(Path(__file__)),
        "shared_launcher_sha256": digest(ROOT / "run_corner_training.py"),
    }


def planned_protocol():
    previous = read_json(PID_RUN / "protocol.json")
    selection = read_json(PID_RUN / "selection-freeze.json")
    corrected = read_json(CORRECTED)
    robot = RobotConfig(**previous["base_robot"])
    episode = EpisodeConfig(**previous["episode"])
    if (previous["episode"] != corrected["episode"] or corrected["evaluator_version"] != FITNESS_VERSION
            or (episode.corridor_half_width, episode.departure_seconds, episode.stall_seconds) != (.10, 2.0, 5.0)):
        raise ValueError("PID tuning and the study must share the corrected episode rules")
    episode = replace(episode, completion_speed_bonus=COMPLETION_SPEED_BONUS,
                      command_jitter_penalty=COMMAND_JITTER_PENALTY)
    verify_geometry(INPUTS["geometry-freeze.json"], robot)
    hashes = {name: digest(path) for name, path in INPUTS.items()}
    for cap, policy in BASELINES.items():
        name = f"{policy}.json"
        expected = selection["selections"][f"tuned-{cap:.2f}"]["sha256"]
        if hashes[name] != expected or ResearchPIDConfig.load(INPUTS[name]).corner_supervision:
            raise ValueError(f"The {cap:.2f} m/s configuration differs from the frozen recovery PID")
    if hashes["pid-050.json"] != corrected["pid_sha256"]:
        raise ValueError("The 0.50 m/s policy must use the original filtered PID gains")
    config = load_neat_config(INPUTS["neat-corner.txt"])
    if (config.genome_config.num_inputs != TEMPORAL_INPUT_COUNT or config.genome_config.num_outputs != 2
            or config.genome_config.activation_default != "tanh"):
        raise ValueError("The corner NEAT controller requires 21 inputs and two tanh outputs")
    return json_values({
        "study": "neat-speed-study-v1",
        "robot": asdict(robot),
        "episode": asdict(episode),
        "software": software_fingerprints(),
        "inputs": hashes,
        "suite_seed": SUITE_SEED,
        "validation_tracks": VALIDATION_COUNT,
        "test_tracks": TEST_COUNT,
        "training": {seed: {phase: asdict(training_settings(seed, phase)) for phase in ("low", "ramp")}
                     for seed in SEEDS},
        "training_track_seed_namespace": "training (derived from each stage seed)",
        "selection": "Highest validation completion, then highest validation fitness, then earliest generation at the final cap",
        "assessment_caps_m_s": CAPS,
        "primary_assessment": {"cap_m_s": 1.50, "metric": "completion", "neat": "ramp", "pid": "pid-150"},
        "pid_baselines": BASELINES,
    })


def verify_prepared(prepared):
    protocol_path = prepared / "protocol.json"
    if not protocol_path.is_file():
        raise ValueError("Prepare the study before running a training or assessment command")
    plan = read_json(protocol_path)
    if plan["software"] != software_fingerprints():
        raise ValueError("The study source or dependencies differ from the prepared protocol")
    if any(digest(prepared / "inputs" / name) != expected for name, expected in plan["inputs"].items()):
        raise ValueError("A frozen study input changed")
    suite = prepared / "suite/suite.json"
    if digest(suite) != plan["suite_sha256"]:
        raise ValueError("The prepared suite manifest changed")
    return plan


def save_preparation(output):
    prepared = output / "prepared"
    if prepared.exists():
        return verify_prepared(prepared)
    plan = planned_protocol()
    print("Preparing shared validation and held-out tracks.", flush=True)
    protected = protected_tracks() | protected_geometry()
    with TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
        ready = Path(temporary) / "prepared"
        inputs = ready / "inputs"
        inputs.mkdir(parents=True)
        for name, source in INPUTS.items():
            shutil.copyfile(source, inputs / name)
        suite = create_suite(ready / "suite", seed=SUITE_SEED, validation_count=VALIDATION_COUNT,
                             test_count=TEST_COUNT, protected_hashes=protected)
        splits = read_suite(suite)["splits"]
        for split, count in (("validation", VALIDATION_COUNT), ("test", TEST_COUNT)):
            entries = splits[split]
            if Counter(e["difficulty"] for e in entries) != {d: count // len(DIFFICULTIES) for d in DIFFICULTIES}:
                raise ValueError(f"Unexpected difficulty distribution in the {split} split")
            if any(e["geometry_hash"] in protected for e in entries):
                raise ValueError(f"The {split} split contains previously used geometry")
        plan["suite_sha256"] = digest(suite)
        write_json(ready / "protocol.json", plan)
        snapshot_sources(ready / "source.zip")
        with zipfile.ZipFile(ready / "source.zip", "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__), Path(__file__).name)
        verify_prepared(ready)
        ready.rename(prepared)
    return plan


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def stage_input_record(prepared, seed, phase, parent):
    return {
        "study_protocol_sha256": digest(prepared / "protocol.json"),
        "training": json_values(asdict(training_settings(seed, phase))),
        "parent_sha256": parent["sha256"] if parent else None,
    }


def selected_model(directory, plan, seed, phase):
    status_path = directory / "status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    if (not status.get("complete") or status.get("next_generation") != GENERATIONS
            or status.get("target_generations") != GENERATIONS):
        raise ValueError(f"Seed {seed} {phase} training is incomplete")
    settings = training_settings(seed, phase)
    rows = read_csv(directory / "generations.csv")
    schedule = [(int(row["generation"]), float(row["wheel_speed_cap_m_s"])) for row in rows]
    if schedule != [(g, settings.speed_at(g, plan["robot"]["max_speed"])) for g in range(GENERATIONS)]:
        raise ValueError(f"Seed {seed} {phase} has an incomplete or unexpected generation schedule")
    run = read_json(directory / "run.json")
    for key, value in {"settings": plan["training"][str(seed)][phase], "robot": plan["robot"],
                       "episode": plan["episode"], "suite_sha256": plan["suite_sha256"]}.items():
        if run.get(key) != value:
            raise ValueError(f"Seed {seed} {phase} run has unexpected {key}")

    best = directory / "best.pkl"
    artifact = read_model_artifact(best)
    metadata = artifact["metadata"]
    start, cap = settings.speed_schedule[-1]
    winner = max(rows[start:], key=lambda row: (float(row["validation_completion_rate"]), float(row["validation_score"])))
    validation_key = [float(winner["validation_completion_rate"]), float(winner["validation_score"])]
    expected = {
        "generation": int(winner["generation"]), "evolution_seed": settings.seed,
        "robot": {**plan["robot"], "max_speed": cap}, "episode": plan["episode"],
        "fitness_version": FITNESS_VERSION, "suite_sha256": plan["suite_sha256"],
        "controller_semantics": "mirrored-sensor-history-v1", "training_backend": "cpu",
        "validation_backend": "cpu", "time_limit_override": None,
    }
    if (any(metadata.get(key) != value for key, value in expected.items())
            or list(metadata.get("validation_key", ())) != validation_key):
        raise ValueError(f"Seed {seed} {phase} best.pkl does not match its validation winner")
    return {"generation": expected["generation"], "cap_m_s": cap, "validation_key": validation_key,
            "evolution_seed": settings.seed, "sha256": digest(best)}


def completed_stage(output, prepared, plan, seed, phase, parent=None):
    seed_dir = output / f"seed-{seed:02d}"
    selection = seed_dir / phase / "selection.json"
    if not selection.is_file():
        raise ValueError(f"Seed {seed} {phase} training is incomplete or needs --resume to finalize")
    if read_json(seed_dir / f"{phase}-start.json") != stage_input_record(prepared, seed, phase, parent):
        raise ValueError(f"Seed {seed} {phase} no longer matches its starting protocol")
    chosen = selected_model(seed_dir / phase, plan, seed, phase)
    if read_json(selection) != chosen:
        raise ValueError(f"Seed {seed} {phase} saved winner changed")
    return chosen


def run_stage(output, prepared, plan, seed, phase, resume):
    seed_dir = output / f"seed-{seed:02d}"
    directory = seed_dir / phase
    parent = completed_stage(output, prepared, plan, seed, "low") if phase == "ramp" else None
    start_path = seed_dir / f"{phase}-start.json"
    start = stage_input_record(prepared, seed, phase, parent)
    checkpoint = directory / "checkpoints/latest.pkl.gz"
    nonempty = directory.exists() and any(directory.iterdir())
    if resume:
        if not start_path.is_file() or read_json(start_path) != start:
            raise ValueError("The saved stage inputs differ from the inputs required for resume")
        if nonempty and not checkpoint.is_file():
            raise ValueError(f"No committed checkpoint exists. Move aside {directory} and retry with --resume.")
    else:
        if start_path.exists() or nonempty:
            raise ValueError("This stage already exists. Use --resume for the original stage.")
        write_json(start_path, start)

    selection = directory / "selection.json"
    if selection.is_file():
        return completed_stage(output, prepared, plan, seed, phase, parent)
    status = directory / "status.json"
    if not status.is_file() or not read_json(status).get("complete"):
        initial_model = seed_dir / "low/best.pkl" if parent and not checkpoint.is_file() else None
        train(directory, prepared / "suite/suite.json", training_settings(seed, phase),
              RobotConfig(**plan["robot"]), EpisodeConfig(**plan["episode"]),
              neat_path=prepared / "inputs/neat-corner.txt" if phase == "low" else None,
              resume=checkpoint if checkpoint.is_file() else None,
              geometry_lock=prepared / "inputs/geometry-freeze.json", live_preview=True,
              initial_model=initial_model)
    chosen = selected_model(directory, plan, seed, phase)
    write_json(selection, chosen)
    print(f"Seed {seed} {phase} winner: generation {chosen['generation']} at {chosen['cap_m_s']:.2f} m/s")
    return chosen


def freeze_assessment(output, prepared, plan):
    """Collect and verify all winners before resolving any test geometry."""
    sources = {}
    policies = {}
    for seed in SEEDS:
        low = completed_stage(output, prepared, plan, seed, "low")
        ramp = completed_stage(output, prepared, plan, seed, "ramp", low)
        for phase, selection in (("low", low), ("ramp", ramp)):
            policy = f"seed-{seed:02d}-{phase}"
            filename = f"{policy}.pkl"
            sources[filename] = output / f"seed-{seed:02d}/{phase}/best.pkl"
            policies[policy] = {**selection, "file": filename, "study_seed": seed, "condition": f"neat-{phase}"}
    for cap, policy in BASELINES.items():
        filename = f"{policy}.json"
        sources[filename] = prepared / "inputs" / filename
        policies[policy] = {"file": filename, "sha256": plan["inputs"][filename],
                            "study_seed": "", "evolution_seed": "", "condition": policy, "selected_for_cap_m_s": cap}
    record = {"frozen_before_test_geometry_load": True, "suite_sha256": plan["suite_sha256"],
              "study_protocol_sha256": digest(prepared / "protocol.json"),
              "caps_m_s": list(CAPS), "policies": policies}
    directory = output / "assessment"
    directory.mkdir(exist_ok=True)
    frozen = directory / "frozen"
    if not frozen.exists():
        with TemporaryDirectory(prefix=".freeze-", dir=directory) as temporary:
            ready = Path(temporary) / "frozen"
            ready.mkdir()
            for filename, source in sources.items():
                shutil.copyfile(source, ready / filename)
            verify_frozen_files(ready, policies)
            write_json(ready / "selection-freeze.json", record)
            ready.rename(frozen)
    if read_json(frozen / "selection-freeze.json") != record:
        raise ValueError("The frozen assessment selections have changed")
    verify_frozen_files(frozen, policies)
    return frozen, policies


def verify_frozen_files(frozen, policies):
    for policy, details in policies.items():
        if digest(frozen / details["file"]) != details["sha256"]:
            raise ValueError(f"Frozen assessment input changed: {policy}")


def assessment_jobs(suite, frozen, policies, plan):
    entries = read_suite(suite)["splits"]["test"]
    for start in range(0, len(entries), 10):
        tracks = [str(suite.parent / entry["path"]) for entry in entries[start:start+10]]
        for cap in CAPS:
            robot = json.dumps({**plan["robot"], "max_speed": cap}, sort_keys=True)
            for policy, details in policies.items():
                saved = frozen / details["file"]
                controller = ({"neat_model": str(saved)} if saved.suffix == ".pkl"
                              else {"parameters": read_json(saved)})
                yield {"candidate": policy, "robot_json": robot, "episode": plan["episode"],
                       "seed": SUITE_SEED, "phase": "test", "tracks": tracks,
                       "slot_offset": start, "geometric_tracking": True, **controller}


def report(results, destination):
    groups = {}
    for row in results:
        groups.setdefault((row["evaluation_speed_limit_m_s"], row["policy_id"]), {})[row["track_hash"]] = row
    summaries, failures, comparisons = [], [], []
    for (cap, policy), tracks in groups.items():
        rows = list(tracks.values())
        summary = {"cap_m_s": cap, "policy": policy, "condition": rows[0]["condition"],
                   "study_seed": rows[0]["study_seed"], "tracks": len(rows)}
        summaries.append({**summary, "completed": sum(row["completed"] for row in rows),
                          "completion_rate": mean(row["completed"] for row in rows),
                          **{name: mean(row[name] for row in rows) for name in
                             ("progress", "geometry_lateral_rmse_m", "geometry_lateral_mae_m",
                              "average_forward_speed_m_s", "command_jitter")}})
        failures.extend({**summary, "termination": reason, "episodes": count}
                        for reason, count in sorted(Counter(row["termination"] for row in rows).items()))
    spread = []
    for cap in CAPS:
        for condition in ("neat-low", "neat-ramp"):
            rates = [row["completion_rate"] for row in summaries
                     if row["cap_m_s"] == cap and row["condition"] == condition]
            spread.append({"cap_m_s": cap, "condition": condition, "seeds": len(rates),
                           "mean_completion_rate": mean(rates), "sample_sd": stdev(rates),
                           "min_completion_rate": min(rates), "max_completion_rate": max(rates)})
        pairs = [(BASELINES[CAPS[1]], BASELINES[CAPS[0]], "")]
        for seed in SEEDS:
            low, ramp = f"seed-{seed:02d}-low", f"seed-{seed:02d}-ramp"
            pairs.extend([(ramp, low, seed), (low, BASELINES[cap], seed), (ramp, BASELINES[cap], seed)])
        for policy, reference, seed in pairs:
            left, right = groups[cap, policy], groups[cap, reference]
            paired = [(row, right[track]) for track, row in left.items()]
            shared = [(a["completion_time_s"], b["completion_time_s"])
                      for a, b in paired if a["completed"] and b["completed"]]
            comparisons.append({"cap_m_s": cap, "study_seed": seed, "policy": policy, "reference": reference,
                                "tracks": len(paired), "both_finished": len(shared),
                                "policy_only": sum(a["completed"] and not b["completed"] for a, b in paired),
                                "reference_only": sum(b["completed"] and not a["completed"] for a, b in paired),
                                "policy_mean_shared_time_s": mean(a for a, _ in shared) if shared else "",
                                "reference_mean_shared_time_s": mean(b for _, b in shared) if shared else ""})
    for name, rows in (("test-episodes", results), ("test-summary", summaries), ("terminations", failures),
                       ("seed-spread", spread), ("paired-comparisons", comparisons)):
        append_csv(destination / f"{name}.csv", rows, list(rows[0]))


def assess(output, prepared, plan):
    frozen, policies = freeze_assessment(output, prepared, plan)
    destination = frozen.parent / "results"
    if destination.exists():
        status = read_json(destination / "status.json")
        if any(digest(destination / file) != sha for file, sha in status["files"].items()):
            raise ValueError("The saved final assessment results changed")
        return destination
    suite = prepared / "suite/suite.json"
    tracks = load_split(suite, "test")
    jobs = list(assessment_jobs(suite, frozen, policies, plan))
    expected = {(cap, policy, track.geometry_hash) for cap in CAPS for policy in policies for track in tracks}
    results = []
    seen = set()
    print(f"Assessing {len(policies)} frozen controllers at both caps on {len(tracks)} held-out tracks.", flush=True)
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for job, rows in zip(jobs, pool.imap(evaluate_candidate, jobs), strict=True):
            policy = job["candidate"]
            details = policies[policy]
            for slot, row in enumerate(rows, job["slot_offset"]):
                key = (row["evaluation_speed_limit_m_s"], policy, row["track_hash"])
                if key in seen or key not in expected or row["track_hash"] != tracks[slot].geometry_hash:
                    raise RuntimeError("Evaluation returned duplicate or unexpected test geometry")
                seen.add(key)
                results.append({**row, "condition": details["condition"], "policy_id": policy, "candidate": policy,
                                "study_seed": details["study_seed"], "evolution_seed": details["evolution_seed"],
                                "tuning_seed": "", "evaluation_seed": SUITE_SEED, "track_slot": slot})
            if len(results) % 180 == 0 or len(results) == len(expected):
                print(f"Final assessment {len(results)}/{len(expected)} episodes", flush=True)
    if seen != expected:
        raise RuntimeError("The final test set is incomplete")
    verify_frozen_files(frozen, policies)
    verify_prepared(prepared)
    errors = [row for row in results if row["error"]]
    if errors:
        append_csv(frozen.parent / "technical-failures.csv", errors, list(errors[0]))
        raise RuntimeError("Final assessment had technical errors. See assessment/technical-failures.csv.")
    with TemporaryDirectory(prefix=".results-", dir=frozen.parent) as temporary:
        ready = Path(temporary) / "results"
        report(results, ready)
        write_json(ready / "status.json", {"complete": True, "episodes": len(results),
                   "files": {path.name: digest(path) for path in sorted(ready.glob("*.csv"))}})
        ready.rename(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Study directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Verify settings and fingerprints without starting evolution")
    commands.add_parser("prepare", help="Save the shared suite and frozen inputs without starting evolution")
    for phase in ("low", "ramp"):
        command = commands.add_parser(f"train-{phase}", help=f"Start only the {phase} training stage")
        command.add_argument("--seed", type=int, required=True, choices=SEEDS)
        command.add_argument("--resume", action="store_true")
    commands.add_parser("assess", help="Evaluate all frozen winners after all five seeds finish")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    prepared = output / "prepared"
    if args.command == "preflight":
        plan = verify_prepared(prepared) if prepared.exists() else planned_protocol()
        print(json.dumps({**plan, "output": str(output), "training_started": False}, indent=2))
        return
    with session_lock(output):
        if args.command == "prepare":
            save_preparation(output)
            print(f"Study inputs prepared at {prepared}. Evolution has not started.")
            return
        plan = verify_prepared(prepared)
        if args.command == "assess":
            print(f"Final assessment saved at {assess(output, prepared, plan)}")
        else:
            run_stage(output, prepared, plan, args.seed, args.command.removeprefix("train-"), args.resume)


if __name__ == "__main__":
    mp.freeze_support()
    main()
