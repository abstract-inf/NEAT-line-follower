"""Local 1 m/s filtered-PID smoothness experiments; frozen NEAT inputs stay intact."""

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.episodes import Episode
from line_follower.experiments import append_csv, snapshot_sources
from line_follower.fast_study import centered_course, speed_course
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.pid_experiments import _mask, _simulation, evaluate_candidate
from line_follower.suites import corner_track, create_suite, derived_seed, load_split, read_suite
from line_follower.tracks import GeneratorConfig, Section, TrackGenerationError, generate_track, validate_track

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs/pid-100-smoothness-20261119"
BASELINE = ROOT / "runs/pid-speed-followup-20261116/tuned-1.00.json"
SUITE_SEED = 2026111902


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(seed=2026111901, radius=2., sign=1, angle=np.pi/3):
    return centered_course([Section("straight", length=1., width=.020),
                            Section("arc", radius=radius, angle=sign*angle, width=.020),
                            Section("straight", length=1., width=.020)],
                           f"smooth-curve-{seed}", seed, "development")


def trace_metrics(path, lower=1.4, upper=2.7):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return steady_metrics(rows)


def steady_metrics(rows):
    yaw = np.array([float(row["yaw_rate_rad_s"]) for row in rows])
    if len(yaw) < 21:
        raise ValueError("Too few steady-curve samples")
    residual = yaw[7:-7] - np.convolve(yaw, np.ones(15)/15, mode="valid")
    actions = np.array([[float(row["action_L"]), float(row["action_R"])] for row in rows])
    return {"samples": len(rows), "yaw_oscillation_rms_rad_s": float(np.sqrt(np.mean(residual**2))),
            "yaw_acceleration_rms_rad_s2": float(np.sqrt(np.mean(np.diff(yaw)**2)))*60,
            "curve_command_jitter": float(np.mean(np.sum(np.diff(actions, axis=0)**2, axis=1)/8)),
            "mean_forward_m_s": float(np.mean([float(row["forward_speed_m_s"]) for row in rows]))}


def probe(config_path, output, quiet=False):
    output.mkdir(parents=True, exist_ok=True)
    track = curve()
    path = output / "diagnostic-curve.json"
    if not path.exists():
        track.save(path)
    protocol = read_json(ROOT / "runs/neat-fast-study/prepared/protocol.json")
    name = "baseline" if Path(config_path).resolve() == BASELINE else Path(config_path).stem
    result, = evaluate_candidate({"robot_json": json.dumps({**protocol["robot"], "max_speed": 1.}),
                                  "episode": protocol["episode"], "candidate": name,
                                  "parameters": read_json(config_path), "tracks": [str(path)],
                                  "phase": "probe", "seed": 2026111901, "output": str(output),
                                  "trace_slots": [0], "detailed_metrics": True})
    with Path(result["telemetry"]).open(newline="", encoding="utf-8") as stream:
        telemetry = list(csv.DictReader(stream))
    steady = [row for row in telemetry if 1.4 <= float(row["progress"])*track.route.length <= 2.7]
    steady_path = output / f"{name}-steady-curve.csv"
    with steady_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(steady[0]))
        writer.writeheader()
        writer.writerows(steady)
    metrics = trace_metrics(steady_path)
    record = {"config_sha256": digest(config_path), "track_hash": track.geometry_hash,
              "completed": result["completed"], "rmse_m": result["geometry_lateral_rmse_m"], **metrics}
    write_json(output / f"probe-{name}.json", record)
    baseline_path = output / "probe-baseline.json"
    baseline = read_json(baseline_path) if baseline_path.exists() else record
    passed = result["completed"] and record["yaw_oscillation_rms_rad_s"] <= .8*baseline["yaw_oscillation_rms_rad_s"]
    if not quiet:
        print(json.dumps({**record, "target": "At least 20% less steady-curve yaw oscillation than the saved baseline",
                          "smoothness_target_passed": passed}, indent=2), flush=True)
    return 0 if passed else 1


def screen_task(task):
    path, output = task
    probe(path, output, quiet=True)
    record = read_json(output / f"probe-{path.stem}.json")
    return {"candidate": path.stem, **read_json(path), **record}


def screen(output, workers):
    original = read_json(BASELINE)
    paths = []
    for field, multipliers in (("kp", (.95, .9, .85)), ("kd", (1.05, 1.1, .9, .75)),
                               ("derivative_filter_s", (1.5, 2., 3.)), ("ki", (.5, 0.))):
        for factor in multipliers:
            path = output / "probe-configs" / f"{field}-{factor:.2f}.json"
            write_json(path, {**original, field: original[field]*factor})
            paths.append(path)
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = list(pool.imap(screen_task, [(path, output) for path in paths]))
    with (output / "one-factor-screen.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in sorted(rows, key=lambda row: row["yaw_oscillation_rms_rad_s"]):
        print(f"{row['candidate']}: yaw oscillation={row['yaw_oscillation_rms_rad_s']:.4f} rad/s, "
              f"curve jitter={row['curve_command_jitter']:.6f}, speed={row['mean_forward_m_s']:.4f} m/s", flush=True)


def course_windows(track):
    position, windows = 0., []
    for section in track.sections:
        length = section.length if section.kind == "straight" else abs(section.radius*section.angle) if section.kind == "arc" else 0.
        if section.kind == "arc":
            windows.append((position+.25*length, position+.75*length))
        elif section.kind == "straight" and length >= 3:
            windows.append((position+length/2-1, position+length/2+1))
        position += length
    return windows


def smooth_evaluate(task):
    """Use the unchanged episode/controller; add steady-section physical metrics."""
    simulation = _simulation(task["robot_json"])
    mask = _mask(task["track"])
    controller = ResearchPIDController(ResearchPIDConfig(**task["parameters"]), simulation.config)
    episode = Episode(simulation, mask.track, controller, EpisodeConfig(**task["episode"]), mask, detailed=True)
    windows = task.get("windows", [])
    samples = [[] for _ in windows]
    trace = [] if task.get("telemetry") else None
    while not episode.done:
        row = episode.advance(telemetry=bool(windows) or trace is not None)
        if row:
            if trace is not None:
                trace.append({**row, **controller.debug})
            distance = row["progress"] * mask.track.route.length
            for index, (lower, upper) in enumerate(windows):
                if lower <= distance <= upper:
                    samples[index].append(row)
    result = asdict(episode.result())
    if result["error"] or result["geometry_lateral_rmse_m"] is None:
        raise ValueError(f"Technical evaluation failure: {result['error']}")
    if trace is not None:
        path = Path(task["telemetry"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
    metrics = [steady_metrics(rows) for rows in samples if len(rows) >= 21]
    return {"candidate": task["candidate"], "group": task["group"], **result,
            "steady_yaw_rms_rad_s": mean(r["yaw_oscillation_rms_rad_s"] for r in metrics) if metrics else None,
            "steady_command_jitter": mean(r["curve_command_jitter"] for r in metrics) if metrics else None,
            "steady_forward_m_s": mean(r["mean_forward_m_s"] for r in metrics) if metrics else None}


def prepare_courses(output):
    path = output / "suite/suite.json"
    if path.exists():
        return path
    old = read_suite(ROOT / "runs/neat-fast-study/prepared/suite/suite.json")
    protected = set(old["protected_geometry_hashes"])
    protected.update(row["geometry_hash"] for rows in old["splits"].values() for row in rows)
    for batch in (ROOT / "runs/neat-fast-study").glob("seed-*/training/tracks/generation-*/batch.json"):
        protected.update(read_json(batch)["hashes"])
    path = create_suite(path.parent, seed=SUITE_SEED, validation_count=30, test_count=90, protected_hashes=protected)
    suite = read_suite(path)
    used = protected | {row["geometry_hash"] for rows in suite["splits"].values() for row in rows}
    for rows in suite["splits"].values():
        for row in rows:
            row["group"] = "mixed"
            row["windows"] = []
    for split, group, count in (("training", "speed", 20), ("training", "mixed", 20), ("training", "corner", 10),
                                ("validation", "speed", 20), ("validation", "corner", 10),
                                ("test", "speed", 30), ("test", "corner", 30)):
        for index in range(count):
            for attempt in range(100):
                seed = derived_seed(SUITE_SEED, f"{split}-{group}", index, attempt)
                try:
                    if group == "speed":
                        track = speed_course(seed, index*20//count, split)
                    elif group == "corner":
                        track = replace(corner_track(seed, 3, index % 2, "advanced", GeneratorConfig()), split=split)
                    else:
                        track = replace(generate_track(seed, "intermediate" if index % 2 else "advanced"), split=split)
                except TrackGenerationError:
                    continue
                if not validate_track(track) and track.geometry_hash not in used:
                    break
            else:
                raise ValueError("Could not generate a distinct valid course")
            used.add(track.geometry_hash)
            relative = f"{split}/{group}-{index:03d}.json"
            track.save(path.parent / relative)
            suite["splits"][split].append({"path": relative, "geometry_hash": track.geometry_hash,
                                            "group": group, "difficulty": track.difficulty,
                                            "windows": course_windows(track) if group == "speed" else []})
    write_json(path, suite)
    return path


def batch(pool, candidates, entries, suite, protocol, directory, phase, traces=False):
    tasks = [{"candidate": key, "parameters": parameters, "track": str(suite.parent / entry["path"]),
              "group": entry["group"], "windows": entry.get("windows", []),
              "robot_json": json.dumps({**protocol["robot"], "max_speed": 1.}, sort_keys=True),
              "episode": protocol["episode"],
              "telemetry": str(directory / "telemetry" / phase / f"{key}-{index:03d}.csv") if traces else None}
             for entry in entries for key, parameters in candidates.items()]
    results = defaultdict(list)
    path = directory / f"{phase}-episodes.csv"
    if path.exists():
        raise FileExistsError(f"Phase already exists: {phase}")
    for index, row in enumerate(pool.imap(smooth_evaluate, tasks), 1):
        results[row["candidate"]].append(row)
        append_csv(path, [row], list(row))
        if index % 250 == 0 or index == len(tasks):
            print(f"{phase}: {index}/{len(tasks)} episodes", flush=True)
    return dict(results)


def comparison(rows, reference):
    old = {row["track_hash"]: row for row in reference}
    shared = [row for row in rows if row["completed"] and old[row["track_hash"]]["completed"]]
    baseline_shared = [old[row["track_hash"]] for row in shared]
    steady = [row for row in rows if row["steady_yaw_rms_rad_s"] is not None]
    old_steady = [old[row["track_hash"]] for row in steady]
    average = lambda values, field: mean(row[field] for row in values)
    group_counts = {group: sum(row["completed"] for row in rows if row["group"] == group) for group in ("mixed", "speed", "corner")}
    old_counts = {group: sum(row["completed"] for row in reference if row["group"] == group) for group in group_counts}
    yaw = average(steady, "steady_yaw_rms_rad_s")
    old_yaw = average(old_steady, "steady_yaw_rms_rad_s")
    paired_ratio = average(shared, "completion_time_s") / average(baseline_shared, "completion_time_s") if shared else 1e9
    rmse = average(shared, "geometry_lateral_rmse_m") if shared else 1e9
    old_rmse = average(baseline_shared, "geometry_lateral_rmse_m") if shared else 0.
    reasons = []
    if any(group_counts[group] < old_counts[group] for group in group_counts):
        reasons.append("group_completion")
    if any(not row["completed"] and old[row["track_hash"]]["completed"] for row in rows):
        reasons.append("lost_baseline_finish")
    if paired_ratio > 1.03:
        reasons.append("finish_time")
    if rmse > 1.05*old_rmse:
        reasons.append("shared_tracking")
    if average(rows, "command_jitter") > average(reference, "command_jitter"):
        reasons.append("command_jitter")
    if yaw > .8*old_yaw:
        reasons.append("steady_yaw_target")
    return {"candidate": rows[0]["candidate"], "completed": sum(group_counts.values()),
            **{f"{group}_completed": value for group, value in group_counts.items()},
            "progress": average(rows, "progress"), "shared_finishes": len(shared),
            "paired_time_ratio": paired_ratio, "paired_new_time_s": average(shared, "completion_time_s") if shared else None,
            "paired_old_time_s": average(baseline_shared, "completion_time_s") if shared else None,
            "rmse_all_mm": 1000*average(rows, "geometry_lateral_rmse_m"), "rmse_shared_mm": 1000*rmse,
            "command_jitter": average(rows, "command_jitter"), "steady_yaw_rms_rad_s": yaw,
            "steady_yaw_ratio": yaw/old_yaw, "mean_forward_m_s": average(rows, "average_forward_speed_m_s"),
            "eligible": not reasons, "reasons": ";".join(reasons)}


def run(output, workers):
    started = time.perf_counter()
    if (output / "protocol.json").exists():
        raise FileExistsError("Choose a new output directory for a new tuning run")
    protocol = read_json(ROOT / "runs/neat-fast-study/prepared/protocol.json")
    original = read_json(BASELINE)
    assert not original["corner_supervision"]
    suite = prepare_courses(output)
    manifest = read_suite(suite)
    candidates = {"baseline": original}
    seen = {tuple(original[name] for name in ("kp", "kd", "derivative_filter_s"))}
    for p, d, smoothing in product((.85, .9, .95, 1., 1.05), (.8, .9, 1., 1.05, 1.1), (.03, .045, .06, .08)):
        parameters = {**original, "kp": original["kp"]*p, "kd": original["kd"]*d, "derivative_filter_s": smoothing}
        triple = tuple(parameters[name] for name in ("kp", "kd", "derivative_filter_s"))
        if triple not in seen:
            key = f"local-{len(candidates):03d}"
            candidates[key] = parameters
            seen.add(triple)
    write_json(output / "candidates.json", candidates)
    write_json(output / "protocol.json", {"study": "pid-100-smoothness-v1", "seed": SUITE_SEED,
               "robot": {**protocol["robot"], "max_speed": 1.}, "episode": protocol["episode"],
               "baseline_path": str(BASELINE), "baseline_sha256": digest(BASELINE), "suite_sha256": digest(suite),
               "candidate_count": len(candidates), "changed_parameters": ["kp", "kd", "derivative_filter_s"],
               "fixed_base_command": original["base"], "workers": workers, "source_sha256": digest(__file__),
               "selection": "Retain baseline finishes and group completion; paired time <=1.03 baseline; shared RMSE <=1.05 baseline; command jitter <=baseline; steady yaw oscillation <=0.8 baseline. Rank by completion, then lower yaw oscillation, then jitter, then paired time.",
               "test_rule": "Freeze a validation selection before loading this PID-only test split; no NEAT final-test evaluation."})
    snapshot_sources(output / "source.zip")
    with ZipFile(output / "source.zip", "a", compression=ZIP_DEFLATED) as archive:
        archive.write(__file__, Path(__file__).name)
    entries = manifest["splits"]["training"]
    coarse = [entry for entry in entries if entry["path"] in {"training/speed-000.json", "training/speed-006.json", "training/speed-007.json", "training/speed-010.json", "training/speed-012.json", "training/speed-013.json", "training/speed-016.json", "training/speed-019.json", "training/corner-000.json", "training/corner-001.json", "training/mixed-000.json", "training/mixed-001.json"}]
    assert len(coarse) == 12
    rank = lambda info: (info["eligible"], info["completed"], -info["steady_yaw_rms_rad_s"], -info["command_jitter"], -info["paired_time_ratio"])
    with mp.get_context("spawn").Pool(workers) as pool:
        coarse_results = batch(pool, candidates, coarse, suite, protocol, output, "coarse")
        leaders = sorted(candidates, key=lambda key: rank(comparison(coarse_results[key], coarse_results["baseline"])), reverse=True)
        finalists = dict.fromkeys(["baseline", *leaders[:12]])
        expanded_results = batch(pool, {key: candidates[key] for key in finalists}, entries, suite, protocol, output, "expanded")
        expanded_summary = {key: comparison(rows, expanded_results["baseline"]) for key, rows in expanded_results.items()}
        leaders = sorted(expanded_summary, key=lambda key: rank(expanded_summary[key]), reverse=True)
        finalists = dict.fromkeys(["baseline", *leaders[:6]])
        validation_results = batch(pool, {key: candidates[key] for key in finalists}, manifest["splits"]["validation"], suite, protocol, output, "validation")
        summaries = [comparison(rows, validation_results["baseline"]) for rows in validation_results.values()]
        append_csv(output / "validation-summary.csv", summaries, list(summaries[0]))
        eligible = [row for row in summaries if row["eligible"]]
        if not eligible:
            write_json(output / "status.json", {"complete": True, "selected": None, "validation": summaries})
            print("No candidate met the declared validation criteria.", flush=True)
            return
        selected = max(eligible, key=rank)
        name = selected["candidate"]
        write_json(output / "tuned-1.00.json", candidates[name])
        write_json(output / "selection-freeze.json", {"candidate": name, "config": "tuned-1.00.json", "sha256": digest(output/"tuned-1.00.json"), "validation": selected, "frozen_before_test": True})
        load_split(suite, "test")
        test_results = batch(pool, {"baseline": original, "selected": candidates[name]}, manifest["splits"]["test"], suite, protocol, output, "test")
        summaries = [comparison(rows, test_results["baseline"]) for rows in test_results.values()]
        append_csv(output / "test-summary.csv", summaries, list(summaries[0]))
    assert digest(BASELINE) == read_json(output / "protocol.json")["baseline_sha256"]
    write_json(output / "status.json", {"complete": True, "selected": name, "wall_seconds": time.perf_counter()-started})
    print(json.dumps({"selected": name, "parameters": candidates[name], "test_summary": summaries}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "screen", "run", "view"))
    parser.add_argument("--config", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--track", type=Path, help="Saved track for view; omitted uses the diagnostic curve")
    args = parser.parse_args()
    if args.command == "probe":
        raise SystemExit(probe(args.config, args.output.resolve()))
    if args.command == "screen":
        screen(args.output.resolve(), args.workers)
    elif args.command == "run":
        run(args.output.resolve(), args.workers)
    else:
        from line_follower.tracks import TrackSpec
        from line_follower.viewer import view_tracks
        protocol = read_json(ROOT / "runs/neat-fast-study/prepared/protocol.json")
        robot = RobotConfig(**{**protocol["robot"], "max_speed": 1.})
        controller = ResearchPIDController(ResearchPIDConfig(**read_json(args.config)), robot)
        track = TrackSpec.load(args.track) if args.track else curve()
        view_tracks([track], robot, controller, episode_config=EpisodeConfig(**protocol["episode"]))


if __name__ == "__main__":
    mp.freeze_support()
    main()
