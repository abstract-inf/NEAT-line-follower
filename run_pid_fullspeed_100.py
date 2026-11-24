"""Tune a full-cruise PID, verify near-1 m/s travel, and compare frozen policies."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from statistics import mean
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import compare_neat_pid_100 as comparison
from line_follower.experiments import append_csv, snapshot_sources, write_json
from line_follower.fast_study import centered_course, speed_course
from line_follower.pid import ResearchPIDConfig
from line_follower.suites import corner_track, derived_seed, read_suite
from line_follower.tracks import GeneratorConfig, Section, TrackGenerationError, generate_track, validate_track
from run_corner_training import digest, protected_tracks, read_json, session_lock
from run_neat_fast_study import publish_directory, software
from run_pid_smoothness_tuning import smooth_evaluate

ROOT = Path(__file__).resolve().parent
TUNING = ROOT / "runs/pid-fullspeed-100-20261119"
OUTPUT = ROOT / "runs/neat134-vs-pid100-fullspeed-20261119"
SLOW_PID = ROOT / "runs/pid-100-smoothness-20261119/tuned-1.00.json"
FIELDS = ("kp", "ki", "kd", "derivative_filter_s", "slowdown_gain")
BOUNDS = ((.2, 3.), (0., .25), (.005, .20), (.02, .16), (.3, 10.))
TUNING_SEED = 2026111904
TEST_SEED = 2026111905


def protected_geometry():
    hashes = protected_tracks()
    for path in (ROOT/"runs").rglob("suite.json"):
        hashes.update(read_suite(path).get("protected_geometry_hashes", []))
    for path in (ROOT/"runs/long-course-speed-challenge/track.json",
                 ROOT/"runs/pid-100-smoothness-20261119/diagnostic-curve.json"):
        hashes.add(comparison.TrackSpec.load(path).geometry_hash)
    return hashes


def make_suite(directory, seed, splits):
    path = directory / "suite.json"
    if path.exists():
        return path
    protected = protected_geometry()
    used = set(protected)
    manifest = {"suite_version": 1, "generator_version": "1.0", "root_seed": seed,
                "protected_geometry_hashes": sorted(protected), "splits": {}}
    for split, counts in splits.items():
        entries = []
        for group, count in counts.items():
            for index in range(count):
                for attempt in range(100):
                    track_seed = derived_seed(seed, split, group, index, attempt)
                    try:
                        if group == "speed":
                            track = speed_course(track_seed, index*20//count, "test" if split == "training" else split)
                        elif group == "corner":
                            track = corner_track(track_seed, 3, index % 2, "advanced", GeneratorConfig())
                        else:
                            difficulty = ("beginner", "intermediate", "advanced")[index % 3] if split != "training" else ("intermediate", "advanced")[index % 2]
                            track = generate_track(track_seed, difficulty)
                        track = replace(track, split=split)
                    except TrackGenerationError:
                        continue
                    if track.geometry_hash not in used and not validate_track(track):
                        break
                else:
                    raise ValueError("Could not generate distinct valid geometry")
                used.add(track.geometry_hash)
                relative = f"{split}/{group}-{index:03d}.json"
                track.save(directory / relative)
                entries.append({"path": relative, "geometry_hash": track.geometry_hash,
                                "group": group, "difficulty": track.difficulty, "features": track.features,
                                "dedicated_straight": group == "speed" and all(s.kind == "straight" for s in track.sections),
                                "windows": comparison.steady_windows(track) if group == "speed" else [],
                                "replay": f"{group}-{index:03d}" in comparison.REPLAY})
        manifest["splits"][split] = entries
    write_json(path, manifest)
    return path


def probe(output):
    output.mkdir(parents=True, exist_ok=True)
    base = read_json(SLOW_PID)
    protocol = read_json(ROOT/"runs/neat-fast-study/prepared/protocol.json")
    track = centered_course([Section("straight", length=6., width=.02)], "fullspeed-feasibility", 2026111904, "development")
    path = output/"feasibility-straight.json"
    if not path.exists():
        track.save(path)
    rows = []
    for name, parameters in (("previous-base-075", base), ("full-cruise-unretuned", {**base, "base": 1.})):
        row = smooth_evaluate({"candidate": name, "parameters": parameters, "track": str(path),
                               "group": "speed", "windows": [(2., 4.)],
                               "robot_json": json.dumps({**protocol["robot"], "max_speed": 1.}, sort_keys=True),
                               "episode": protocol["episode"], "telemetry": str(output/"telemetry"/f"{name}.csv")})
        rows.append(row)
    write_json(output/"feasibility.json", rows)
    print(json.dumps([{key: row[key] for key in ("candidate", "completed", "mean_action_L", "mean_action_R", "straight_speed_m_s", "peak_forward_speed_m_s")}
                      for row in rows], indent=2), flush=True)
    if not (rows[0]["straight_speed_m_s"] < .8 and rows[1]["completed"] and rows[1]["straight_speed_m_s"] >= .95):
        raise ValueError("Full-cruise feasibility check failed")


def candidates():
    old = read_json(SLOW_PID)
    full = {**old, "base": 1.}
    policies = {"previous-base-075": old, "full-cruise-reference": full}
    seen = {tuple(full[field] for field in FIELDS)}

    def add(values, prefix="gain"):
        point = tuple(float(np.clip(value, *bounds)) for value, bounds in zip(values, BOUNDS, strict=True))
        if point in seen:
            return None
        seen.add(point)
        key = f"{prefix}-{len(policies):03d}"
        parameters = {**full, **dict(zip(FIELDS, point, strict=True))}
        ResearchPIDConfig(**parameters)
        policies[key] = parameters
        return key

    for p in (.85, 1.25, 2., 3.):
        for d in (.6, 1., 1.4, 2.):
            for smoothing in (.06, .10):
                for slowdown in (1.5, 4.):
                    add((full["kp"]*p, full["ki"], full["kd"]*d, smoothing, slowdown))
    for field, values in (("kp", (.3, .6, 1., 1.8)), ("ki", (0., .04, .16)),
                          ("kd", (.025, .075, .12)), ("derivative_filter_s", (.04, .06, .12)),
                          ("slowdown_gain", (1., 2., 4., 6.))):
        for value in values:
            point = [full[name] if name != field else value for name in FIELDS]
            add(point, "axis")
    prior = read_json(ROOT/"runs/pid-speed-followup-20261116/tuned-1.50.json")
    add((prior["kp"], prior["ki"], prior["kd"], .08, 2.), "prior150")
    rng = np.random.default_rng(derived_seed(TUNING_SEED, "search"))
    for index in range(64):
        add((np.exp(rng.uniform(np.log(.25), np.log(2.5))), 0. if index % 3 == 0 else rng.uniform(0, .2),
             np.exp(rng.uniform(np.log(.015), np.log(.15))), rng.uniform(.04, .15),
             np.exp(rng.uniform(np.log(.5), np.log(8.)))), "sample")
    return policies, add


def batch(pool, policies, entries, suite, protocol, output, phase):
    path = output/f"{phase}-episodes.csv"
    if path.exists():
        raise FileExistsError(f"Phase exists: {phase}; choose a separate output for a new search")
    tasks = [{"candidate": name, "parameters": parameters, "track": str(suite.parent/entry["path"]),
              "group": entry["group"], "windows": entry["windows"],
              "robot_json": json.dumps({**protocol["robot"], "max_speed": 1.}, sort_keys=True), "episode": protocol["episode"]}
             for entry in entries for name, parameters in policies.items()]
    records = defaultdict(list)
    for index, row in enumerate(pool.imap(smooth_evaluate, tasks), 1):
        records[row["candidate"]].append(row)
        append_csv(path, [row], list(row))
        if index % 300 == 0 or index == len(tasks):
            print(f"{phase}: {index}/{len(tasks)} episodes", flush=True)
    return dict(records)


def metrics(rows):
    straights = [row for row in rows if row["features"] == "speed-straight"]
    feasible = bool(straights) and all(row["completed"] and row["straight_speed_m_s"] is not None and row["straight_speed_m_s"] >= .95 for row in straights)
    steady = [row["steady_yaw_rms_rad_s"] for row in rows if row["steady_yaw_rms_rad_s"] is not None]
    return {"candidate": rows[0]["candidate"], "completed": sum(row["completed"] for row in rows),
            "tracks": len(rows), "progress": mean(row["progress"] for row in rows),
            "speed_feasible": feasible, "minimum_straight_m_s": min(row["straight_speed_m_s"] for row in straights if row["straight_speed_m_s"] is not None) if any(row["straight_speed_m_s"] is not None for row in straights) else None,
            "failure_penalized_mean_time_s": mean(row["completion_time_s"] if row["completed"] else row["time_limit_s"] for row in rows),
            "command_jitter": mean(row["command_jitter"] for row in rows),
            "steady_yaw_rms_rad_s": mean(steady) if steady else 1e9,
            "geometric_rmse_mm": 1000*mean(row["geometry_lateral_rmse_m"] for row in rows)}


def rank(record):
    return (record["speed_feasible"], record["completed"], -record["failure_penalized_mean_time_s"],
            -record["command_jitter"], -record["geometric_rmse_mm"])


def promote(summaries, limit=12):
    eligible = [row for row in summaries.values() if row["speed_feasible"]]
    if not eligible:
        raise ValueError("No full-speed-capable candidate")
    maximum = max(row["completed"] for row in eligible)
    fastest = sorted(eligible, key=rank, reverse=True)
    smoothest = sorted((row for row in eligible if row["completed"] == maximum),
                       key=lambda row: (row["steady_yaw_rms_rad_s"], row["command_jitter"]))
    return list(dict.fromkeys([*(row["candidate"] for row in fastest[:8]),
                               *(row["candidate"] for row in smoothest[:4]),
                               *(row["candidate"] for row in fastest)]))[:limit]


def tune(output, workers):
    if (output/"selection-freeze.json").exists():
        record = read_json(output/"selection-freeze.json")
        if (digest(output/record["config"]) != record["sha256"]
                or read_json(output/"protocol.json")["source_sha256"] != digest(Path(__file__))):
            raise ValueError("Frozen full-speed tune changed")
        return output/record["config"]
    if (output/"protocol.json").exists():
        raise FileExistsError("An incomplete search exists; preserve it and choose another output")
    probe(output)
    started = time.perf_counter()
    protocol = read_json(ROOT/"runs/neat-fast-study/prepared/protocol.json")
    suite = make_suite(output/"suite", TUNING_SEED, {"training": {"mixed": 20, "speed": 20, "corner": 10},
                                                    "validation": {"mixed": 30, "speed": 20, "corner": 10}})
    policies, add = candidates()
    write_json(output/"candidates.json", policies)
    write_json(output/"protocol.json", {"study": "pid-fullspeed-100-v1", "seed": TUNING_SEED,
               "robot": {**protocol["robot"], "max_speed": 1.}, "episode": protocol["episode"],
               "source_sha256": digest(Path(__file__)), "baseline_sha256": digest(SLOW_PID),
               "suite_sha256": digest(suite), "base": 1., "gain_fields": FIELDS, "bounds": BOUNDS,
               "initial_candidates": len(policies), "workers": workers,
               "selection": "Require all dedicated straights completed with central-2m speed >=0.95 m/s. Maximize validation completion, retain candidates within 2% of the fastest failure-penalized mean episode time among those, then minimize steady yaw oscillation, command jitter, and geometric RMSE.",
               "refinement": "Two rounds of 24 candidates around promoted full-speed candidates; base and recovery logic fixed.",
               "comparison": "Freeze selected PID and unchanged NEAT g134 before generating a fresh shared test suite."})
    snapshot_sources(output/"source.zip")
    with ZipFile(output/"source.zip", "a", compression=ZIP_DEFLATED) as archive:
        for source in (Path(__file__), ROOT/"run_pid_smoothness_tuning.py", ROOT/"compare_neat_pid_100.py"):
            archive.write(source, source.name)
    entries = read_suite(suite)["splits"]
    names = {"speed-000", "speed-006", "speed-007", "speed-010", "speed-012", "speed-013", "speed-016", "speed-019", "corner-000", "corner-001", "mixed-000", "mixed-001"}
    coarse = [entry for entry in entries["training"] if Path(entry["path"]).stem in names]
    assert len(coarse) == 12
    rng = np.random.default_rng(derived_seed(TUNING_SEED, "refinement"))
    with mp.get_context("spawn").Pool(workers) as pool:
        results = batch(pool, policies, coarse, suite, protocol, output, "coarse")
        summaries = {key: metrics(rows) for key, rows in results.items()}
        promoted = list(dict.fromkeys(["previous-base-075", "full-cruise-reference", *promote(summaries)]))
        results = batch(pool, {key: policies[key] for key in promoted}, entries["training"], suite, protocol, output, "expanded")
        for generation in (1, 2):
            summaries = {key: metrics(rows) for key, rows in results.items()}
            parents = promote(summaries, 6)
            children = []
            for index in range(24):
                parent = policies[parents[index % len(parents)]]
                for _ in range(100):
                    center = np.array([parent[field] for field in FIELDS])
                    scales = np.array([max(.08, center[0]*.15), .025, max(.005, center[2]*.15), .012, max(.3, center[4]*.2)])
                    child = add(rng.normal(center, scales), f"refine{generation}")
                    if child:
                        children.append(child)
                        break
            write_json(output/"candidates.json", policies)
            results.update(batch(pool, {key: policies[key] for key in children}, entries["training"], suite, protocol, output, f"refine-{generation}"))
        summaries = {key: metrics(rows) for key, rows in results.items()}
        finalists = list(dict.fromkeys(["previous-base-075", "full-cruise-reference", *promote(summaries, 10)]))
        validation = batch(pool, {key: policies[key] for key in finalists}, entries["validation"], suite, protocol, output, "validation")
    summaries = [metrics(rows) for rows in validation.values()]
    append_csv(output/"validation-summary.csv", summaries, list(summaries[0]))
    eligible = [row for row in summaries if row["speed_feasible"] and policies[row["candidate"]]["base"] == 1.]
    if not eligible:
        raise ValueError("No candidate met the straight-speed feasibility requirement")
    maximum = max(row["completed"] for row in eligible)
    complete = [row for row in eligible if row["completed"] == maximum]
    fastest = min(row["failure_penalized_mean_time_s"] for row in complete)
    shortlist = [row for row in complete if row["failure_penalized_mean_time_s"] <= 1.02*fastest]
    winner = min(shortlist, key=lambda row: (row["steady_yaw_rms_rad_s"], row["command_jitter"], row["geometric_rmse_mm"], row["failure_penalized_mean_time_s"]))
    path = output/"tuned-1.00-fullspeed.json"
    write_json(path, policies[winner["candidate"]])
    write_json(output/"selection-freeze.json", {"candidate": winner["candidate"], "config": path.name,
               "sha256": digest(path), "validation": winner, "fastest_validation_cost_s": fastest,
               "shortlist": [row["candidate"] for row in shortlist], "frozen_before_shared_test": True})
    write_json(output/"status.json", {"complete": True, "candidates": len(policies), "wall_seconds": time.perf_counter()-started})
    print(json.dumps({"selected": winner, "parameters": policies[winner["candidate"]]}, indent=2), flush=True)
    return path


def freeze_comparison(output, pid_path):
    if (output/"frozen").exists():
        manifest = comparison.verify_frozen(output)
        if manifest["orchestrator_sha256"] != digest(Path(__file__)):
            raise ValueError("Full-speed experiment source changed")
        return manifest
    prior = comparison.verify_frozen(comparison.OUTPUT)
    robot = prior["robot"]
    selected = read_json(pid_path.parent/"selection-freeze.json")
    if (digest(pid_path) != selected["sha256"] or read_json(pid_path)["base"] != 1.
            or not selected["validation"]["speed_feasible"] or selected["validation"]["minimum_straight_m_s"] < .95):
        raise ValueError("Expected the frozen full-cruise PID")
    manifest = {**prior, "study": "neat134-vs-pid100-fullspeed-v1", "software": software(),
                "driver_sha256": digest(Path(comparison.__file__)),
                "orchestrator_sha256": digest(Path(__file__)),
                "comparison": "Both policies can command full 1.00 m/s straight travel; PID base fixed at 1.00 and steering/slowdown retuned.",
                "policies": {"neat": {**prior["policies"]["neat"], "source": str(comparison.OUTPUT/"frozen"/prior["policies"]["neat"]["file"])},
                             "pid": {"file": "pid-filtered-100.json", "source": str(pid_path), "sha256": digest(pid_path), "selection": selected["candidate"], "base": 1.}}}
    from datetime import datetime, timezone
    manifest["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    with TemporaryDirectory(prefix=".freeze-", dir=output) as temporary:
        ready = Path(temporary)/"frozen"
        ready.mkdir()
        shutil.copyfile(Path(manifest["policies"]["neat"]["source"]), ready/manifest["policies"]["neat"]["file"])
        shutil.copyfile(pid_path, ready/"pid-filtered-100.json")
        for name in ("robot.json", "episode.json", "neat-training-run.json"):
            shutil.copyfile(comparison.OUTPUT/"frozen"/name, ready/name)
        shutil.copyfile(pid_path.parent/"selection-freeze.json", ready/"pid-selection-freeze.json")
        manifest["config_sha256"] = {name: digest(ready/name) for name in ("robot.json", "episode.json", "neat-training-run.json", "pid-selection-freeze.json")}
        write_json(ready/"manifest.json", manifest)
        publish_directory(ready, output/"frozen", "manifest.json")
    return comparison.verify_frozen(output)


def assess(output, pid_path, workers):
    output.mkdir(parents=True, exist_ok=True)
    manifest = freeze_comparison(output, pid_path)
    suite = make_suite(output/"suite", TEST_SEED, {"test": {"mixed": 90, "speed": 30, "corner": 30}})
    entries = read_suite(suite)["splits"]["test"]
    if not (output/"protocol.json").exists():
        snapshot_sources(output/"source.zip")
        with ZipFile(output/"source.zip", "a", compression=ZIP_DEFLATED) as archive:
            for source in (Path(__file__), ROOT/"compare_neat_pid_100.py", ROOT/"run_pid_smoothness_tuning.py", ROOT/"replay_neat_pid_100.py"):
                archive.write(source, source.name)
        files = [suite, output/"source.zip", *(suite.parent/entry["path"] for entry in entries)]
        write_json(output/"protocol.json", {"study": "neat134-vs-pid100-fullspeed-v1", "seed": TEST_SEED,
                   "freeze_sha256": digest(output/"frozen/manifest.json"), "workers": workers,
                   "files": {path.relative_to(output).as_posix(): digest(path) for path in files},
                   "groups": {"mixed": 90, "speed": 30, "corner": 30},
                   "replay_tracks": [entry["path"] for entry in entries if entry["replay"]],
                   "statistics_helper_seed": comparison.SEED, "bootstrap_repetitions": 20000,
                   "correction": "Previous PID base 0.75 limited cruise speed. All candidate full-speed PIDs use base 1.00 and verified >=0.95 m/s sustained on validation straights."})
    comparison.verify(output)
    destination = output/"results"
    if not destination.exists():
        tasks = [{"policy": policy, "config": str(output/"frozen"/manifest["policies"][policy]["file"]),
                  "track": str(suite.parent/entry["path"]), "entry": entry,
                  "robot_json": json.dumps(manifest["robot"], sort_keys=True), "episode": manifest["episode"],
                  "trace": str(output/"telemetry"/f"{policy}-{Path(entry['path']).stem}.csv") if entry["replay"] else None}
                 for entry in entries for policy in comparison.POLICIES]
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = []
            for index, row in enumerate(pool.imap(comparison.worker, tasks), 1):
                rows.append(row)
                if index % 60 == 0:
                    print(f"Fresh full-speed comparison: {index}/300 episodes", flush=True)
        summary = comparison.summarize(rows, entries)
        if len(rows) != 300:
            raise ValueError("Expected 300 paired episode records")
        with TemporaryDirectory(prefix=".results-", dir=output) as temporary:
            ready = Path(temporary)/"results"
            ready.mkdir()
            append_csv(ready/"episodes.csv", rows, list(rows[0]))
            write_json(ready/"summary.json", summary)
            write_json(ready/"status.json", {"complete": True, "episodes": len(rows),
                       "files": {name: digest(ready/name) for name in ("episodes.csv", "summary.json")}})
            publish_directory(ready, destination, "status.json")
    status = read_json(destination/"status.json")
    if any(digest(destination/name) != sha for name, sha in status["files"].items()):
        raise ValueError("Frozen comparison results changed")
    if not (output/"exploratory-long-course.json").exists():
        track_path = ROOT/"runs/long-course-speed-challenge/track.json"
        track = comparison.TrackSpec.load(track_path)
        records = [comparison.worker({"policy": policy, "config": str(output/"frozen"/manifest["policies"][policy]["file"]),
                    "track": str(track_path), "entry": {"group": "exploratory", "replay": False, "windows": comparison.steady_windows(track)},
                    "robot_json": json.dumps(manifest["robot"], sort_keys=True), "episode": manifest["episode"],
                    "trace": str(output/"telemetry"/f"{policy}-exploratory-long.csv")}) for policy in comparison.POLICIES]
        write_json(output/"exploratory-long-course.json", records)
    comparison.verify(output)
    print(json.dumps(read_json(destination/"summary.json"), indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "tune", "compare", "all"))
    parser.add_argument("--tuning", type=Path, default=TUNING)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    tuning, output = args.tuning.resolve(), args.output.resolve()
    if args.command == "probe":
        probe(tuning)
        return
    with session_lock(tuning):
        if args.command in ("tune", "all"):
            path = tune(tuning, args.workers)
        else:
            path = tuning/"tuned-1.00-fullspeed.json"
        if args.command in ("compare", "all"):
            with session_lock(output):
                assess(output, path, args.workers)


if __name__ == "__main__":
    mp.freeze_support()
    main()
