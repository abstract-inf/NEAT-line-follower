"""Freeze and compare the selected NEAT/PID controllers at a 1.00 m/s wheel cap."""

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import SimpleQueue
import shutil
from statistics import mean
from tempfile import TemporaryDirectory
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.controllers import load_model, read_model_artifact
from line_follower.episodes import Episode
from line_follower.experiments import append_csv, snapshot_sources, write_json
from line_follower.fast_study import speed_course
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.pid_experiments import _mask, _simulation
from line_follower.suites import corner_track, derived_seed, read_suite
from line_follower.tracks import GeneratorConfig, TrackGenerationError, TrackSpec, generate_track, validate_track
from run_corner_training import digest, protected_tracks, read_json, session_lock
from run_neat_fast_study import publish_directory, software, verify as verify_neat_study
from run_pid_smoothness_tuning import steady_metrics

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs/neat134-vs-pid100-20261119"
NEAT = ROOT / "runs/neat-fast-study/seed-01/training/champions/generation-0134.pkl"
PID = ROOT / "runs/pid-100-smoothness-20261119/tuned-1.00.json"
SEED = 2026111903
POLICIES = ("neat", "pid")
REPLAY = ("mixed-000", "mixed-001", "mixed-002", "mixed-005", "speed-000",
          "speed-009", "speed-018", "speed-024", "corner-000", "corner-001")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def freeze(output):
    frozen = output / "frozen"
    if frozen.exists():
        return verify_frozen(output)
    verify_neat_study(ROOT / "runs/neat-fast-study/prepared")
    artifact = read_model_artifact(NEAT)
    metadata = artifact["metadata"]
    training_path = NEAT.parent.parent / "run.json"
    training = read_json(training_path)
    robot = RobotConfig(**metadata["robot"])
    episode = EpisodeConfig(**metadata["episode"])
    prior = read_json(PID.parent / "protocol.json")
    selection = read_json(PID.parent / "selection-freeze.json")
    if metadata["generation"] != 134 or robot.max_speed != 1.:
        raise ValueError("Expected the generation 134 champion at its 1.00 m/s cap")
    if robot != RobotConfig(**prior["robot"]) or episode != EpisodeConfig(**prior["episode"]):
        raise ValueError("The two saved controllers require different physical or episode settings")
    if digest(PID) != selection["sha256"] or ResearchPIDConfig.load(PID).corner_supervision:
        raise ValueError("Expected the frozen filtered recovery PID refinement")
    manifest = {"study": "neat134-vs-pid100-v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "software": software(), "driver_sha256": digest(Path(__file__)),
                "metric_helper_sha256": digest(ROOT / "run_pid_smoothness_tuning.py"),
                "policies": {"neat": {"file": "neat-generation-0134.pkl", "sha256": digest(NEAT),
                                      "source": str(NEAT), "generation": 134, "lineage_seed": 1,
                                      "evolution_seed": training["settings"]["seed"],
                                      "selection": "Exploratory checkpoint identified from prior validation and user replay"},
                             "pid": {"file": "pid-filtered-100.json", "sha256": digest(PID),
                                     "source": str(PID), "selection": selection["candidate"]}},
                "robot": asdict(robot), "episode": asdict(episode),
                "comparison": "Fixed resulting controllers at an identical wheel-speed cap; throttle commands remain policy-specific"}
    with TemporaryDirectory(prefix=".freeze-", dir=output) as temporary:
        ready = Path(temporary) / "frozen"
        ready.mkdir()
        for policy, source in (("neat", NEAT), ("pid", PID)):
            shutil.copyfile(source, ready / manifest["policies"][policy]["file"])
        write_json(ready / "robot.json", asdict(robot))
        write_json(ready / "episode.json", asdict(episode))
        shutil.copyfile(training_path, ready / "neat-training-run.json")
        manifest["config_sha256"] = {name: digest(ready/name) for name in ("robot.json", "episode.json", "neat-training-run.json")}
        write_json(ready / "manifest.json", manifest)
        publish_directory(ready, frozen, "manifest.json")
    return verify_frozen(output)


def verify_frozen(output):
    directory = output / "frozen"
    manifest = read_json(directory / "manifest.json")
    if (manifest["software"] != software() or manifest["driver_sha256"] != digest(Path(__file__))
            or manifest["metric_helper_sha256"] != digest(ROOT / "run_pid_smoothness_tuning.py")):
        raise ValueError("Comparison source or dependencies changed; use the preserved source snapshot")
    for record in manifest["policies"].values():
        if digest(directory / record["file"]) != record["sha256"]:
            raise ValueError("A frozen controller changed")
    for name, sha in manifest["config_sha256"].items():
        if digest(directory / name) != sha:
            raise ValueError("A frozen configuration changed")
    return manifest


def steady_windows(track):
    """Use windows long enough for both policies at the shared cap."""
    position, windows = 0., []
    for section in track.sections:
        length = section.length if section.kind == "straight" else abs(section.radius*section.angle) if section.kind == "arc" else 0.
        if section.kind == "arc" and length >= 1.4:
            windows.append((position+.25*length, position+.75*length))
        elif section.kind == "straight" and length >= 3:
            windows.append((position+length/2-1, position+length/2+1))
        position += length
    return windows


def prepare_tracks(output):
    suite_path = output / "suite/suite.json"
    if suite_path.exists():
        return suite_path
    protected = protected_tracks()
    # Include previous protected geometry as well as every currently saved split.
    for path in (ROOT / "runs").rglob("suite.json"):
        protected.update(read_suite(path).get("protected_geometry_hashes", []))
    for path in (ROOT / "runs/long-course-speed-challenge/track.json",
                 ROOT / "runs/pid-100-smoothness-20261119/diagnostic-curve.json"):
        protected.add(TrackSpec.load(path).geometry_hash)
    used = set(protected)
    entries = []
    for group, count in (("mixed", 90), ("speed", 30), ("corner", 30)):
        for index in range(count):
            for attempt in range(100):
                seed = derived_seed(SEED, group, index, attempt)
                try:
                    if group == "mixed":
                        track = generate_track(seed, ("beginner", "intermediate", "advanced")[index % 3])
                    elif group == "speed":
                        track = speed_course(seed, index*20//count, "test")
                    else:
                        track = corner_track(seed, 3, index % 2, "advanced", GeneratorConfig())
                    track = replace(track, split="test")
                except TrackGenerationError:
                    continue
                if track.geometry_hash not in used and not validate_track(track):
                    break
            else:
                raise ValueError("Could not generate a fresh valid comparison course")
            used.add(track.geometry_hash)
            path = f"test/{group}-{index:03d}.json"
            track.save(suite_path.parent / path)
            entries.append({"path": path, "geometry_hash": track.geometry_hash, "group": group,
                            "difficulty": track.difficulty, "features": track.features,
                            "dedicated_straight": group == "speed" and all(s.kind == "straight" for s in track.sections),
                            "windows": steady_windows(track) if group == "speed" else [],
                            "replay": f"{group}-{index:03d}" in REPLAY})
    write_json(suite_path, {"suite_version": 1, "generator_version": "1.0", "root_seed": SEED,
                           "protected_geometry_hashes": sorted(protected), "splits": {"test": entries}})
    return suite_path


def worker(task):
    simulation = _simulation(task["robot_json"])
    mask = _mask(task["track"])
    if task["policy"] == "neat":
        controller, _ = load_model(task["config"])
    else:
        controller = ResearchPIDController(ResearchPIDConfig.load(task["config"]), simulation.config)
    episode = Episode(simulation, mask.track, controller, EpisodeConfig(**task["episode"]), mask, detailed=True)
    windows = task["entry"].get("windows", [])
    samples = [[] for _ in windows]
    trace_path = Path(task["trace"]) if task.get("trace") else None
    stream = writer = None
    try:
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            stream = trace_path.open("w", newline="", encoding="utf-8")
        while not episode.done:
            row = episode.advance(telemetry=bool(windows) or bool(trace_path))
            if row:
                if stream:
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                distance = row["progress"]*mask.track.route.length
                for index, (lower, upper) in enumerate(windows):
                    if lower <= distance <= upper:
                        samples[index].append(row)
    finally:
        if stream:
            stream.close()
    result = asdict(episode.result())
    if result["error"] or result["geometry_lateral_rmse_m"] is None:
        raise ValueError(f"Technical comparison failure: {result['error']}")
    metrics = [steady_metrics(rows) for rows in samples if len(rows) >= 21]
    if windows and result["completed"] and len(metrics) != len(windows):
        raise ValueError("Missing a steady window on a completed speed course")
    return {"policy": task["policy"], "group": task["entry"]["group"],
            "replay": task["entry"]["replay"], "telemetry": str(trace_path) if trace_path else "",
            **result, "steady_yaw_rms_rad_s": mean(row["yaw_oscillation_rms_rad_s"] for row in metrics) if metrics else None,
            "steady_windows": len(metrics)}


def verify(output):
    manifest = verify_frozen(output)
    protocol = read_json(output / "protocol.json")
    if digest(output / "frozen/manifest.json") != protocol["freeze_sha256"]:
        raise ValueError("Frozen manifest changed")
    for path, sha in protocol["files"].items():
        if digest(output / path) != sha:
            raise ValueError(f"Frozen comparison file changed: {path}")
    return manifest, protocol


def run(output, workers):
    started = time.perf_counter()
    manifest = freeze(output)  # Freeze the requested policies before constructing test geometry.
    suite = prepare_tracks(output)
    entries = read_suite(suite)["splits"]["test"]
    if not (output / "protocol.json").exists():
        snapshot_sources(output / "source.zip")
        with ZipFile(output / "source.zip", "a", compression=ZIP_DEFLATED) as archive:
            for path in (Path(__file__), ROOT / "run_pid_smoothness_tuning.py"):
                archive.write(path, path.name)
        files = [output / "source.zip", suite, *(suite.parent / entry["path"] for entry in entries)]
        write_json(output / "protocol.json", {"study": "neat134-vs-pid100-v1", "seed": SEED,
                   "freeze_sha256": digest(output/"frozen/manifest.json"),
                   "files": {path.relative_to(output).as_posix(): digest(path) for path in files},
                   "groups": {"mixed": 90, "speed": 30, "corner": 30}, "workers": workers,
                   "replay_tracks": [entry["path"] for entry in entries if entry["replay"]],
                   "bootstrap_repetitions": 20000,
                   "analysis": "Stratified paired track bootstrap; finish-time comparison on shared successes; exact McNemar completion test. Inference concerns these fixed controllers across generated tracks, not NEAT seed variability."})
    verify(output)
    destination = output / "results"
    if destination.exists():
        status = read_json(destination / "status.json")
        if any(digest(destination/name) != sha for name, sha in status["files"].items()):
            raise ValueError("Saved comparison results changed")
        print("Verified and reused completed comparison.")
        return
    tasks = [{"policy": policy, "config": str(output / "frozen" / manifest["policies"][policy]["file"]),
              "track": str(suite.parent / entry["path"]), "entry": entry,
              "robot_json": json.dumps(manifest["robot"], sort_keys=True), "episode": manifest["episode"],
              "trace": str(output / "telemetry" / f"{policy}-{Path(entry['path']).stem}.csv") if entry["replay"] else None}
             for entry in entries for policy in POLICIES]
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for index, row in enumerate(pool.imap(worker, tasks), 1):
            rows.append(row)
            if index % 60 == 0:
                print(f"Frozen comparison: {index}/{len(tasks)} episodes", flush=True)
    if len(rows) != 300:
        raise ValueError("Expected 300 paired episode records")
    verify(output)
    with TemporaryDirectory(prefix=".results-", dir=output) as temporary:
        ready = Path(temporary) / "results"
        ready.mkdir()
        append_csv(ready / "episodes.csv", rows, list(rows[0]))
        summary = summarize(rows, entries)
        write_json(ready / "summary.json", summary)
        write_json(ready / "status.json", {"complete": True, "episodes": len(rows), "wall_seconds": time.perf_counter()-started,
                   "files": {name: digest(ready/name) for name in ("episodes.csv", "summary.json")}})
        publish_directory(ready, destination, "status.json")
    print(json.dumps(summary, indent=2), flush=True)


def wilson(successes, count):
    z = 1.959963984540054
    center = (successes/count+z*z/(2*count))/(1+z*z/count)
    radius = z*math.sqrt(successes/count*(1-successes/count)/count+z*z/(4*count*count))/(1+z*z/count)
    return [center-radius, center+radius]


def summarize(rows, entries):
    indexed = {policy: {row["track_hash"]: row for row in rows if row["policy"] == policy} for policy in POLICIES}
    hashes = [entry["geometry_hash"] for entry in entries]
    if any(set(records) != set(hashes) for records in indexed.values()):
        raise ValueError("Both policies must have identical track sets")
    shared = [key for key in hashes if all(indexed[policy][key]["completed"] for policy in POLICIES)]
    speed_shared = [key for key in shared if indexed["neat"][key]["group"] == "speed"]
    summaries = {}
    for policy, records in indexed.items():
        summaries[policy] = {}
        for group in ("all", "mixed", "speed", "corner"):
            subset = [row for row in records.values() if group == "all" or row["group"] == group]
            complete = sum(row["completed"] for row in subset)
            straight_keys = [entry["geometry_hash"] for entry in entries if entry["dedicated_straight"]
                             and (group == "all" or entry["group"] == group)]
            measured = [records[key]["straight_speed_m_s"] for key in straight_keys if records[key]["straight_speed_m_s"] is not None]
            summaries[policy][group] = {"completed": complete, "tracks": len(subset),
                "completion_rate": complete/len(subset), "completion_wilson_95": wilson(complete, len(subset)),
                "mean_progress": mean(row["progress"] for row in subset),
                "rmse_all_mm": 1000*mean(row["geometry_lateral_rmse_m"] for row in subset),
                "command_jitter_all": mean(row["command_jitter"] for row in subset),
                "mean_episode_forward_m_s": mean(row["average_forward_speed_m_s"] for row in subset),
                "peak_forward_m_s": max(row["peak_forward_speed_m_s"] for row in subset),
                "straight_segments_measured": len(measured), "straight_segments_total": len(straight_keys),
                "mean_straight_m_s": mean(measured) if measured else None,
                "minimum_straight_m_s": min(measured) if measured else None,
                "failure_reasons": dict(Counter(row["termination"] for row in subset if not row["completed"]))}
        paired = [records[key] for key in shared]
        steady = [records[key] for key in speed_shared]
        summaries[policy]["shared"] = {"tracks": len(shared),
            "mean_finish_time_s": mean(row["completion_time_s"] for row in paired) if paired else None,
            "rmse_mm": 1000*mean(row["geometry_lateral_rmse_m"] for row in paired) if paired else None,
            "command_jitter": mean(row["command_jitter"] for row in paired) if paired else None,
            "steady_speed_courses": len(steady),
            "steady_yaw_rms_rad_s": mean(row["steady_yaw_rms_rad_s"] for row in steady) if steady else None}
    neat_only = sum(indexed["neat"][key]["completed"] and not indexed["pid"][key]["completed"] for key in hashes)
    pid_only = sum(indexed["pid"][key]["completed"] and not indexed["neat"][key]["completed"] for key in hashes)
    discordant = neat_only+pid_only
    exact_p = min(1., 2*sum(math.comb(discordant, i) for i in range(min(neat_only, pid_only)+1))/2**discordant) if discordant else 1.
    rng = np.random.default_rng(derived_seed(SEED, "paired-bootstrap"))

    def bootstrap(keys, statistic):
        strata = defaultdict(list)
        for index, key in enumerate(keys):
            row = indexed["neat"][key]
            strata[row["group"], row["difficulty"]].append(index)
        samples = np.concatenate([rng.choice(indices, (20000, len(indices)), replace=True) for indices in strata.values()], axis=1)
        return np.percentile(statistic(samples), [2.5, 97.5]).tolist()

    completion_difference = np.array([int(indexed["neat"][key]["completed"])-int(indexed["pid"][key]["completed"]) for key in hashes])
    paired = {"shared_finishes": len(shared), "neat_only": neat_only, "pid_only": pid_only,
              "both_failed": len(hashes)-len(shared)-neat_only-pid_only,
              "completion_difference_neat_minus_pid": float(np.mean(completion_difference)),
              "completion_difference_paired_bootstrap_95": bootstrap(hashes, lambda samples: completion_difference[samples].mean(axis=1)),
              "mcnemar_exact_two_sided_p": exact_p}
    if shared:
        times = {policy: np.array([indexed[policy][key]["completion_time_s"] for key in shared]) for policy in POLICIES}
        changes = times["neat"]-times["pid"]
        paired.update({"mean_time_difference_neat_minus_pid_s": float(changes.mean()),
                       "time_difference_paired_bootstrap_95_s": bootstrap(shared, lambda samples: changes[samples].mean(axis=1)),
                       "relative_time_change_pct": 100*(float(times["neat"].mean()/times["pid"].mean())-1),
                       "relative_time_change_paired_bootstrap_95_pct": bootstrap(shared, lambda samples: 100*(times["neat"][samples].mean(axis=1)/times["pid"][samples].mean(axis=1)-1)),
                       "neat_faster": int(np.sum(changes < -1e-9)), "pid_faster": int(np.sum(changes > 1e-9)),
                       "time_ties": int(np.sum(np.abs(changes) <= 1e-9))})
    return {"policies": summaries, "paired": paired}


def replay_setup(output):
    manifest, protocol = verify(output)
    tracks = [TrackSpec.load(output / "suite" / path) for path in protocol["replay_tracks"]]
    if len(tracks) != 10:
        raise ValueError("Expected ten prespecified replay tracks")
    robot, episode = RobotConfig(**manifest["robot"]), EpisodeConfig(**manifest["episode"])
    neat, _ = load_model(output / "frozen" / manifest["policies"]["neat"]["file"])
    pid = ResearchPIDController(ResearchPIDConfig.load(output / "frozen" / manifest["policies"]["pid"]["file"]), robot)
    return tracks, robot, episode, (neat, pid)


def view(output):
    import mujoco.viewer
    from line_follower.simulation import Simulation
    from line_follower.viewer import add_overlays, camera_for, set_transparency
    tracks, robot, config, controllers = replay_setup(output)
    names = ("Frozen NEAT generation 134", "Frozen tuned filtered PID")
    simulation = Simulation(robot)
    simulation.reset(tracks[0])
    simulation.enable_rendering()
    keys = SimpleQueue()
    trajectory = deque(maxlen=1200)
    policy = track_index = 0
    paused = reference = transparent = False
    episode = Episode(simulation, tracks[0], controllers[policy], config, detailed=True)
    print("C: switch frozen NEAT/PID | N/P: next/previous track | R: reset | Space: pause | T: route", flush=True)
    with mujoco.viewer.launch_passive(simulation.model, simulation.data, key_callback=keys.put) as viewer:
        reset = True
        while viewer.is_running():
            started = time.perf_counter()
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    paused = not paused
                elif key in (ord("N"), ord("P")):
                    track_index = (track_index+(1 if key == ord("N") else -1)) % len(tracks)
                    reset = True
                elif key == ord("C"):
                    policy, reset = 1-policy, True
                elif key == ord("R"):
                    reset = True
                elif key == ord("T"):
                    reference = not reference
                elif key == ord("B"):
                    transparent = not transparent
                    set_transparency(simulation, transparent)
            if reset:
                episode = Episode(simulation, tracks[track_index], controllers[policy], config, detailed=True)
                simulation.update_visual_track(viewer)
                camera = camera_for(simulation, "track")
                with viewer.lock():
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
                trajectory.clear()
                paused, reset = False, False
                print(f"Track {track_index+1}/10: {tracks[track_index].name} | {names[policy]} | cap 1.00 m/s", flush=True)
            if not paused and not episode.done:
                episode.advance()
                trajectory.append(simulation.state.position.copy())
                if episode.done:
                    print(asdict(episode.result()), flush=True)
                    paused = True
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                add_overlays(viewer.user_scn, simulation, reference, list(trajectory))
            viewer.sync()
            time.sleep(max(0, robot.dt-(time.perf_counter()-started)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "view", "list"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.command == "run":
        with session_lock(output):
            run(output, args.workers)
    elif args.command == "view":
        view(output)
    else:
        tracks, robot, _, _ = replay_setup(output)
        print(f"Both frozen controllers use a {robot.max_speed:.2f} m/s cap.")
        for index, track in enumerate(tracks, 1):
            print(index, track.name, track.difficulty, track.features)


if __name__ == "__main__":
    mp.freeze_support()
    main()
