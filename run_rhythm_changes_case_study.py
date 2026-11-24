"""Freeze and evaluate saved NEAT/PID policies on candidate 4, without training."""

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import gzip
import json
import multiprocessing as mp
from pathlib import Path
from queue import SimpleQueue
import shutil
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.controllers import load_model, read_model_artifact
from line_follower.episodes import Episode, TELEMETRY_FIELDS
from line_follower.experiments import write_json
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.pid_experiments import PID_FIELDS
from line_follower.simulation import Simulation
from line_follower.tracks import TrackMask, TrackSpec
import run_neat_fast_study as study

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs/rhythm-changes-case-study-20261121"
CAPS = (.50, .65, .80, 1.00, 1.25, 1.50)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def cap_label(cap):
    return f"{round(cap * 100):03d}"


def freeze(output):
    """Commit all model choices and evaluation tasks before any course outcomes."""
    if (output / "protocol.json").exists():
        return verify(output)
    if output.exists():
        raise FileExistsError("An incomplete output exists; preserve it and choose another --output")
    prepared = study.OUTPUT / "prepared"
    old = study.verify(prepared)
    candidates = ROOT / "runs/long-course-candidates-20261121"
    course = next(c for c in read(candidates / "manifest.json")["candidates"] if c["id"] == 4)
    source_track = candidates / course["file"]
    if study.digest(source_track) != course["file_sha256"]:
        raise ValueError("The chosen course changed after its previews were published")
    track = TrackSpec.load(source_track)
    if track.geometry_hash != course["geometry_sha256"]:
        raise ValueError("Unexpected course geometry")
    selections = []
    for seed in range(1, 6):
        training = study.OUTPUT / f"seed-{seed:02d}/training"
        log = rows(training / "generations.csv")
        if len(log) != 300 or {int(r["generation"]) for r in log} != set(range(300)):
            raise ValueError(f"Seed {seed} has not completed its fast training")
        evolution_seed = read(training / "run.json")["settings"]["seed"]
        low = prepared / f"inputs/seed-{seed:02d}-low.pkl"
        selections.append((f"neat-s{seed}-low-050", low, .5, seed, "low_parent", None))
        for cap in CAPS[1:]:
            eligible = [r for r in log if float(r["wheel_speed_cap_m_s"]) == cap]
            winner = max(eligible, key=lambda r: (float(r["validation_completion_rate"]),
                         float(r["validation_score"]), -int(r["generation"])))
            generation = int(winner["generation"])
            source = training / f"champions/generation-{generation:04d}.pkl"
            selections.append((f"neat-s{seed}-fast-{cap_label(cap)}", source, cap, seed,
                               "validation_phase_best", {"generation": generation,
                               "evolution_seed": evolution_seed,
                               "validation_completion_rate": float(winner["validation_completion_rate"]),
                               "validation_score": float(winner["validation_score"])}))
    # The only formally quality-gated final winner is a separate prespecified control.
    for seed in range(1, 6):
        source = study.OUTPUT / f"seed-{seed:02d}/training/best.pkl"
        if source.exists() and all(study.digest(source) != study.digest(s[1]) for s in selections):
            selections.append((f"neat-s{seed}-formal-150", source, 1.5, seed, "formal_quality_gate", None))
    smooth = ROOT / "runs/pid-100-smoothness-20261119"
    full = study.OUTPUT / "final-review-20261121/comparison-1m/frozen"
    pid_sources = [
        ("pid-low-050", prepared / "inputs/pid-050.json", .5, old["inputs"]["pid-050.json"]),
        ("pid-followup-100", prepared / "inputs/pid-100.json", 1., old["inputs"]["pid-100.json"]),
        ("pid-smooth-100", smooth / "tuned-1.00.json", 1., read(smooth / "selection-freeze.json")["sha256"]),
        ("pid-fullspeed-100", full / "pid-fullspeed.json", 1., read(full / "manifest.json")["policies"]["pid"]["sha256"]),
        ("pid-followup-150", prepared / "inputs/pid-150.json", 1.5, old["inputs"]["pid-150.json"]),
    ]
    for name, source, cap, expected in pid_sources:
        if study.digest(source) != expected or ResearchPIDConfig.load(source).corner_supervision:
            raise ValueError(f"Frozen filtered PID identity changed: {name}")
        selections.append((name, source, cap, None, "saved_pid_tune", None))
    (output / "frozen").mkdir(parents=True)
    shutil.copy2(source_track, output / "track.json")
    shutil.copy2(candidates / "manifest.json", output / "candidate-proposal.json")
    policies = {}
    for name, source, cap, seed, selection, validation in selections:
        path = output / "frozen" / (name + source.suffix)
        shutil.copy2(source, path)
        record = {"file": str(path.relative_to(output)), "source": str(source.relative_to(ROOT)),
                  "sha256": study.digest(path), "native_cap_m_s": cap, "seed": seed,
                  "family": "NEAT" if seed else "PID", "selection": selection}
        if seed:
            controller, metadata = load_model(path)
            if metadata.get("controller_semantics") != "mirrored-sensor-history-v1":
                raise ValueError("Unexpected NEAT controller semantics")
            if float(metadata["robot"]["max_speed"]) != cap:
                raise ValueError("Native cap disagrees with champion metadata")
            if any(metadata["robot"][k] != v for k, v in old["robot"].items() if k != "max_speed"):
                raise ValueError("Champion physics differ from the frozen robot")
            outputs = [n[0] for n in controller.network.node_evals]
            keys = read_model_artifact(path)["config"].genome_config.output_keys
            record.update(generation=metadata["generation"], evolution_seed=metadata.get("evolution_seed"),
                          drive_evaluated=keys[0] in outputs, steering_evaluated=keys[1] in outputs)
            if validation:
                if metadata["generation"] != validation["generation"]:
                    raise ValueError("Champion generation disagrees with validation selection")
                record.update(validation)
        else:
            record["parameters"] = asdict(ResearchPIDConfig.load(path))
        policies[name] = record
    tasks = []
    for name, policy in policies.items():
        caps = CAPS if policy["family"] == "PID" else sorted({policy["native_cap_m_s"], 1.})
        for cap in caps:
            tasks.append({"id": f"{name}__cap-{cap_label(cap)}", "policy": name, "cap_m_s": cap,
                          "native_cap": cap == policy["native_cap_m_s"],
                          "time_limit_s": 180 / min(cap, 1.)})
    tasks.sort(key=lambda t: (t["cap_m_s"], t["policy"]))
    protocol = {"study": "rhythm-changes-long-course-v1", "frozen_utc": stamp(),
        "purpose": "Exploratory single-course comparison, chosen from layouts before testing",
        "chosen_candidate": 4, "course": course, "robot": old["robot"], "episode": old["episode"],
        "primary_common_cap_m_s": 1., "pid_caps_m_s": CAPS,
        "selection": "Fast phase: validation completion, then fitness, earliest tie, within each cap and seed. Also all low parents and the formal final-stage winner. No course outcomes used.",
        "time_limit_rule": "180 / min(wheel_speed_cap_m_s, 1.0); 180 s at caps >= 1, scaled at lower caps, identical for all policies at each cap",
        "rules": "Existing corridor, no-contact departure and ordered-progress stall rules; fresh controller/reset per episode. No tuning or evolution.",
        "metrics": "Geometric error to nearest route segment; step-weighted chassis forward speed; jitter=sum(deltaL^2+deltaR^2)/(8*(steps-1)); section times use interpolated first ordered-progress crossings; straight windows central 2 m of each primitive >=4 m.",
        "interpretation": "One deterministic episode per controller/cap; no across-track uncertainty or robustness claim. Native-cap and common-cap results kept separate; elapsed failure times are not lap times.",
        "software": study.software(), "runner_sha256": study.digest(Path(__file__)),
        "track_sha256": study.digest(output / "track.json"), "policies": policies, "tasks": tasks}
    with ZipFile(output / "source-snapshot.zip", "w", ZIP_DEFLATED) as archive:
        for path in [Path(__file__), ROOT / "prepare_long_course_candidates.py", *sorted((ROOT / "line_follower").rglob("*.py")),
                     *sorted((ROOT / "line_follower/assets").glob("*")), ROOT / "requirements-lock.txt"]:
            if path.is_file():
                archive.write(path, str(path.relative_to(ROOT)))
    protocol["source_snapshot_sha256"] = study.digest(output / "source-snapshot.zip")
    write_json(output / "protocol.json", protocol)
    print(f"Frozen {len(policies)} policies and {len(tasks)} episodes before testing", flush=True)
    return verify(output)


def verify(output):
    plan = read(output / "protocol.json")
    if plan["software"] != study.software() or plan["runner_sha256"] != study.digest(Path(__file__)):
        raise ValueError("Evaluation source or dependencies changed")
    for path, expected in [(output / "track.json", plan["track_sha256"]),
                           (output / "source-snapshot.zip", plan["source_snapshot_sha256"])]:
        if study.digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    for policy in plan["policies"].values():
        if study.digest(output / policy["file"]) != policy["sha256"]:
            raise ValueError("Frozen controller changed")
    return plan


def geometric_errors(positions, route_points):
    """Exact polyline distances, pruning segment blocks by bounding-box bounds.

    This avoids an episodes-by-whole-route matrix on the 46 m course. Every
    segment whose bounding box can beat the current minimum is still evaluated.
    """
    starts = route_points[:-1]
    delta = np.diff(route_points, axis=0)
    lengths2 = np.sum(delta * delta, axis=1)
    keep = lengths2 > 0
    starts, delta, lengths2 = starts[keep], delta[keep], lengths2[keep]
    blocks = [(starts[i:i+128], delta[i:i+128], lengths2[i:i+128]) for i in range(0, len(starts), 128)]
    lower = np.array([np.minimum(a, a+d).min(axis=0) for a, d, _ in blocks])
    upper = np.array([np.maximum(a, a+d).max(axis=0) for a, d, _ in blocks])
    def distances(points, block):
        a, d, length2 = block
        offset = points[:, None, :] - a
        projection = np.clip(np.sum(offset*d, axis=2) / length2, 0, 1)
        return np.min(np.sum((offset-projection[:, :, None]*d)**2, axis=2), axis=1)
    output = np.empty(len(positions))
    for begin in range(0, len(positions), 128):
        points = positions[begin:begin+128]
        offset = np.maximum(np.maximum(lower-points[:, None, :], points[:, None, :]-upper), 0)
        bounds = np.sum(offset**2, axis=2)
        first = np.argmin(bounds, axis=1)
        best = np.full(len(points), np.inf)
        for block in np.unique(first):
            use = first == block
            best[use] = distances(points[use], blocks[block])
        for block in range(len(blocks)):
            use = (first != block) & (bounds[:, block] <= best)
            if np.any(use):
                best[use] = np.minimum(best[use], distances(points[use], blocks[block]))
        output[begin:begin+len(points)] = np.sqrt(best)
    return output


_OUTPUT = _PLAN = _TRACK = _MASK = None


def initialize_worker(output):
    global _OUTPUT, _PLAN, _TRACK, _MASK
    _OUTPUT = Path(output)
    _PLAN = read(_OUTPUT / "protocol.json")
    _TRACK = TrackSpec.load(_OUTPUT / "track.json")
    _MASK = TrackMask(_TRACK)  # One raster, shared across this worker's episodes.


def setup(output, plan, task, mask):
    policy = plan["policies"][task["policy"]]
    robot = RobotConfig(**{**plan["robot"], "max_speed": task["cap_m_s"]})
    path = output / policy["file"]
    controller = load_model(path)[0] if policy["family"] == "NEAT" else ResearchPIDController(ResearchPIDConfig.load(path), robot)
    episode = Episode(Simulation(robot), mask.track, controller, EpisodeConfig(**plan["episode"]),
                      mask, time_limit=task["time_limit_s"])
    return episode


def evaluate(task):
    started = time.perf_counter()
    episode = setup(_OUTPUT, _PLAN, task, _MASK)
    fields = TELEMETRY_FIELDS + PID_FIELDS
    trace_path = _OUTPUT / "telemetry" / (task["id"] + ".csv.gz")
    temporary = trace_path.with_suffix(".tmp")
    positions = []
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=3) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while not episode.done:
            row = episode.advance(telemetry=True)
            if row is not None:
                if isinstance(episode.controller, ResearchPIDController):
                    row.update(episode.controller.debug)
                writer.writerow(row)
                positions.append((row["x_m"], row["y_m"]))
    temporary.replace(trace_path)
    result = asdict(episode.result())
    if positions:
        error = geometric_errors(np.asarray(positions), _TRACK.route.points)
        result["geometry_lateral_rmse_m"] = float(np.sqrt(np.mean(error**2)))
    record = {**task, **result, "policy_family": _PLAN["policies"][task["policy"]]["family"],
              "final_x_m": float(episode.simulation.state.position[0]),
              "final_y_m": float(episode.simulation.state.position[1]),
              "wall_s": time.perf_counter()-started, "finished_utc": stamp(),
              "telemetry": str(trace_path.relative_to(_OUTPUT)), "telemetry_sha256": study.digest(trace_path),
              "protocol_sha256": study.digest(_OUTPUT / "protocol.json")}
    write_json(_OUTPUT / "episodes" / (task["id"] + ".json"), record)
    return record


def run(output, workers):
    plan = freeze(output)
    (output / "episodes").mkdir(exist_ok=True)
    (output / "telemetry").mkdir(exist_ok=True)
    pending = []
    for task in plan["tasks"]:
        path = output / "episodes" / (task["id"] + ".json")
        if path.exists():
            result = read(path)
            if result["protocol_sha256"] != study.digest(output / "protocol.json") or any(result[k] != v for k, v in task.items()):
                raise ValueError("A completed episode belongs to a different protocol/task")
            if study.digest(output / result["telemetry"]) != result["telemetry_sha256"]:
                raise ValueError("Saved telemetry changed")
        else:
            pending.append(task)
    count = len(plan["tasks"])-len(pending)
    print(f"{count}/{len(plan['tasks'])} already committed; {workers} CPU workers, one floor mask per worker", flush=True)
    if pending:
        with mp.get_context("spawn").Pool(workers, initialize_worker, (str(output),), maxtasksperchild=24) as pool:
            iterator = pool.imap_unordered(evaluate, pending, chunksize=1)
            while count < len(plan["tasks"]):
                try:
                    result = iterator.next(timeout=25)
                except mp.TimeoutError:
                    print(f"Working: {count}/{len(plan['tasks'])} committed", flush=True)
                    continue
                count += 1
                print(f"{count}/{len(plan['tasks'])}: {result['id']} | {result['termination']} | "
                      f"{result['progress']:.1%} | elapsed {result['elapsed_s']:.2f}s | "
                      f"mean {result['average_forward_speed_m_s']:.3f} m/s", flush=True)
    all_rows = [read(output / "episodes" / (t["id"] + ".json")) for t in plan["tasks"]]
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(output / "completion.json", {"finished_utc": stamp(), "episodes": len(all_rows),
               "errors": sum(bool(r["error"]) for r in all_rows), "protocol_sha256": study.digest(output / "protocol.json"),
               "results_sha256": study.digest(output / "results.csv")})
    verify(output)


def replay(output, task_id):
    import mujoco.viewer
    from line_follower.viewer import add_overlays, camera_for, set_transparency
    plan = verify(output)
    task = next((t for t in plan["tasks"] if t["id"] == task_id), None)
    if task is None:
        raise ValueError("Choose an episode id from results.csv / protocol.json")
    mask = TrackMask(TrackSpec.load(output / "track.json"))
    episode = setup(output, plan, task, mask)
    sim, controller = episode.simulation, episode.controller
    sim.enable_rendering()
    def reset():
        return Episode(sim, mask.track, controller, EpisodeConfig(**plan["episode"]), mask, time_limit=task["time_limit_s"])
    episode = reset()
    print(f"{task_id} | controller={type(controller).__name__} | cap={task['cap_m_s']:.2f} m/s | "
          f"source={plan['policies'][task['policy']]['file']}", flush=True)
    print("SPACE: pause | R: reset | T: route | B: transparency. Replay does not overwrite results.", flush=True)
    keys, trail = SimpleQueue(), deque(maxlen=2400)
    paused = reference = transparent = False
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys.put) as viewer:
        camera = camera_for(sim, "track")
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        while viewer.is_running():
            tick = time.perf_counter()
            while not keys.empty():
                key = keys.get()
                if key == 32: paused = not paused
                elif key == ord("R"):
                    episode, paused = reset(), False
                    trail.clear()
                elif key == ord("T"): reference = not reference
                elif key == ord("B"):
                    transparent = not transparent
                    set_transparency(sim, transparent)
            if not paused and not episode.done:
                episode.advance()
                trail.append(sim.state.position.copy())
                if episode.done:
                    print(asdict(episode.result()), flush=True)
                    paused = True
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                add_overlays(viewer.user_scn, sim, reference, list(trail))
            viewer.sync()
            time.sleep(max(0, sim.config.dt-(time.perf_counter()-tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify", "replay"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episode", help="Frozen episode id to replay")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.workers < 1: parser.error("workers must be positive")
    if args.command == "prepare": freeze(output)
    elif args.command == "run": run(output, args.workers)
    elif args.command == "verify":
        plan = verify(output)
        print(f"Verified frozen course and {len(plan['policies'])} controllers; {len(plan['tasks'])} tasks")
    elif args.command == "replay": replay(output, args.episode)


if __name__ == "__main__":
    main()
