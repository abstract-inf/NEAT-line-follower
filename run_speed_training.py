"""Warm start a new NEAT population from frozen g94 and increase its speed cap."""

import argparse
import csv
from dataclasses import asdict, replace
import json
import multiprocessing as mp
from pathlib import Path
import time

from line_follower.config import EpisodeConfig, RobotConfig, verify_geometry
from line_follower.controllers import load_model
from line_follower.experiments import TrainConfig, append_csv, environment_record, train, write_json
from line_follower.pid_experiments import evaluate_candidate
from line_follower.suites import create_suite, load_split, read_suite
from run_corner_training import ROOT, RUNS, digest, protected_tracks, read_json, session_lock


PARENT = RUNS / "neat-corner-100-20261104/assessment"
SOURCE_MODEL = PARENT / "frozen-best.pkl"
GEOMETRY_LOCK = RUNS / "28-10-2026-hard/geometry-freeze.json"
OUTPUT = RUNS / "neat-speed-100-20261104"
SUITE_SEED = 2026110404
TRACE_SLOTS = (0, 1, 2, 30, 31, 32)
SETTINGS = TrainConfig(
    strategy="multi", seed=2026110405, generations=100, population=100, workers=8,
    backend="cpu", episodes_per_genome=4, track_profile="zigzag",
    aggregation="mean_cvar", cvar_fraction=.5,
    speed_schedule=((0, .50), (10, .65), (25, .80), (40, 1.00), (55, 1.25), (70, 1.50)),
)


def protocol():
    freeze = read_json(PARENT / "selection-freeze.json")
    _, metadata = load_model(SOURCE_MODEL)
    robot = RobotConfig(**metadata["robot"])
    episode = EpisodeConfig(**metadata["episode"])
    verify_geometry(GEOMETRY_LOCK, robot)
    if metadata["generation"] != 94 or digest(SOURCE_MODEL) != freeze["new_neat_sha256"]:
        raise ValueError("Expected the original frozen g94 model")
    environment = environment_record()
    plan = {
        "study": "neat-warm-start-speed-1", "training": asdict(SETTINGS),
        "warm_start": {"generation": 94, "evolution_seed": metadata["evolution_seed"],
                       "sha256": freeze["new_neat_sha256"],
                       "exact_copies": 1, "mutated_copies": SETTINGS.population - 1},
        "base_robot": asdict(robot), "episode": asdict(episode), "suite_seed": SUITE_SEED,
        "validation_tracks": 30, "test_tracks": 90, "trace_slots": TRACE_SLOTS,
        "assessment_caps_m_s": [SETTINGS.speed_schedule[0][1], SETTINGS.speed_schedule[-1][1]],
        "selection": "Final speed phase only. Rank by validation completion, then validation fitness.",
        "source_sha256": environment["source_sha256"], "dependencies": environment["dependencies"],
        "runner_sha256": digest(Path(__file__)), "shared_launcher_sha256": digest(ROOT / "run_corner_training.py"),
    }
    return json.loads(json.dumps(plan)), robot, episode


def assess(output, suite, plan, start, best, robot, episode):
    """Compare both frozen policies at both caps on the same untouched courses."""
    assessment = output / "assessment"
    model = assessment / "frozen-best.pkl"
    freeze_path = assessment / "selection-freeze.json"
    status_path = assessment / "status.json"
    results_path = assessment / "test-episodes.csv"
    _, metadata = load_model(best)
    final_start, final_cap = SETTINGS.speed_schedule[-1]
    if metadata["robot"] != asdict(replace(robot, max_speed=final_cap)) or metadata["generation"] < final_start:
        raise ValueError("The selected model must come from the final speed phase")
    entries = read_suite(suite)["splits"]["test"]
    freeze = {
        "frozen_before_test_geometry_load": True, "g94_sha256": digest(start), "final_sha256": digest(best),
        "selected_generation": metadata["generation"], "validation_key": list(metadata["validation_key"]),
        "suite_sha256": digest(suite), "test_hashes": [e["geometry_hash"] for e in entries],
        "caps_m_s": plan["assessment_caps_m_s"], "trace_slots": list(TRACE_SLOTS),
    }
    if freeze_path.exists():
        if read_json(freeze_path) != freeze or digest(model) != freeze["final_sha256"]:
            raise ValueError("Frozen assessment inputs have changed")
    else:
        assessment.mkdir(exist_ok=True)
        model.write_bytes(best.read_bytes())
        write_json(freeze_path, freeze)

    total = 4 * len(entries)
    if results_path.exists():
        results_hash = digest(results_path)
        if status_path.exists():
            if read_json(status_path)["results_sha256"] != results_hash:
                raise ValueError("Saved final assessment differs from its checksum")
        else:
            with results_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != total or any(row["error"] for row in rows):
                raise ValueError("Saved final assessment is incomplete")
            write_json(status_path, {"complete": True, "episodes": total, "results_sha256": results_hash})
        print(f"Final assessment already saved: {results_path}", flush=True)
        return

    tracks = load_split(suite, "test")
    jobs = []
    for slot, entry in enumerate(entries):
        for cap in freeze["caps_m_s"]:
            for policy, path, seed in (("g94-parent", start, plan["warm_start"]["evolution_seed"]),
                                       ("speed-trained", model, SETTINGS.seed)):
                task = {"neat_model": str(path), "robot_json": json.dumps(asdict(replace(robot, max_speed=cap)), sort_keys=True),
                        "episode": asdict(episode), "seed": SUITE_SEED, "phase": "test",
                        "output": str(assessment / f"{policy}-{cap:.2f}"),
                        "tracks": [str(suite.parent / entry["path"])], "slot_offset": slot, "trace_slots": TRACE_SLOTS}
                jobs.append((policy, seed, task))
    started = time.perf_counter()
    records = []
    print(f"Frozen generation {metadata['generation']}. Starting {total} paired test episodes.", flush=True)
    with mp.get_context("spawn").Pool(SETTINGS.workers) as pool:
        results = pool.imap(evaluate_candidate, [task for _, _, task in jobs])
        for index, ((policy, seed, task), batch) in enumerate(zip(jobs, results, strict=True), 1):
            row, = batch
            slot = task["slot_offset"]
            if row["track_hash"] != tracks[slot].geometry_hash:
                raise RuntimeError("Unexpected final-test geometry")
            records.append({**row, "policy_id": policy, "candidate": policy, "track_slot": slot,
                            "tuning_seed": "", "evolution_seed": seed, "evaluation_seed": SUITE_SEED})
            if index % 45 == 0:
                print(f"Final assessment {index}/{total} episodes", flush=True)
    temporary = results_path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    append_csv(temporary, records, list(records[0]))
    if any(r["error"] for r in records):
        temporary.replace(assessment / "technical-failures.csv")
        raise RuntimeError("Final assessment is incomplete. See technical-failures.csv.")
    if (digest(start) != freeze["g94_sha256"] or digest(model) != freeze["final_sha256"]
            or digest(suite) != freeze["suite_sha256"]):
        raise RuntimeError("A frozen input changed during the final assessment")
    temporary.replace(results_path)
    write_json(status_path, {"complete": True, "episodes": total, "results_sha256": digest(results_path),
                             "wall_seconds": time.perf_counter() - started})
    print(f"Final assessment saved: {results_path}", flush=True)


def run(output, plan, robot, episode, resume):
    start = output / "initial-g94.pkl"
    suite = output / "suite/suite.json"
    training = output / "training"
    plan_path = output / "protocol.json"
    checkpoint = training / "checkpoints/latest.pkl.gz"
    status = training / "status.json"
    if resume:
        if read_json(plan_path) != {**plan, "suite_sha256": digest(suite)} or digest(start) != plan["warm_start"]["sha256"]:
            raise ValueError("Frozen protocol, suite or parent model differs from the original run")
        if not checkpoint.is_file() and training.exists() and any(training.iterdir()):
            raise ValueError("No generation checkpoint exists yet. Start with a fresh --output folder.")
    else:
        if any(path.name != "session.lock" for path in output.iterdir()):
            raise FileExistsError(f"Choose a fresh --output or use --resume for {output}")
        print(f"Preparing frozen suite for {output}", flush=True)
        create_suite(suite.parent, seed=SUITE_SEED, validation_count=30, test_count=90,
                     protected_hashes=protected_tracks())
        start.write_bytes(SOURCE_MODEL.read_bytes())
        if digest(start) != plan["warm_start"]["sha256"]:
            raise ValueError("The copied warm-start model differs from the frozen g94 model")
        write_json(plan_path, {**plan, "suite_sha256": digest(suite)})
    if not status.exists() or not read_json(status)["complete"]:
        train(training, suite, SETTINGS, robot, episode, geometry_lock=GEOMETRY_LOCK, live_preview=True,
              resume=checkpoint if checkpoint.exists() else None,
              initial_model=None if checkpoint.exists() else start)

    with (training / "generations.csv").open(newline="", encoding="utf-8") as stream:
        generations = list(csv.DictReader(stream))
    expected = [(g, SETTINGS.speed_at(g, robot.max_speed)) for g in range(SETTINGS.generations)]
    actual = [(int(r["generation"]), float(r["wheel_speed_cap_m_s"])) for r in generations]
    if not read_json(status)["complete"] or actual != expected:
        raise RuntimeError("Final assessment requires all scheduled training generations")
    assess(output, suite, plan, start, training / "best.pkl", robot, episode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true", help="Resume from this run's last committed checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print and validate the protocol without starting")
    args = parser.parse_args()
    plan, robot, episode = protocol()
    output = args.output.resolve()
    if args.dry_run:
        print(json.dumps({**plan, "output": str(output), "training_started": False}, indent=2))
        return
    with session_lock(output):
        run(output, plan, robot, episode, args.resume)


if __name__ == "__main__":
    mp.freeze_support()
    main()
