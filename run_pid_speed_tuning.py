"""Tune the saved recovery PID independently at two wheel speed caps."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import time
import zipfile

import numpy as np

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.experiments import append_csv, environment_record, snapshot_sources, write_json
from line_follower.fitness import FITNESS_VERSION
from line_follower.pid import ResearchPIDConfig
from line_follower.pid_experiments import ranking
from line_follower.suites import derived_seed, load_split
from run_corner_training import protected_tracks
from run_pid_gain_tuning import (
    GAIN_BOUNDS,
    GAIN_NAMES,
    PROJECT_ROOT,
    digest,
    evaluate_batch,
    gain_config,
    initial_parameters,
    prepare_tracks,
    protected_geometry,
    read_json,
)

BASELINE = PROJECT_ROOT / "runs/pid-gains-20261029/selected-pid_filtered.json"
ROBOT_PROTOCOL = PROJECT_ROOT / "runs/neat-speed-100-20261104/protocol.json"
EVALUATOR = PROJECT_ROOT / "research/artifacts/neat-speed-results-corrected/evaluation-freeze.json"
CAPS = (0.50, 1.50)
SUITE_SEED = 2026111501
PROMOTED = 16
FINALISTS = 8
REFINEMENT_STEPS = ((0.20, 0.05, 0.025), (0.07, 0.02, 0.009))
REFINEMENTS = 24


def tune(pool, baseline, initial, coarse, expanded, validation, base, output, cap_index):
    candidates = {}
    existing = set()

    def add(origin, config):
        key = f"gain-{len(candidates):03d}"
        candidates[key] = config
        existing.add(tuple(getattr(config, name) for name in GAIN_NAMES))
        row = {"candidate": key, "origin": origin, **asdict(config)}
        append_csv(output / "candidates.csv", [row], list(row))
        return key

    output.mkdir()
    baseline_id = add("baseline", baseline)
    evaluate_batch(pool, candidates, [baseline_id], coarse, "baseline", base, output)
    initial_ids = [baseline_id, *(add(origin, config) for origin, config in initial[1:])]
    coarse_results = evaluate_batch(pool, candidates, initial_ids, coarse, "coarse", base, output)
    leaders = sorted(coarse_results, key=lambda key: ranking(coarse_results[key]), reverse=True)
    promoted = list(dict.fromkeys([baseline_id, *leaders]))[:PROMOTED]
    expanded_results = evaluate_batch(pool, candidates, promoted, expanded, "expanded", base, output)

    lower, upper = np.array(list(GAIN_BOUNDS.values())).T
    rng = np.random.default_rng(derived_seed(SUITE_SEED, "speed-gain-refinement", cap_index))
    for phase, steps in enumerate(REFINEMENT_STEPS, 1):
        parents = sorted(expanded_results, key=lambda key: ranking(expanded_results[key]), reverse=True)[:4]
        new_ids = []
        for index in range(REFINEMENTS):
            parent = parents[index % len(parents)]
            center = np.array([getattr(candidates[parent], name) for name in GAIN_NAMES])
            for _ in range(100):
                gains = np.clip(rng.normal(center, steps), lower, upper)
                if tuple(gains) not in existing:
                    break
            else:
                raise ValueError("Could not generate a distinct refinement")
            new_ids.append(add(f"refine-{phase}-{parent}", gain_config(baseline, gains)))
        expanded_results.update(evaluate_batch(pool, candidates, new_ids, expanded, f"refine-{phase}", base, output))

    leaders = sorted(expanded_results, key=lambda key: ranking(expanded_results[key]), reverse=True)
    finalists = list(dict.fromkeys([baseline_id, *leaders[:FINALISTS]]))
    results = evaluate_batch(pool, candidates, finalists, validation, "validation", base, output)
    winner = max(results, key=lambda key: ranking(results[key]))
    print(f"Selected {winner} at {CAPS[cap_index]:.2f} m/s with validation rank {ranking(results[winner])}", flush=True)
    return candidates[winner], winner, ranking(results[winner])


def summarize(results, cap, output):
    original = {row["track_hash"]: row for row in results["original"]}
    summaries = []
    for key, rows in results.items():
        if len(rows) != len(original) or {row["track_hash"] for row in rows} != original.keys():
            raise ValueError("Test results must contain identical tracks for every gain set")
        completed = [row for row in rows if row["completed"]]
        shared = [row for row in completed if original[row["track_hash"]]["completed"]]
        failures = Counter(row["termination"] for row in rows if not row["completed"])
        summaries.append({
            "cap_m_s": cap, "policy": key, "finished": len(completed), "tracks": len(rows),
            "off_route": failures["off_route"], "stalled": failures["stalled"],
            "other_failures": sum(failures.values()) - failures["off_route"] - failures["stalled"],
            "mean_progress": ranking(rows)[1], "route_local_rmse_mm": -1000 * ranking(rows)[2],
            "shared_finishes_with_original": len(shared),
            "shared_time_s": np.mean([row["completion_time_s"] for row in shared]) if shared else "",
            "original_shared_time_s": np.mean([original[row["track_hash"]]["completion_time_s"] for row in shared]) if shared else "",
            "mean_forward_speed_m_s": np.mean([row["average_forward_speed_m_s"] for row in rows]),
        })
    append_csv(output / "test-summary.csv", summaries, list(summaries[0]))


def run(output, workers):
    started = time.perf_counter()
    baseline = ResearchPIDConfig.load(BASELINE)
    evaluator = read_json(EVALUATOR)
    if FITNESS_VERSION != evaluator["evaluator_version"] or digest(BASELINE) != evaluator["pid_sha256"]:
        raise ValueError("Saved PID or corrected evaluator version does not match the reference")
    episode = asdict(EpisodeConfig(**evaluator["episode"]))
    robot = RobotConfig(**read_json(ROBOT_PROTOCOL)["base_robot"])
    initial = initial_parameters(baseline)
    if len(initial) != len({tuple(getattr(config, name) for name in GAIN_NAMES) for _, config in initial}):
        raise ValueError("Coarse search contains duplicate gains")
    output.mkdir(parents=True)
    baseline.save(output / "original.json")
    write_json(output / "protocol.json", {
        "caps_m_s": CAPS, "original_tuning_cap_m_s": evaluator["pid_tuning_cap_m_s"],
        "baseline_sha256": digest(BASELINE), "baseline": asdict(baseline),
        "base_robot": asdict(robot), "episode": episode, "environment": environment_record(),
        "suite_seed": SUITE_SEED, "workers": workers, "gain_bounds": GAIN_BOUNDS,
        "coarse_candidates": len(initial), "expanded_candidates": PROMOTED,
        "refinements_per_round": REFINEMENTS, "refinement_steps": REFINEMENT_STEPS,
        "validation_challengers": FINALISTS, "selection": "completion, progress, lower route-local RMSE, faster finishes",
    })
    snapshot_sources(output / "source.zip")
    with zipfile.ZipFile(output / "source.zip", "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (Path(__file__).name, "run_pid_gain_tuning.py"):
            archive.write(PROJECT_ROOT / name, name)
    suite, coarse, expanded = prepare_tracks(output, protected_tracks() | protected_geometry(), seed=SUITE_SEED)
    validation = [str(suite.parent / "validation" / f"{i:03d}.json") for i, _ in enumerate(load_split(suite, "validation"))]
    policies = {"original": baseline}
    selections = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for index, cap in enumerate(CAPS):
            key = f"tuned-{cap:.2f}"
            directory = output / f"cap-{cap:.2f}"
            base = {"robot_json": json.dumps(asdict(replace(robot, max_speed=cap)), sort_keys=True),
                    "episode": episode, "seed": SUITE_SEED, "start_wall": started, "output": str(directory)}
            print(f"Tuning recovery PID at {cap:.2f} m/s", flush=True)
            policy, winner, rank = tune(pool, baseline, initial, coarse, expanded, validation, base, directory, index)
            policies[key] = policy
            config_path = output / f"{key}.json"
            policy.save(config_path)
            selections[key] = {"winner": winner, "validation_rank": rank, "sha256": digest(config_path),
                               "gains": {name: getattr(policy, name) for name in GAIN_NAMES}}
        write_json(output / "selection-freeze.json", {
            "frozen_utc": datetime.now(timezone.utc).isoformat(), "suite_sha256": digest(suite),
            "selections": selections,
        })
        test = [str(suite.parent / "test" / f"{i:03d}.json") for i, _ in enumerate(load_split(suite, "test"))]
        for cap in CAPS:
            directory = output / f"cap-{cap:.2f}"
            base = {"robot_json": json.dumps(asdict(replace(robot, max_speed=cap)), sort_keys=True),
                    "episode": episode, "seed": SUITE_SEED, "start_wall": started, "output": str(directory),
                    "trace_slots": [0, 1, 2, 31, 62]}
            print(f"Assessing all frozen gains at {cap:.2f} m/s", flush=True)
            results = evaluate_batch(pool, policies, list(policies), test, "test", base, directory)
            summarize(results, cap, output)
    write_json(output / "status.json", {"complete": True, "selections": selections,
                                       "test_episodes": len(test) * len(policies) * len(CAPS),
                                       "wall_seconds": time.perf_counter() - started})
    print(f"Completed PID speed tuning: {output}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/pid-speed-tuning-20261115")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.output.resolve(), args.workers)
