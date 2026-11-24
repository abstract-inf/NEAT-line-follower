"""Refine recovery PID gains at 1.00 and 1.50 m/s without using the NEAT test set."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from line_follower.experiments import append_csv, snapshot_sources, write_json
from line_follower.pid import ResearchPIDConfig
from line_follower.suites import derived_seed, load_split, read_suite
from run_corner_training import protected_tracks
from run_neat_speed_study import OUTPUT as NEAT_STUDY
from run_neat_speed_study import verify_prepared
from run_pid_gain_tuning import GAIN_NAMES, PROJECT_ROOT, digest, evaluate_batch, prepare_tracks, protected_geometry

OUTPUT = PROJECT_ROOT / "runs/pid-speed-followup-20261116"
CAPS = (1.0, 1.5)
SUITE_SEED = 2026111601
BOUNDS = ((.15, 10.0), (0.0, .65), (0.0, .25))
COARSE_CANDIDATES = 192
PROMOTED = 24
FINALISTS = 12
REFINEMENT_STEPS = ((.55, .09, .025), (.21, .035, .009))
REFINEMENTS = 24


def selection_key(rows):
    return (sum(row["completed"] for row in rows), mean(row["progress"] for row in rows),
            mean(row["score"] for row in rows))


def tune(pool, references, coarse, expanded, validation, base, output, cap):
    rng = np.random.default_rng(derived_seed(SUITE_SEED, "pid-search", round(100 * cap)))
    lower, upper = np.array(BOUNDS).T
    candidates = references.copy()
    existing = {tuple(getattr(policy, name) for name in GAIN_NAMES) for policy in references.values()}
    original = references["original"]
    output.mkdir()

    def add(origin, gains):
        triple = tuple(float(gain) for gain in np.clip(gains, lower, upper))
        if triple in existing:
            return None
        existing.add(triple)
        key = f"gain-{len(candidates):03d}"
        policy = replace(original, **dict(zip(GAIN_NAMES, triple, strict=True)))
        candidates[key] = policy
        record(key, origin, policy)
        return key

    def record(key, origin, policy):
        row = {"candidate": key, "origin": origin, **asdict(policy)}
        append_csv(output / "candidates.csv", [row], list(row))

    for key, policy in references.items():
        record(key, "saved-reference", policy)
        center = np.array([getattr(policy, name) for name in GAIN_NAMES])
        for axis in range(3):
            for multiplier in (.5, .75, 1.25, 1.5, 2.0):
                point = center.copy()
                point[axis] *= multiplier
                add(f"axis-{key}", point)
        for _ in range(16):
            add(f"local-{key}", rng.normal(center, [.25 * center[0], .08, .018]))
    for kp in (.45, .75, 1.35, 2.1, 3.2, 4.6, 6.5, 9.0):
        for kd in (.004, .012, .025, .040, .065, .10, .16):
            add("pd-grid", (kp, 0, kd))
    remaining = COARSE_CANDIDATES - len(candidates)
    strata = np.array([(rng.permutation(remaining) + rng.random(remaining)) / remaining for _ in GAIN_NAMES]).T
    for index, (p, i, d) in enumerate(strata):
        add("wide", (lower[0] * (upper[0] / lower[0]) ** p, upper[1] * i if index % 3 else 0,
                     .001 * (upper[2] / .001) ** d))
    if len(candidates) != COARSE_CANDIDATES:
        raise ValueError("Unexpected coarse candidate count")

    def evaluate(keys, tracks, stage):
        return evaluate_batch(pool, candidates, keys, tracks, stage, base, output)

    def leaders(results):
        return sorted(results, key=lambda key: selection_key(results[key]), reverse=True)

    coarse_results = evaluate(list(candidates), coarse, "coarse")
    initial = list(dict.fromkeys([*references, *leaders(coarse_results)]))[:PROMOTED]
    results = evaluate(initial, expanded, "expanded")
    for phase, steps in enumerate(REFINEMENT_STEPS, 1):
        parents = leaders(results)[:6]
        children = []
        for index in range(REFINEMENTS):
            parent = parents[index % len(parents)]
            center = [getattr(candidates[parent], name) for name in GAIN_NAMES]
            for _ in range(100):
                child = add(f"refine-{phase}-{parent}", rng.normal(center, steps))
                if child is not None:
                    children.append(child)
                    break
            else:
                raise ValueError("Could not sample a distinct refinement")
        results.update(evaluate(children, expanded, f"refine-{phase}"))
    finalists = list(dict.fromkeys([*references, *leaders(results)[:FINALISTS]]))
    validation_results = evaluate(finalists, validation, "validation")
    winner = leaders(validation_results)[0]
    print(f"{cap:.2f} m/s selected {winner}. Validation ranks {[(key, selection_key(validation_results[key])) for key in dict.fromkeys([*references, winner])]}", flush=True)
    return winner, candidates[winner], selection_key(validation_results[winner]), len(candidates)


def summarize(cap, results, output):
    summaries = []
    indexed = {name: {row["track_hash"]: row for row in rows} for name, rows in results.items()}
    track_hashes = indexed["original"].keys()
    for name, rows in results.items():
        if len(rows) != 90 or indexed[name].keys() != track_hashes:
            raise ValueError("Expected 90 identical final test tracks for each policy")
        failures = Counter(row["termination"] for row in rows if not row["completed"])
        summaries.append({"cap_m_s": cap, "policy": name, "finished": sum(row["completed"] for row in rows),
                          "tracks": len(rows), "off_route": failures["off_route"], "stalled": failures["stalled"],
                          "mean_progress": mean(row["progress"] for row in rows),
                          "geometric_rmse_mm": 1000 * mean(row["geometry_lateral_rmse_m"] for row in rows),
                          "mean_command_jitter": mean(row["command_jitter"] for row in rows),
                          "mean_forward_speed_m_s": mean(row["average_forward_speed_m_s"] for row in rows)})
    comparisons = []
    for name in ("original", "prior-150"):
        paired = [(indexed["selected"][key], indexed[name][key]) for key in track_hashes]
        shared = [(a["completion_time_s"], b["completion_time_s"]) for a, b in paired if a["completed"] and b["completed"]]
        comparisons.append({"cap_m_s": cap, "reference": name, "shared_finishes": len(shared),
                            "selected_only": sum(a["completed"] and not b["completed"] for a, b in paired),
                            "reference_only": sum(b["completed"] and not a["completed"] for a, b in paired),
                            "selected_shared_time_s": mean(a for a, _ in shared) if shared else "",
                            "reference_shared_time_s": mean(b for _, b in shared) if shared else ""})
    append_csv(output / "test-summary.csv", summaries, list(summaries[0]))
    append_csv(output / "paired-summary.csv", comparisons, list(comparisons[0]))


def run(output, workers):
    started = time.perf_counter()
    prepared = NEAT_STUDY / "prepared"
    study = verify_prepared(prepared)
    prior = {"original": prepared / "inputs/pid-050.json", "prior-150": prepared / "inputs/pid-150.json"}
    references = {key: ResearchPIDConfig.load(path) for key, path in prior.items()}
    original, previous = references.values()
    if any(getattr(original, key) != getattr(previous, key) for key in asdict(original) if key not in GAIN_NAMES):
        raise ValueError("The saved PID settings differ beyond P, I, and D")
    output.mkdir(parents=True)
    write_json(output / "protocol.json", {
        "caps_m_s": CAPS, "suite_seed": SUITE_SEED, "workers": workers, "bounds": BOUNDS,
        "neat_protocol_sha256": digest(prepared / "protocol.json"),
        "robot": study["robot"], "episode": study["episode"], "software": study["software"],
        "reference_sha256": {key: digest(path) for key, path in prior.items()},
        "coarse_candidates": COARSE_CANDIDATES, "promoted": PROMOTED,
        "refinements": REFINEMENTS, "refinement_steps": REFINEMENT_STEPS, "finalists": FINALISTS,
        "selection": ["higher completion count", "higher mean progress", "higher mean study fitness"],
    })
    snapshot_sources(output / "source.zip")
    with ZipFile(output / "source.zip", "a", compression=ZIP_DEFLATED) as snapshot:
        for name in (Path(__file__).name, "run_pid_gain_tuning.py", "run_neat_speed_study.py"):
            snapshot.write(PROJECT_ROOT / name, name)
    print("Preparing PID-only development, validation, and held-out tracks.", flush=True)
    suite, _, expanded = prepare_tracks(output, protected_tracks() | protected_geometry(), seed=SUITE_SEED)
    coarse = [expanded[i] for i in (1, 3, 4, *range(9, 21))]
    splits = read_suite(suite)["splits"]
    load_split(suite, "validation")
    paths = lambda split: [str(suite.parent / row["path"]) for row in splits[split]]
    selections = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for cap in CAPS:
            directory = output / f"cap-{cap:.2f}"
            base = {"robot_json": json.dumps({**study["robot"], "max_speed": cap}, sort_keys=True),
                    "episode": study["episode"], "seed": SUITE_SEED, "start_wall": started, "output": str(directory)}
            print(f"Searching gains at {cap:.2f} m/s.", flush=True)
            key, winner, rank, count = tune(pool, references, coarse, expanded, paths("validation"), base, directory, cap)
            config = output / f"tuned-{cap:.2f}.json"
            winner.save(config)
            selections[f"{cap:.2f}"] = {"candidate": key, "config": config.name, "sha256": digest(config),
                                        "validation_rank": rank, "candidates": count}
        write_json(output / "selection-freeze.json", {"suite_sha256": digest(suite), "selections": selections,
                                                      "frozen_before_test": True})
        load_split(suite, "test")
        for cap in CAPS:
            directory = output / f"cap-{cap:.2f}"
            saved = selections[f"{cap:.2f}"]
            config = output / saved["config"]
            if digest(config) != saved["sha256"]:
                raise ValueError("Selected gains changed after freezing")
            policies = {**references, "selected": ResearchPIDConfig.load(config)}
            base = {"robot_json": json.dumps({**study["robot"], "max_speed": cap}, sort_keys=True),
                    "episode": study["episode"], "seed": SUITE_SEED, "start_wall": started, "output": str(directory),
                    "geometric_tracking": True}
            print(f"Assessing frozen gains at {cap:.2f} m/s.", flush=True)
            results = evaluate_batch(pool, policies, list(policies), paths("test"), "test", base, directory)
            summarize(cap, results, output)
    write_json(output / "status.json", {"complete": True, "test_episodes": 540,
                                       "wall_seconds": time.perf_counter() - started})
    print(f"Finished. Results saved in {output}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.output.resolve(), args.workers)
