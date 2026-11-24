"""Tune only P/I/D of frozen PID variants, then assess a new held-out cohort.

Editable research controls are below. Existing runs are never overwritten.
Controller implementations are reused unchanged; no NEAT training is performed.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import zipfile

import numpy as np

from line_follower.experiments import append_csv, environment_record, snapshot_sources, write_json
from line_follower.pid import ResearchPIDConfig
from line_follower.pid_experiments import evaluate_candidate, ranking
from line_follower.suites import create_suite, derived_seed, load_split, read_suite
from line_follower.tracks import DIFFICULTIES, TrackSpec, generate_track, validate_track

PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_RUN = PROJECT_ROOT / "runs/pid-research-20261029"
OUTPUT_DIRECTORY = "runs/pid-gains-20261029"
PROTOCOL = PROJECT_ROOT / "research/PID_Gain_Tuning_Protocol.md"
WORKERS = 4
SUITE_SEED = 2026102902
SEARCH_SEEDS = (11, 29, 47)
RANDOM_CANDIDATES_PER_SEED = 48
PROMOTED_PER_FAMILY = 12
REFINEMENTS_PER_FAMILY = 24
VALIDATION_CANDIDATES_PER_FAMILY = 3  # The unchanged baseline is also validated.
TRAINING_PER_DIFFICULTY = 12
VALIDATION_TRACKS = 30
TEST_TRACKS = 90
TRACK_CHUNK_SIZE = 15  # Fits the unchanged evaluator's 16-mask worker cache.
GAIN_BOUNDS = {"kp": (.1, 5.0), "ki": (0.0, .4), "kd": (0.0, .25)}
GAIN_NAMES = tuple(GAIN_BOUNDS)
FAMILIES = ("pid_filtered", "pid_corner_supervised")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gain_config(baseline: ResearchPIDConfig, gains) -> ResearchPIDConfig:
    """Only these three fields may differ from the reference configuration."""
    values = tuple(gains)
    if len(values) != 3:
        raise ValueError("Exactly three gains are required")
    parameters = {name: float(value) for name, value in zip(GAIN_NAMES, values)}
    if any(not GAIN_BOUNDS[n][0] <= v <= GAIN_BOUNDS[n][1] for n, v in parameters.items()):
        raise ValueError("Gain triple is outside the prespecified bounds")
    return replace(baseline, **parameters)


def initial_parameters(baseline: ResearchPIDConfig) -> list[tuple[str, ResearchPIDConfig]]:
    """Baseline, one-axis checks, and local/wider reproducible random samples."""
    center = np.array([getattr(baseline, name) for name in GAIN_NAMES])
    samples = [("baseline", baseline)]
    for axis, name in enumerate(GAIN_NAMES):
        multipliers = (0, .5, .8, 1.2, 1.5) if name == "ki" else (.5, .75, 1.25, 1.5, 2)
        for factor in multipliers:
            gains = center.copy()
            gains[axis] *= factor
            samples.append((f"axis-{name}-{factor}", gain_config(baseline, gains)))
    lower, upper = np.array(list(GAIN_BOUNDS.values())).T
    for seed in SEARCH_SEEDS:
        rng = np.random.default_rng(derived_seed(seed, baseline.family))
        half = RANDOM_CANDIDATES_PER_SEED//2
        for _ in range(half):
            gains = np.clip(center+rng.normal(size=3)*[.35*center[0], .07, .04], lower, upper)
            samples.append((f"local-{seed}", gain_config(baseline, gains)))
        # Each dimension covers its whole interval without another dependency.
        strata = np.array([(rng.permutation(half)+rng.random(half))/half for _ in GAIN_NAMES]).T
        for index, unit in enumerate(strata):
            gains = lower+unit*(upper-lower)
            if index % 3 == 0:
                gains[1] = 0  # Explicitly consider PD as well as PID.
            samples.append((f"wide-{seed}", gain_config(baseline, gains)))
    return samples


def preserve_old_records() -> dict[str, str]:
    """Record the completed PID study and historical NEAT CSV identities."""
    paths = [p for p in BASELINE_RUN.rglob("*") if p.is_file()]
    manifest = read_json(PROJECT_ROOT / "research/archive/neat-20261029/manifest.json")
    paths.extend(PROJECT_ROOT / r["path"] for r in manifest["files"] if r["path"].endswith(".csv"))
    return {p.relative_to(PROJECT_ROOT).as_posix(): digest(p) for p in sorted(set(paths))}


def protected_geometry() -> set[str]:
    """Exclude known old routes; controller outcomes are never consulted."""
    identities = set()
    paths = list((PROJECT_ROOT / "benchmarks").rglob("*.json"))
    paths.extend(p for p in (PROJECT_ROOT / "runs").rglob("*.json") if "tracks" in p.parts)
    for path in paths:
        values = read_json(path)
        if "sections" in values:
            identities.add(TrackSpec.from_dict(values).geometry_hash)
    return identities


def prepare_tracks(output: Path, protected: set[str], *, seed: int = SUITE_SEED) -> tuple[Path, list[str], list[str]]:
    suite = create_suite(output / "suite", seed=seed, validation_count=VALIDATION_TRACKS,
                         test_count=TEST_TRACKS, protected_hashes=protected)
    held_out = {r["geometry_hash"] for rows in read_suite(suite)["splits"].values() for r in rows}
    tracks = [TrackSpec.load(BASELINE_RUN / "tracks/training" / f"{i:03d}.json") for i in range(9)]
    used = protected | held_out | {t.geometry_hash for t in tracks}
    attempts = []
    for slot in range(TRAINING_PER_DIFFICULTY):
        for difficulty in DIFFICULTIES:
            for attempt in range(100):
                track_seed = derived_seed(seed, "gain-training", slot, DIFFICULTIES.index(difficulty), attempt)
                track = replace(generate_track(track_seed, difficulty), split="training")
                if track.geometry_hash not in used:
                    break
                attempts.append({"seed": track_seed, "reason": "duplicate or protected geometry"})
            else:
                raise RuntimeError("Could not generate disjoint training geometry")
            if validate_track(track):
                raise ValueError(f"Invalid training geometry: {track_seed}")
            used.add(track.geometry_hash)
            tracks.append(track)
    paths = []
    for slot, track in enumerate(tracks):
        if track.geometry_hash in held_out:
            raise ValueError("Training overlaps a new held-out route")
        path = output / "tracks/training" / f"{slot:03d}.json"
        track.save(path)
        paths.append(str(path.resolve()))
    write_json(output / "training-manifest.json", {"coarse_slots": list(range(15)), "rejections": attempts,
        "tracks": [{"path": p, "geometry_hash": t.geometry_hash, "seed": t.seed,
                    "difficulty": t.difficulty, "features": t.features} for p, t in zip(paths, tracks)]})
    return suite, paths[:15], paths


def evaluate_batch(pool, candidates: dict, identifiers: list[str], tracks: list[str],
                   phase: str, base: dict, output: Path) -> dict[str, list[dict]]:
    """Group by small shared track chunks so cached reflectance fields are reused."""
    tasks = []
    for start in range(0, len(tracks), TRACK_CHUNK_SIZE):
        for key in identifiers:
            tasks.append({**base, "candidate": key, "parameters": asdict(candidates[key]),
                          "tracks": tracks[start:start+TRACK_CHUNK_SIZE], "slot_offset": start, "phase": phase})
    results = {key: [] for key in identifiers}
    for index, records in enumerate(pool.imap(evaluate_candidate, tasks), 1):
        append_csv(output / f"{phase}-episodes.csv", records, list(records[0]))
        if any(r["error"] for r in records):
            raise RuntimeError(f"Technical error recorded in {phase}; do not select from incomplete data")
        results[records[0]["candidate"]].extend(records)
        if index % 16 == 0 or index == len(tasks):
            print(f"PID gains {phase}: {index}/{len(tasks)} tasks | wall {time.perf_counter()-base['start_wall']:.1f}s", flush=True)
    summaries = []
    for key, rows in results.items():
        rank = ranking(rows)
        summaries.append({"phase": phase, "candidate": key, "family": candidates[key].family,
                          "episodes": len(rows), "completed": sum(r["completed"] for r in rows),
                          "completion_rate": rank[0], "mean_progress": rank[1], "mean_rmse_m": -rank[2],
                          "successful_time_s": -rank[3] if any(r["completed"] for r in rows) else ""})
    append_csv(output / "candidate-summary.csv", summaries, list(summaries[0]))
    return results


def family_best(candidates: dict, results: dict, family: str, count: int) -> list[str]:
    keys = [key for key in results if candidates[key].family == family]
    return sorted(keys, key=lambda k: ranking(results[k]), reverse=True)[:count]


def run(output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Choose a fresh output; previous research records are immutable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    original = read_json(BASELINE_RUN / "run.json")
    environment = environment_record()
    runner_hash, protocol_hash = digest(Path(__file__)), digest(PROTOCOL)
    baselines = {family: ResearchPIDConfig.load(BASELINE_RUN / f"selected-{family}.json") for family in FAMILIES}
    old_records = preserve_old_records()
    write_json(output / "preserved-records.json", old_records)
    write_json(output / "run.json", {"study": "pid-gains-only-1", "environment": environment,
        "robot": original["robot"], "episode": original["episode"], "baseline_run": str(BASELINE_RUN),
        "baseline_parameters": {f: asdict(p) for f, p in baselines.items()}, "runner_sha256": runner_hash,
        "protocol_sha256": protocol_hash, "workers": WORKERS, "suite_seed": SUITE_SEED,
        "search_seeds": SEARCH_SEEDS, "gain_bounds": GAIN_BOUNDS,
        "budget": {"random_per_seed_per_family": RANDOM_CANDIDATES_PER_SEED,
                   "promoted_per_family": PROMOTED_PER_FAMILY,
                   "refinements_per_family": REFINEMENTS_PER_FAMILY,
                   "validation_per_family_plus_baseline": VALIDATION_CANDIDATES_PER_FAMILY,
                   "training_per_difficulty": TRAINING_PER_DIFFICULTY,
                   "validation_tracks": VALIDATION_TRACKS, "test_tracks": TEST_TRACKS},
        "selection": "completion, progress, negative RMSE, negative successful time"})
    snapshot_sources(output / "source.zip")
    with zipfile.ZipFile(output / "source.zip", "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (Path(__file__), PROTOCOL, PROJECT_ROOT / "tests/test_pid_gain_tuning.py"):
            archive.write(path, path.relative_to(PROJECT_ROOT))
    suite, coarse, expanded = prepare_tracks(output, protected_geometry())
    candidates, baseline_ids, provenance = {}, {}, {}

    def add(family: str, parameters: ResearchPIDConfig, origin: str) -> str:
        # Fail visibly if a future edit accidentally changes a non-gain parameter.
        assert all(getattr(parameters, n) == getattr(baselines[family], n)
                   for n in asdict(parameters) if n not in GAIN_NAMES)
        key = f"gain-{len(candidates):03d}"
        candidates[key], provenance[key] = parameters, origin
        row = {"candidate": key, "family": family, "origin": origin, **asdict(parameters)}
        append_csv(output / "candidates.csv", [row], list(row))
        return key

    for family, baseline in baselines.items():
        for origin, parameters in initial_parameters(baseline):
            key = add(family, parameters, origin)
            if origin == "baseline":
                baseline_ids[family] = key
    base = {"robot_json": json.dumps(original["robot"], sort_keys=True), "episode": original["episode"],
            "output": str(output.resolve()), "seed": SUITE_SEED, "start_wall": started}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        coarse_results = evaluate_batch(pool, candidates, list(candidates), coarse, "coarse", base, output)
        promoted = []
        for family in FAMILIES:
            keys = family_best(candidates, coarse_results, family, PROMOTED_PER_FAMILY)
            if baseline_ids[family] not in keys:
                keys[-1] = baseline_ids[family]
            promoted.extend(keys)
        expanded_results = evaluate_batch(pool, candidates, promoted, expanded, "expanded", base, output)
        refinement_ids = []
        for family in FAMILIES:
            parents = family_best(candidates, expanded_results, family, 3)
            rng = np.random.default_rng(derived_seed(SUITE_SEED, "refinements", FAMILIES.index(family)))
            lower, upper = np.array(list(GAIN_BOUNDS.values())).T
            existing = {tuple(getattr(p, n) for n in GAIN_NAMES) for p in candidates.values() if p.family == family}
            for index in range(REFINEMENTS_PER_FAMILY):
                parent = candidates[parents[index % len(parents)]]
                center = np.array([getattr(parent, n) for n in GAIN_NAMES])
                for _ in range(100):
                    gains = np.clip(center+rng.normal(size=3)*[.12*center[0], .03, .02], lower, upper)
                    if tuple(gains) not in existing:
                        break
                else:
                    raise ValueError("Could not sample a distinct local gain triple")
                existing.add(tuple(gains))
                refinement_ids.append(add(family, gain_config(baselines[family], gains), f"refine-{parents[index % len(parents)]}"))
        refined = evaluate_batch(pool, candidates, refinement_ids, expanded, "refine", base, output)
        expanded_results.update(refined)
        validation_ids = []
        for family in FAMILIES:
            keys = family_best(candidates, expanded_results, family, VALIDATION_CANDIDATES_PER_FAMILY)
            validation_ids.extend(dict.fromkeys([baseline_ids[family], *keys]))
        validation_paths = [str((suite.parent / r["path"]).resolve()) for r in read_suite(suite)["splits"]["validation"]]
        validation = evaluate_batch(pool, candidates, validation_ids, validation_paths, "validation", base, output)
        selected = {family: family_best(candidates, validation, family, 1)[0] for family in FAMILIES}
        assert environment_record()["source_sha256"] == environment["source_sha256"]
        assert digest(Path(__file__)) == runner_hash and digest(PROTOCOL) == protocol_hash
        assert all(digest(PROJECT_ROOT / p) == sha for p, sha in old_records.items())
        for family, key in selected.items():
            candidates[key].save(output / f"selected-{family}.json")
        overall = max(selected.values(), key=lambda k: ranking(validation[k]))
        candidates[overall].save(output / "best.json")
        freeze = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "frozen_before_test": True,
            "selected": selected, "baseline_ids": baseline_ids, "overall": overall,
            "parameters": {k: asdict(candidates[k]) for k in set(selected.values()) | set(baseline_ids.values())},
            "source_sha256": environment["source_sha256"], "runner_sha256": runner_hash,
            "suite_sha256": digest(suite), "protocol_sha256": protocol_hash,
            "test_rule": "No tuning or reselection after final outcomes; old cohort is legacy only"}
        write_json(output / "selection-freeze.json", freeze)
        final_ids = list(dict.fromkeys([*baseline_ids.values(), *selected.values()]))
        # The new test geometry is read for evaluation only after the freeze.
        test_tracks = load_split(suite, "test")
        test_paths = [str((suite.parent / r["path"]).resolve()) for r in read_suite(suite)["splits"]["test"]]
        evaluate_batch(pool, candidates, final_ids, test_paths, "test", base, output)
        legacy_suite = PROJECT_ROOT / "benchmarks/suite/suite.json"
        legacy_paths = [str((legacy_suite.parent / r["path"]).resolve()) for r in read_suite(legacy_suite)["splits"]["test"]]
        evaluate_batch(pool, candidates, final_ids, legacy_paths, "legacy-test", base, output)
        for key in final_ids:
            task = {**base, "candidate": key, "parameters": asdict(candidates[key]), "tracks": expanded[:9],
                    "phase": "representative", "trace_slots": list(range(9))}
            rows = pool.apply(evaluate_candidate, (task,))
            append_csv(output / "representative-episodes.csv", rows, list(rows[0]))
            task.update(tracks=test_paths, phase="test-traces", trace_slots=[0, 1, 2, 31, 62])
            # Replay only designated trace tracks; do not duplicate all test episodes.
            for slot in (0, 1, 2, 31, 62):
                trace_task = {**task, "tracks": [test_paths[slot]], "slot_offset": slot}
                rows = pool.apply(evaluate_candidate, (trace_task,))
                append_csv(output / "test-trace-replays.csv", rows, list(rows[0]))
    assert environment_record()["source_sha256"] == environment["source_sha256"]
    assert digest(Path(__file__)) == runner_hash and digest(PROTOCOL) == protocol_hash
    assert digest(suite) == freeze["suite_sha256"]
    assert all(digest(PROJECT_ROOT / p) == sha for p, sha in old_records.items())
    write_json(output / "status.json", {"complete": True, "wall_seconds": time.perf_counter()-started,
        "candidates": len(candidates), "baseline_ids": baseline_ids, "selected": selected,
        "test_tracks": len(test_tracks), "neat_training_started": False, "old_records_unchanged": len(old_records)})
    print(f"Completed gains-only study: {output}", flush=True)
    return output / "best.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / OUTPUT_DIRECTORY)
    args = parser.parse_args()
    print(run(args.output.resolve()))


if __name__ == "__main__":
    mp.freeze_support()
    main()
