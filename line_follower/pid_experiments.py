"""Bounded PID search, frozen selection and paired held-out assessment.

All writers live in the parent process. Each Windows-spawn worker owns its
MuJoCo data and bounded reflectance cache. Controllers receive sensors/dt only.
The held-out test split is loaded only after selected configurations are saved.
"""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
import zipfile

import numpy as np

from .config import EpisodeConfig, RobotConfig
from .controllers import load_model
from .episodes import Episode, TELEMETRY_FIELDS
from .experiments import append_csv, environment_record, snapshot_sources, write_json
from .pid import ResearchPIDConfig, ResearchPIDController
from .simulation import Simulation
from .suites import load_split, read_suite, derived_seed
from .tracks import Section, TrackMask, TrackSpec, generate_track, validate_track

PID_FIELDS = ["pid_error", "pid_integral", "pid_derivative", "pid_turn", "pid_mode"]


@dataclass(frozen=True)
class PIDStudyConfig:
    seed: int = 20261029
    workers: int = 4
    initial_candidates: int = 96
    refinements: int = 24
    promoted_per_family: int = 6
    validation_per_family: int = 3

    def __post_init__(self):
        if self.workers < 1 or self.initial_candidates < 4 or self.refinements < 0:
            raise ValueError("Invalid bounded search budget")
        if self.promoted_per_family < 2 or self.validation_per_family < 1:
            raise ValueError("Need at least two promoted parents and one validation candidate per family")


@lru_cache(maxsize=2)
def _simulation(robot_json: str) -> Simulation:
    return Simulation(RobotConfig(**json.loads(robot_json)))


@lru_cache(maxsize=16)
def _mask(path: str) -> TrackMask:
    return TrackMask(TrackSpec.load(path))


def geometric_tracking_error(positions: np.ndarray, route_points: np.ndarray) -> tuple[float, float]:
    """Return RMSE and MAE to the intended route polyline, independent of the gate cursor."""
    starts = route_points[:-1]
    delta = np.diff(route_points, axis=0)
    length_squared = np.sum(delta * delta, axis=1)
    squared_error = absolute_error = 0.0
    for start in range(0, len(positions), 64):
        offset = positions[start:start+64, None, :] - starts
        projection = np.clip(np.sum(offset * delta, axis=2) / length_squared, 0, 1)
        residual = offset - projection[:, :, None] * delta
        squared_distances = np.min(np.sum(residual * residual, axis=2), axis=1)
        squared_error += float(np.sum(squared_distances))
        absolute_error += float(np.sum(np.sqrt(squared_distances)))
    return math.sqrt(squared_error / len(positions)), absolute_error / len(positions)


def evaluate_candidate(task: dict) -> list[dict]:
    """Evaluate one fixed policy; this function cannot select or modify a policy."""
    simulation = _simulation(task["robot_json"])
    config = EpisodeConfig(**task["episode"])
    if task.get("neat_model"):
        controller, metadata = load_model(task["neat_model"])
        family, policy = f"neat_frozen_g{metadata['generation']}", f"neat-g{metadata['generation']}"
    else:
        parameters = ResearchPIDConfig(**task["parameters"])
        controller = ResearchPIDController(parameters, simulation.config)
        family, policy = parameters.family, task["candidate"]
    records = []
    for slot, path in enumerate(task["tracks"], start=task.get("slot_offset", 0)):
        started = time.perf_counter()
        mask = _mask(path)
        episode = Episode(simulation, mask.track, controller, config, mask, detailed=task.get("detailed_metrics", False))
        absolute_error = 0.0
        positions = [] if task.get("geometric_tracking") else None
        trace = None
        trace_path = ""
        if slot in task.get("trace_slots", []):
            trace_path = str(Path(task["output"]) / "telemetry" / task["phase"] / f"{policy}-{slot:03d}.csv")
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            trace = Path(trace_path).open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(trace, fieldnames=TELEMETRY_FIELDS+PID_FIELDS)
            writer.writeheader()
        try:
            while not episode.done:
                row = episode.advance(telemetry=trace is not None)
                absolute_error += abs(episode.route.lateral_error)
                if positions is not None and not episode.error:
                    positions.append(simulation.data.xpos[simulation.chassis_id, :2].copy())
                if trace is not None and row:
                    writer.writerow({**row, **getattr(controller, "debug", {})})
        finally:
            if trace:
                trace.close()
        result = asdict(episode.result())
        if not task.get("detailed_metrics"):
            result.pop("geometry_lateral_rmse_m")
        if positions is not None:
            rmse, mae = geometric_tracking_error(np.asarray(positions), mask.track.route.points) if not result["error"] else ("", "")
            result.update(geometry_lateral_rmse_m=rmse, geometry_lateral_mae_m=mae)
        records.append({"condition": family, "policy_id": policy, "candidate": policy,
                        "phase": task["phase"], "tuning_seed": task["seed"],
                        "evolution_seed": "0" if task.get("neat_model") else "", "backend": "cpu",
                        "evaluation_speed_limit_m_s": simulation.config.max_speed,
                        "mean_absolute_lateral_error_m": absolute_error/max(1, episode.steps),
                        "episode_wall_s": time.perf_counter()-started, "telemetry": trace_path, **result})
    return records


def ranking(rows: list[dict]) -> tuple[float, ...]:
    complete = [r for r in rows if r["completed"]]
    return (float(np.mean([r["completed"] for r in rows])), float(np.mean([r["progress"] for r in rows])),
            -float(np.mean([r["lateral_rmse_m"] for r in rows])),
            -float(np.mean([r["completion_time_s"] for r in complete])) if complete else -1e9)


def _training_tracks(output: Path, suite: Path, seed: int) -> tuple[list[str], list[str]]:
    """Freeze short engineering cases and separate procedural training courses."""
    straight = lambda length, feature="straight": Section("straight", length=length, feature=feature)
    corner = lambda angle, feature="corner": Section("corner", angle=angle, feature=feature)
    q = math.pi/2
    recipes = [("straight", (straight(1.4),))]
    for sign in (1, -1):
        recipes.append((f"right-angle-{sign}", (straight(.6), corner(sign*q), straight(.6))))
    for sign in (1, -1):
        recipes.append((f"diagnostic-zigzag-{sign}", (straight(.6), corner(sign*q, "zigzag"), straight(.25, "zigzag"),
                         corner(-sign*q, "zigzag"), straight(.25, "zigzag"), corner(sign*q, "zigzag"), straight(.55, "zigzag"))))
    recipes.append(("curve-left", (straight(.5), Section("arc", radius=.25, angle=q, feature="gentle_curve"), straight(.5))))
    for sign in (1, -1):
        recipes.append((f"entry45-zigzag-{sign}", (straight(.5), corner(sign*q/2, "zigzag"), straight(.25, "zigzag"),
                        corner(-sign*q, "zigzag"), straight(.25, "zigzag"), corner(sign*q, "zigzag"), straight(.25, "zigzag"),
                        corner(-sign*q/2, "zigzag"), straight(.55))))
    recipes.append(("curve-right", (straight(.5), Section("arc", radius=.25, angle=-q, feature="gentle_curve"), straight(.5))))
    manifest = read_suite(suite)
    forbidden = {e["geometry_hash"] for split in ("validation", "test") for e in manifest["splits"][split]}
    tracks = [TrackSpec(name, sections, (-.9, -.5, 0), difficulty="manual", split="training") for name, sections in recipes]
    for difficulty in ("beginner", "intermediate", "advanced"):
        for slot in range(2):
            for attempt in range(100):
                track = replace(generate_track(derived_seed(seed, "pid_training", slot, attempt,
                                ("beginner", "intermediate", "advanced").index(difficulty)), difficulty), split="training")
                if track.geometry_hash not in forbidden and track.geometry_hash not in {t.geometry_hash for t in tracks}:
                    tracks.append(track)
                    break
            else:
                raise RuntimeError("Could not find disjoint PID training geometry")
    paths = []
    for slot, track in enumerate(tracks):
        if track.geometry_hash in forbidden or validate_track(track):
            raise ValueError(f"Training track is invalid or overlaps a held-out: {track.name}")
        path = output / "tracks/training" / f"{slot:03d}.json"
        track.save(path)
        paths.append(str(path.resolve()))
    write_json(output / "training-manifest.json", {"coarse_slots": list(range(6)),
               "tracks": [{"path": p, "geometry_hash": t.geometry_hash, "features": t.features,
                           "difficulty": t.difficulty, "seed": t.seed} for p, t in zip(paths, tracks)]})
    return paths[:6], paths


def sample_parameters(rng: np.random.Generator, supervised: bool) -> ResearchPIDConfig:
    return ResearchPIDConfig(kp=float(rng.uniform(.25, 3.5)), ki=float(rng.choice([0, .01, .05, .15])),
        kd=float(rng.uniform(0, .12)), base=float(rng.uniform(.25, .75)),
        minimum_command=float(rng.uniform(.02, .25)), derivative_filter_s=float(rng.uniform(.015, .10)),
        slowdown_gain=float(rng.uniform(0, 5)), loss_hold_s=float(rng.uniform(.10, .8)),
        search_command=float(rng.uniform(.2, .65)), corner_supervision=supervised,
        approach_command=float(rng.uniform(.2, .5)), turn_lead_m=float(rng.uniform(0, .025)),
        pivot_command=float(rng.uniform(.25, .6)))


def _refine(rng: np.random.Generator, parent: ResearchPIDConfig) -> ResearchPIDConfig:
    bounds = {"kp": (.25, 3.5), "ki": (0, .25), "kd": (0, .12), "base": (.25, .75),
              "minimum_command": (.02, .25), "derivative_filter_s": (.015, .10), "slowdown_gain": (0, 5),
              "loss_hold_s": (.1, .8), "search_command": (.2, .65), "approach_command": (.2, .5),
              "turn_lead_m": (0, .025), "pivot_command": (.25, .6)}
    parameters = asdict(parent)
    for name, (lower, upper) in bounds.items():
        parameters[name] = float(np.clip(parameters[name]+rng.normal(0, .12*(upper-lower)), lower, upper))
    return ResearchPIDConfig(**parameters)


def run_study(output: Path, suite: Path, robot: RobotConfig, episode: EpisodeConfig,
              neat_model: Path, settings: PIDStudyConfig = PIDStudyConfig()) -> Path:
    """Execute the declared finite search, validation freeze, then final test once."""
    if output.exists():
        raise FileExistsError("Choose a new PID output directory; existing experiments are immutable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    environment = environment_record()
    original_model_hash = hashlib.sha256(neat_model.read_bytes()).hexdigest()
    original_suite_hash = hashlib.sha256(suite.read_bytes()).hexdigest()
    write_json(output / "run.json", {"study": "pid-research-1", "settings": asdict(settings), "robot": asdict(robot),
               "episode": asdict(episode), "suite": str(suite.resolve()), "environment": environment,
               "selection": "lexicographic completion, progress, negative RMSE, negative successful time",
               "neat_model_sha256_before_tuning": original_model_hash, "suite_sha256": original_suite_hash})
    snapshot_sources(output / "source.zip")
    project = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output / "source.zip", "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in ("run_pid_optimization.py", "research/PID_Protocol.md", "tests/test_pid.py"):
            archive.write(project / relative, relative)
    coarse, expanded = _training_tracks(output, suite, settings.seed)
    rng = np.random.default_rng(settings.seed)
    candidates = {}
    for index in range(settings.initial_candidates):
        supervised = bool(index % 2)
        parameters = replace(ResearchPIDConfig(), corner_supervision=supervised) if index < 2 else sample_parameters(rng, supervised)
        candidates[f"pid-{index:03d}"] = parameters
    candidate_rows = [{"candidate": k, **asdict(v)} for k, v in candidates.items()]
    append_csv(output / "candidates.csv", candidate_rows, list(candidate_rows[0]))
    base = {"robot_json": json.dumps(asdict(robot), sort_keys=True), "episode": asdict(episode),
            "output": str(output.resolve()), "seed": settings.seed}
    results = {}

    def batch(pool, identifiers: list[str], tracks: list[str], phase: str) -> None:
        tasks = [{**base, "candidate": key, "parameters": asdict(candidates[key]), "tracks": tracks, "phase": phase}
                 for key in identifiers]
        for index, records in enumerate(pool.imap(evaluate_candidate, tasks), 1):
            append_csv(output / f"{phase}-episodes.csv", records, list(records[0]))
            if any(r["error"] for r in records):
                raise RuntimeError("Technical evaluation error recorded; stopping before model selection")
            key = records[0]["candidate"]
            results[(phase, key)] = records
            summary = {"phase": phase, "candidate": key, "family": candidates[key].family,
                       "episodes": len(records), "completed": sum(r["completed"] for r in records),
                       "completion_rate": ranking(records)[0], "mean_progress": ranking(records)[1],
                       "mean_rmse_m": -ranking(records)[2], "episode_wall_sum_s": sum(r["episode_wall_s"] for r in records)}
            append_csv(output / "candidate-summary.csv", [summary], list(summary))
            print(f"PID {phase}: {index}/{len(tasks)} | {key} | completed {summary['completed']}/{len(records)} | "
                  f"progress {summary['mean_progress']:.3f}", flush=True)

    def best_in_family(phase: str, supervised: bool, count: int) -> list[str]:
        keys = [key for p, key in results if p == phase and candidates[key].corner_supervision == supervised]
        return sorted(keys, key=lambda k: ranking(results[(phase, k)]), reverse=True)[:count]

    with mp.get_context("spawn").Pool(settings.workers) as pool:
        batch(pool, list(candidates), coarse, "coarse")
        promoted = sum((best_in_family("coarse", family, settings.promoted_per_family) for family in (False, True)), [])
        batch(pool, promoted, expanded, "expanded")
        parents = sum((best_in_family("expanded", family, 2) for family in (False, True)), [])
        refinements = []
        for index in range(settings.refinements):
            key = f"pid-{settings.initial_candidates+index:03d}"
            candidates[key] = _refine(rng, candidates[parents[index % len(parents)]])
            refinements.append(key)
        if refinements:
            rows = [{"candidate": k, **asdict(candidates[k])} for k in refinements]
            append_csv(output / "candidates.csv", rows, list(rows[0]))
            batch(pool, refinements, expanded, "refine")
            # Rank every fully evaluated configuration against the same expanded suite.
            for key in refinements:
                results[("expanded", key)] = results[("refine", key)]
        validation_ids = sum((best_in_family("expanded", family, settings.validation_per_family) for family in (False, True)), [])
        validation_tracks = load_split(suite, "validation")
        validation_paths = []
        for slot, track in enumerate(validation_tracks):
            path = output / "tracks/validation" / f"{slot:03d}.json"
            track.save(path)
            validation_paths.append(str(path.resolve()))
        batch(pool, validation_ids, validation_paths, "validation")
        if (environment_record()["source_sha256"] != environment["source_sha256"]
                or hashlib.sha256(neat_model.read_bytes()).hexdigest() != original_model_hash
                or hashlib.sha256(suite.read_bytes()).hexdigest() != original_suite_hash):
            raise RuntimeError("Code, model or cohort changed during tuning; refusing final test")
        selected = [best_in_family("validation", family, 1)[0] for family in (False, True)]
        overall = max(selected, key=lambda key: ranking(results[("validation", key)]))
        for key in selected:
            candidates[key].save(output / f"selected-{candidates[key].family}.json")
        candidates[overall].save(output / "best.json")
        freeze = {"frozen_before_test": True, "selected": selected, "overall": overall,
                  "frozen_utc": datetime.now(timezone.utc).isoformat(),
                  "parameters": {key: asdict(candidates[key]) for key in selected},
                  "configuration_hashes": {key: hashlib.sha256(json.dumps(asdict(candidates[key]), sort_keys=True).encode()).hexdigest() for key in selected},
                  "neat_model_sha256": hashlib.sha256(neat_model.read_bytes()).hexdigest(),
                  "source_sha256": environment["source_sha256"], "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
                  "test_selection_rule": "No selection, tuning or code changes after observing test outcomes"}
        write_json(output / "selection-freeze.json", freeze)
        # This is the first test-geometry load in the tuning pipeline.
        test_tracks = load_split(suite, "test")
        test_paths = []
        for slot, track in enumerate(test_tracks):
            path = output / "tracks/test" / f"{slot:03d}.json"
            track.save(path)
            test_paths.append(str(path.resolve()))
        test_tasks = []
        for key in selected+["neat-g109"]:
            policy = {"neat_model": str(neat_model.resolve())} if key == "neat-g109" else {"parameters": asdict(candidates[key])}
            test_tasks.extend({**base, **policy, "candidate": key, "tracks": [path], "phase": "test",
                               "slot_offset": slot, "trace_slots": [0, 1, 2, 31, 62]}
                              for slot, path in enumerate(test_paths))
        test_results = []
        for index, records in enumerate(pool.imap(evaluate_candidate, test_tasks), 1):
            append_csv(output / "test-episodes.csv", records, list(records[0]))
            test_results.extend(records)
            if any(r["error"] for r in records):
                raise RuntimeError("Technical error saved during final assessment; do not tune against these outcomes")
            if index % 30 == 0:
                print(f"Frozen paired test: {index}/{len(test_tasks)} episodes", flush=True)
        for family in sorted({r["condition"] for r in test_results}):
            rows = [r for r in test_results if r["condition"] == family]
            print(f"Frozen {family}: {sum(r['completed'] for r in rows)}/90 test tracks", flush=True)
        # Representative diagnostic trajectories are training-only, never extra tuning.
        for key in selected:
            records = pool.apply(evaluate_candidate, ({**base, "candidate": key, "parameters": asdict(candidates[key]),
                                 "tracks": coarse, "phase": "representative", "trace_slots": list(range(len(coarse)))},))
            append_csv(output / "representative-episodes.csv", records, list(records[0]))
    write_json(output / "status.json", {"complete": True, "wall_seconds": time.perf_counter()-started,
               "candidates": len(candidates), "selected": selected, "overall": overall,
               "test_episodes": len(test_paths)*3, "neat_training_started": False})
    return output / "best.json"
