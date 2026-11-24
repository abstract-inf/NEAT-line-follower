"""Benchmark fixed saved networks and inspect tracks; never evolve a population.

Results are short engineering checks, not evidence of trained improvements.
Example: python benchmark_training.py --worlds 32 --seconds 2 --backend gpu
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import csv
import gzip
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.curriculum import CurriculumState, stage_geometry
from line_follower.episodes import run_episode
from line_follower.fitness import RouteProgress
from line_follower.simulation import Simulation
from line_follower.experiments import Evaluator, append_csv, environment_record, write_json
from line_follower.suites import load_split, read_suite, training_batch
from line_follower.tracks import GeneratorConfig, TrackMask, generate_track, validate_track

ORACLE_FORWARD_SPEED = .14  # m/s; engineering probe, never a learned/PID baseline.
ORACLE_LOOKAHEAD = .018  # m along the authenticated route.
ORACLE_HEADING_GAIN = 10.0


def physical_probe(track, robot, episode):
    """Demonstrate dynamics feasibility using evaluator-only ground truth.

    Success does not prove that 15 binary instantaneous readings suffice for a
    controller to learn this route. Never include this oracle in NEAT/PID results.
    """
    simulation = Simulation(robot)
    class GroundTruthOracle:
        def reset(self):
            self.route = RouteProgress(track, episode)
        def act(self, observation, dt):
            state = simulation.state
            self.route.update(state.position)
            target = track.route.point_at(min(track.route.length, self.route.distance+ORACLE_LOOKAHEAD))
            delta = target-state.position[:2]
            error = math.remainder(math.atan2(delta[1], delta[0])-state.yaw, 2*math.pi)
            forward = ORACLE_FORWARD_SPEED*max(0, math.cos(error))**2
            yaw_rate = float(np.clip(ORACLE_HEADING_GAIN*error, -3.5, 3.5))
            return np.clip([(forward-.5*robot.wheel_spacing*yaw_rate)/robot.max_speed,
                            (forward+.5*robot.wheel_spacing*yaw_rate)/robot.max_speed], -1, 1)
    return run_episode(simulation, track, GroundTruthOracle(), episode)


def inspect_tracks(parent: Path, output: Path) -> Path:
    """Save 100 seeded tracks, six representative plots and six physics probes."""
    if output.exists():
        raise FileExistsError("Use a new track-inspection output directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record = json.loads((parent / "run.json").read_text())
    suite = Path(record["suite_path"])
    manifest = read_suite(suite)
    forbidden = {e["geometry_hash"] for split in ("validation", "test") for e in manifest["splits"][split]}
    forbidden.update(manifest.get("protected_geometry_hashes", ()))
    fixed = load_split(suite, "training")[0]
    base = GeneratorConfig(generator_version="2.0", max_attempts=1600)
    output.mkdir(parents=True)
    seen, rows, selected, failures = set(), [], {}, []
    for stage in range(5):
        for batch in range(2):
            generation, rejected = 1000+stage*10+batch, []
            tracks = training_batch(0, "adaptive", generation, 2000, fixed, forbidden, seen, base, 10,
                                    "progressive", curriculum_state=CurriculumState(stage=stage), generation_failures=rejected)
            failures.extend(dict(stage=stage, **r) for r in rejected)
            for slot, track in enumerate(tracks):
                geometry = stage_geometry(base, stage, track.difficulty)
                if validate_track(track, geometry):
                    raise RuntimeError("Generated track failed feasibility validation")
                track.save(output / "tracks" / f"stage-{stage}-batch-{batch}-slot-{slot}.json")
                rows.append({"stage": stage, "generation": generation, "slot": slot, "difficulty": track.difficulty,
                             "length_m": track.route.length, "target_motifs": geometry.target_motifs,
                             "min_length_m": geometry.route_length_min, "max_length_m": geometry.route_length_max,
                             "seed": track.seed, "geometry_hash": track.geometry_hash, "features": ";".join(track.features)})
                selected.setdefault((stage, track.difficulty), track)
                if stage == 4 and track.difficulty == "advanced" and track.route.length > selected[(stage, track.difficulty)].route.length:
                    selected[(stage, track.difficulty)] = track  # Probe the longest sampled final-stage route.
        print(f"Inspected curriculum stage {stage}: {len(rows)} tracks so far", flush=True)
    append_csv(output / "tracks.csv", rows, list(rows[0]))
    choices = [(0, "beginner"), (1, "intermediate"), (2, "advanced"),
               (3, "intermediate"), (3, "advanced"), (4, "advanced")]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    probes = []
    for axis, key in zip(axes.flat, choices):
        track = selected[key]
        axis.imshow(TrackMask(track).field[::4, ::4], extent=(-2, 2, -2, 2), cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest")
        axis.scatter(*track.route.points[0], c="green", s=28, label="start")
        axis.scatter(*track.route.points[-1], c="red", s=28, label="finish")
        axis.set(title=f"Stage {key[0]} | {key[1]} | {track.route.length:.2f} m", xlabel="x (m)", ylabel="y (m)",
                 aspect="equal", xticks=[-2, 0, 2], yticks=[-2, 0, 2])
        axis.legend(fontsize=6, loc="upper right")
        result = physical_probe(track, RobotConfig(max_speed=.35), EpisodeConfig(maximum_time_limit=120, time_reference_speed=.10))
        probes.append(asdict(result))
    figure.suptitle("Progressive training generator v2 | reflectance field | engineering inspection, no NEAT training", fontsize=11)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"representative-tracks.{extension}", dpi=220)
    plt.close(figure)
    write_json(output / "inspection.json", {"training_started": False, "tracks": len(rows), "unique_geometry": len(seen),
               "held_out_overlap": bool(seen & forbidden), "generation_failures": failures, "environment": environment_record(),
               "physical_probe": "Evaluator-only ground-truth oracle; not a sensor-policy comparison", "probes": probes})
    return output / "representative-tracks.png"


def benchmark(parent: Path, output: Path, worlds: int, seconds: float, backend: str) -> Path:
    """Compare 4/10 episode slots with fixed networks and identical time caps."""
    if output.exists():
        raise FileExistsError("Use a new benchmark output directory")
    if worlds < 1 or seconds <= 0:
        raise ValueError("worlds and seconds must be positive")
    with gzip.open(parent / "checkpoints/latest.pkl.gz", "rb") as stream:
        saved = pickle.load(stream)
    record = json.loads((parent / "run.json").read_text())
    suite = Path(record["suite_path"])
    manifest = read_suite(suite)
    forbidden = {e["geometry_hash"] for split in ("validation", "test") for e in manifest["splits"][split]}
    forbidden.update(manifest.get("protected_geometry_hashes", ()))
    fixed = load_split(suite, "training")[0]
    config = GeneratorConfig(generator_version="2.0", max_attempts=1600)
    tracks = training_batch(saved["settings"].seed, "adaptive", 100, 200, fixed, forbidden,
                            set(saved["seen_hashes"]), config, 10, "progressive", curriculum_state=CurriculumState(stage=1))
    output.mkdir(parents=True)
    for i, track in enumerate(tracks):
        track.save(output / "tracks" / f"{i:02d}.json")
    genomes = list(saved["population"].population.values())[:worlds]
    genomes = [deepcopy(genomes[i % len(genomes)]) for i in range(worlds)]
    for i, genome in enumerate(genomes):
        genome.key = i
    robot = replace(saved["robot"], max_speed=.35)
    episode = EpisodeConfig(maximum_time_limit=120, time_reference_speed=.10)
    timings = []
    with Evaluator(robot, episode, saved["population"].config, backend=backend,
                   gpu_batch_size=worlds, time_limit=seconds) as evaluator:
        evaluator.evaluate(genomes, tracks[:1])  # Warm-up; excluded from timing.
        for count in (4, 10):
            # Match the stage-1 50/50 difficulty mixture in both allocations.
            subset = ([t for t in tracks if t.difficulty == "beginner"][:2]
                      + [t for t in tracks if t.difficulty == "intermediate"][:2]) if count == 4 else tracks
            started = time.perf_counter()
            results = evaluator.evaluate(genomes, subset)
            wall = time.perf_counter()-started
            rows = [asdict(r) for episodes in results.values() for r in episodes]
            append_csv(output / f"episodes-{count}.csv", rows, list(rows[0]))
            timings.append({"slots": count, "episodes": len(rows), "wall_seconds": wall,
                            "track_difficulties": [t.difficulty for t in subset],
                            "simulated_world_seconds": sum(r["elapsed_s"] for r in rows),
                            "world_seconds_per_wall_second": sum(r["elapsed_s"] for r in rows)/wall})
            if evaluator.gpu:
                append_csv(output / f"gpu-profile-{count}.csv", evaluator.gpu.batch_profiles,
                           list(evaluator.gpu.batch_profiles[0]))
    histories = list(csv.DictReader((parent / "generations.csv").open()))
    wall_times = np.array([float(r["wall_seconds"]) for r in histories])
    index = next(i for i, r in enumerate(histories) if int(r["generation"]) == 27)
    deltas = [int(r["gpu_world_steps_total"])-(int(histories[i-1]["gpu_world_steps_total"]) if i else 0)
              for i, r in enumerate(histories)]
    estimate = float(np.median(wall_times))*10/4
    report = {"training_started": False, "fixed_networks_only": True, "worlds": worlds,
              "robot": asdict(robot), "episode": asdict(episode),
              "episode_cap_s": seconds, "timings": timings, "environment": environment_record(),
              "historical_generation_27": {"wall_seconds": wall_times[index],
                  "logical_physics_steps": int(histories[index]["physics_steps"]), "executed_gpu_world_steps": deltas[index],
                  "median_wall_seconds": float(np.median(wall_times)),
                  "wall_outlier_ratio": float(wall_times[index]/np.median(wall_times)),
                  "gpu_workload_ratio_to_median": float(deltas[index]/np.median(deltas)),
                  "cause": "Unresolved: aggregate logs do not distinguish sleep, OS scheduling, thermal/resource contention or GPU stalls"},
              "historical_10_slot_budget_estimate": {"seconds_per_generation": estimate,
                  "hours_per_100_generations": estimate*100/3600,
                  "scope": "Linear episode-count estimate at historical durations; longer routes/changed failures/hardware can change cost"},
              "caveat": "Short capped episodes are a throughput probe, not a full training speed estimate or generalization result"}
    write_json(output / "benchmark.json", report)
    return output / "benchmark.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("runs/28-10-2026"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--inspect-only", action="store_true", help="Generate/plot/probe geometry; no network evaluation or evolution")
    args = parser.parse_args()
    print(inspect_tracks(args.parent, args.output) if args.inspect_only else
          benchmark(args.parent, args.output, args.worlds, args.seconds, args.backend))
