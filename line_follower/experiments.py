"""Reproducible NEAT experiments, persistent spawn workers, and CSV evaluation.

    A checkpoint is taken *after* reproduction and generation advancement. This
    avoids replaying generation N with its N+1 population, a subtle problem with
    using the old library's checkpoint reporter without an episode scheduler.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
from importlib.resources import files
from itertools import count
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import random
import time
import zipfile
from uuid import uuid4

import neat
import numpy as np

from .config import EpisodeConfig, RobotConfig, verify_geometry
from .controllers import (NEAT_VERSION, NEATController, PIDConfig, PIDController, load_model,
                          load_neat_config, read_model_artifact, save_model)
from .episodes import EpisodeResult, run_episode
from .fitness import FITNESS_VERSION, aggregate_fitness
from .curriculum import CurriculumConfig, CurriculumState, stage_geometry, training_competence
from .diagnostics import episode_summaries
from .simulation import Simulation
from .suites import corner_transfer_geometry, derived_seed, difficulty_batch, load_split, read_suite, training_batch
from .tracks import GeneratorConfig, TrackMask, TrackSpec
from .population_view import SnapshotPublisher, atomic_snapshot, robot_snapshot

CHECKPOINT_VERSION = 2
MASK_CACHE_SIZE = 4
PROGRESS_REPORT_SECONDS = 2.0
CHECKPOINT_REPLACE_ATTEMPTS = 6
CHECKPOINT_RETRY_BASE_SECONDS = .02  # At most .62 s for transient Windows reader/indexer locks.
SUMMARY_FIELDS = ["condition", "policy_id", "evolution_seed", "repeat_id", "evaluation_speed_limit_m_s"] + list(EpisodeResult.__dataclass_fields__)


def write_json(path: str | Path, value: dict, *, best_effort: bool = False) -> bool:
    """Replace metadata atomically, tolerating short Windows reader locks.

    Essential records retry for at most 150 ms and then report any failure.
    Disposable progress exports may be skipped; they must never abort evolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(value, indent=2, allow_nan=False)
    for attempt in range(5):
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
            return True
        except OSError as error:
            if best_effort:
                return False
            if not isinstance(error, PermissionError) or attempt == 4:
                raise
            time.sleep(.01 * 2**attempt)
    return False  # All loop paths return or raise; retained for type checkers.


def append_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def environment_record() -> dict:
    """Capture source content and machine details without requiring a Git checkout."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".xml", ".json", ".txt"}:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    packages = ["mujoco", "numpy", "Pillow", "neat-python"]
    for optional in ("mujoco-warp", "warp-lang"):
        try:
            importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            continue
        packages.append(optional)
    return {"started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "dependencies": {name: importlib.metadata.version(name) for name in packages},
            "source_sha256": digest.hexdigest(), "fitness_version": FITNESS_VERSION,
            "physical_parameters_calibrated": False}


def snapshot_sources(destination: str | Path) -> None:
    """Keep executable source, public parameters, docs and lockfile with a run."""
    root = Path(__file__).resolve().parent.parent
    sources = [p for p in (root / "line_follower").rglob("*") if p.suffix in {".py", ".xml", ".json", ".txt"}]
    sources += list((root / "configs").glob("*.json"))
    sources += [p for p in (root / "docs").rglob("*") if p.suffix in {".md", ".py", ".txt", ".json"}]
    sources += [root / name for name in ("run_training.py", "run_corner_training.py", "run_speed_training.py", "continue_training.py",
                                        "benchmark_training.py", "plot_training.py", "prepare_resume.py",
                                        "pyproject.toml", "requirements-lock.txt", "README.md", "THIRD_PARTY_NOTICES.md")]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(set(sources)):
            if source.is_file():
                archive.write(source, source.relative_to(root))


@dataclass(frozen=True)
class TrainConfig:
    """Five-seed studies run this independently for each strategy/seed pair."""

    strategy: str = "curriculum"
    seed: int = 0
    generations: int = 500
    population: int = 200
    workers: int = 1
    time_limit: float | None = None
    excluded_features: tuple[str, ...] = ()
    backend: str = "cpu"
    gpu_batch_size: int = 128
    episodes_per_genome: int = 4
    track_profile: str = "standard"
    aggregation: str = "mean_min"
    cvar_fraction: float = .3
    curriculum: CurriculumConfig = CurriculumConfig()
    speed_schedule: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if self.strategy not in {"single", "multi", "curriculum", "adaptive"}:
            raise ValueError("Unknown NEAT training condition")
        if self.generations < 1 or self.population < 2 or self.workers < 1:
            raise ValueError("Training needs generations >= 1, population >= 2, workers >= 1")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.backend not in {"cpu", "gpu"} or self.gpu_batch_size < 1:
            raise ValueError("Backend must be cpu or gpu, with a positive GPU batch size")
        if not isinstance(self.episodes_per_genome, int) or self.episodes_per_genome < 1:
            raise ValueError("episodes_per_genome must be a positive integer")
        if self.track_profile not in {"standard", "zigzag", "progressive", "corner", "fast"}:
            raise ValueError("Unknown training track profile")
        if self.track_profile == "zigzag" and (self.strategy != "multi" or self.episodes_per_genome % 4):
            raise ValueError("Zigzag profile needs multi strategy and a multiple of four episode slots")
        if self.track_profile == "progressive" and self.strategy != "adaptive":
            raise ValueError("Progressive profile requires adaptive strategy")
        if self.track_profile == "corner" and (self.strategy != "adaptive" or self.episodes_per_genome != 4):
            raise ValueError("Corner profile needs adaptive strategy and four shared tracks per generation")
        if self.track_profile == "fast" and (self.strategy != "multi" or self.episodes_per_genome != 6 or self.backend != "cpu"):
            raise ValueError("Fast profile needs multi strategy, six tracks, and CPU physics")
        if self.speed_schedule:
            starts, speeds = zip(*self.speed_schedule)
            if (starts[0] != 0 or starts[-1] >= self.generations or tuple(sorted(set(starts))) != starts
                    or any(type(start) is not int for start in starts)):
                raise ValueError("Speed schedule needs unique generation starts beginning at zero")
            if any(not math.isfinite(speed) or speed <= 0 for speed in speeds) or tuple(sorted(set(speeds))) != speeds:
                raise ValueError("Scheduled speed limits must be positive and strictly increasing")
            if self.backend != "cpu":
                raise ValueError("Speed schedules currently require the CPU backend")
        aggregate_fitness([0.0], self.aggregation, self.cvar_fraction)

    def speed_at(self, generation: int, default: float) -> float:
        return next((speed for start, speed in reversed(self.speed_schedule) if start <= generation), default)


def warm_start_population(model: str | Path, size: int) -> neat.Population:
    """Seed a new population with one exact champion and independent mutated copies."""
    artifact = read_model_artifact(model)
    if artifact["metadata"].get("swap_legacy_motors"):
        raise ValueError("Warm start requires the training motor mapping")
    config = artifact["config"]
    config.pop_size = size
    champion = artifact["genome"]
    genomes = {}
    for key in range(1, size + 1):
        genome = deepcopy(champion)
        genome.key, genome.fitness = key, None
        if key > 1:
            genome.mutate(config.genome_config)
        genomes[key] = genome
    species = config.species_set_type(config.species_set_config, neat.reporting.ReporterSet())
    population = neat.Population(config, (genomes, species, 0))
    population.reproduction.genome_indexer = count(size + 1)
    species.reporters = population.reporters
    species.speciate(config, genomes, 0)
    return population


class WorkerContext:
    """One process owns one simulation and a bounded cache of pose-specific masks."""

    def __init__(self, robot: RobotConfig, episode: EpisodeConfig, neat_config: neat.Config,
                 time_limit: float | None):
        self.simulation = Simulation(robot)
        self.episode = episode
        self.neat_config = neat_config
        self.time_limit = time_limit
        self.masks: OrderedDict[str, TrackMask] = OrderedDict()

    def mask(self, track: TrackSpec) -> TrackMask:
        # Geometry identity intentionally ignores pose; mask identity must not.
        identity = hashlib.sha256(json.dumps(asdict(track), sort_keys=True).encode()).hexdigest()
        if identity not in self.masks:
            self.masks[identity] = TrackMask(track)
            while len(self.masks) > MASK_CACHE_SIZE:
                self.masks.popitem(last=False)
        self.masks.move_to_end(identity)
        return self.masks[identity]

    def evaluate(self, task: tuple[neat.DefaultGenome, list[TrackSpec]]) -> tuple[int, list[EpisodeResult]]:
        genome, tracks, *extra = task
        live = extra[0] if extra else None
        detailed = extra[1] if len(extra) > 1 else False
        controller = NEATController(genome, self.neat_config)
        results = []
        for index, track in enumerate(tracks):
            publisher = SnapshotPublisher(live, f"genome-{genome.key}-track-{index}")
            def observe(episode, force):
                if force or publisher.due:
                    publisher.publish([robot_snapshot(genome.key, index, episode)], force)
            results.append(run_episode(self.simulation, track, controller, self.episode, self.mask(track),
                                       time_limit=self.time_limit, observer=observe if live else None, detailed=detailed))
        if any(result.error for result in results):
            raise RuntimeError(f"Genome {genome.key} evaluation error: {[r.error for r in results if r.error]}")
        return genome.key, results


# The only mutable module global is a process-local worker handle required by
# multiprocessing's spawn entry point. It never stores state shared by runs or
# processes. The parent/serial evaluator uses its own explicit WorkerContext.
_WORKER_CONTEXT: WorkerContext | None = None


def _initialize_worker(robot: RobotConfig, episode: EpisodeConfig, neat_config: neat.Config,
                       time_limit: float | None) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = WorkerContext(robot, episode, neat_config, time_limit)


def _evaluate_worker(task: tuple[neat.DefaultGenome, list[TrackSpec]]) -> tuple[int, list[EpisodeResult]]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker was not initialized")
    return _WORKER_CONTEXT.evaluate(task)


class Evaluator:
    """One long-lived pool per run; its scheduling cannot select the track batch."""

    def __init__(self, robot: RobotConfig, episode: EpisodeConfig, neat_config: neat.Config,
                 workers: int = 1, time_limit: float | None = None, backend: str = "cpu",
                 gpu_batch_size: int = 128):
        self.pool = None
        self.gpu = None
        self.serial = None
        if backend == "gpu":
            from .gpu_backend import GPUEvaluator
            self.gpu = GPUEvaluator(robot, episode, neat_config, time_limit, gpu_batch_size)
        elif backend == "cpu":
            self.serial = WorkerContext(robot, episode, neat_config, time_limit)
        else:
            raise ValueError("Unknown evaluation backend")
        if backend == "cpu" and workers > 1:
            self.pool = mp.get_context("spawn").Pool(workers, initializer=_initialize_worker,
                                                     initargs=(robot, episode, neat_config, time_limit))

    def evaluate(self, genomes: list[neat.DefaultGenome], tracks: list[TrackSpec], progress=None,
                 live: dict | None = None) -> dict[int, list[EpisodeResult]]:
        if self.gpu is not None:
            return self.gpu.evaluate(genomes, tracks, progress, live)
        tasks = [(genome, tracks, live) for genome in genomes]
        iterator = self.pool.imap(_evaluate_worker, tasks) if self.pool else map(self.serial.evaluate, tasks)
        results = {}
        for key, episodes in iterator:
            results[key] = episodes
            if progress is not None:
                progress({"completed_episodes": len(results)*len(tracks), "total_episodes": len(tasks)*len(tracks)})
        return results

    def validate(self, genome: neat.DefaultGenome, tracks: list[TrackSpec], progress=None,
                 detailed: bool = False) -> list[EpisodeResult]:
        if self.gpu is not None:
            return self.gpu.evaluate([genome], tracks, progress)[genome.key]
        tasks = [(genome, [track], None, detailed) for track in tracks]
        iterator = self.pool.imap(_evaluate_worker, tasks) if self.pool else map(self.serial.evaluate, tasks)
        results = []
        for _, episodes in iterator:
            results.extend(episodes)
            if progress is not None:
                progress({"completed_episodes": len(results), "total_episodes": len(tracks)})
        return results

    def close(self) -> None:
        if self.gpu is not None:
            self.gpu.close()
        if self.pool:
            self.pool.close()
            self.pool.join()

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *_args) -> None:
        if self.pool:
            if _args[0] is not None:
                self.pool.terminate()
            self.close()


def _save_checkpoint(path: Path, state: dict) -> None:
    """Commit a complete pickle atomically; keep the old checkpoint on failure."""
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(CHECKPOINT_REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == CHECKPOINT_REPLACE_ATTEMPTS-1:
                raise
            time.sleep(CHECKPOINT_RETRY_BASE_SECONDS*2**attempt)


def _restore_logs(output: Path, next_generation: int) -> None:
    """Discard only log rows newer than the committed checkpoint after a crash."""
    for name in ("generations.csv", "champion-episodes.csv", "training-episodes.csv", "episode-fitness.csv",
                 "diagnostics.csv", "curriculum.csv", "gpu-profile.csv", "evolution.csv",
                 "quality.csv", "fitness-components.csv"):
        path = output / name
        if not path.exists():
            continue
        temporary = path.with_suffix(".restore.tmp")
        with path.open(newline="", encoding="utf-8") as source, temporary.open("w", newline="", encoding="utf-8") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                try:
                    generation = int(row["generation"])
                except (ValueError, TypeError):
                    continue  # A torn final row cannot belong to a committed generation.
                if generation < next_generation:
                    writer.writerow(row)
        temporary.replace(path)


def _progress_reporter(output: Path, generation: int, phase: str):
    """Report parent-process progress without affecting evaluation or random state."""
    last_report = -float("inf")
    started = time.perf_counter()
    export_warning = False

    def report(update: dict) -> None:
        nonlocal last_report, export_warning
        now = time.perf_counter()
        if now-last_report < PROGRESS_REPORT_SECONDS and update["completed_episodes"] < update["total_episodes"]:
            return
        last_report = now
        record = {"generation": generation, "phase": phase, "wall_seconds": now-started, **update}
        exported = write_json(output / "progress.json", record, best_effort=True)
        if not exported and not export_warning:
            print("Progress display file temporarily unavailable; training continues. Check the console for progress.", flush=True)
        export_warning = not exported
        batch = f" batch {update['batch']}/{update['batches']}" if "batch" in update else ""
        simulated = f"; batch simulated time {update['simulated_seconds']:.1f}s" if "simulated_seconds" in update else ""
        if update.get("batch_wall_seconds", 0) > 0:
            simulated += (f"; {update['simulated_seconds']/update['batch_wall_seconds']:.2f}x world speed"
                          f"; {update['aggregate_simulated_seconds']/update['batch_wall_seconds']:.1f} simulated world-s/s")
        print(f"generation {generation} {phase}:{batch} episodes {update['completed_episodes']}/{update['total_episodes']}; "
              f"wall {now-started:.1f}s{simulated}", flush=True)

    return report


def _report_training_fitness(output: Path, generation: int,
                             evaluated: dict[int, list[EpisodeResult]], tracks: list[TrackSpec],
                             aggregation: str = "mean_min", cvar_fraction: float = .3) -> None:
    """Print population fitness and per-track scores without changing selection.

    Population fitness uses the declared mean/minimum aggregation. Episode
    scores are summarized separately; fresh tracks can differ across generations.
    Read the previous committed CSV row so deltas also work after a checkpoint.
    """
    fitness = [aggregate_fitness([episode.score for episode in episodes], aggregation, cvar_fraction) for episodes in evaluated.values()]
    average, best = float(np.mean(fitness)), max(fitness)
    previous = None
    history = output / "generations.csv"
    if history.exists():
        with history.open(newline="", encoding="utf-8") as stream:
            previous = next((row for row in csv.DictReader(stream) if int(row["generation"]) == generation-1), None)
    change = (f" | change: average={average-float(previous['training_mean']):+.6f}, "
              f"best={best-float(previous['training_best']):+.6f}" if previous else " | change: n/a (first generation)")
    print(f"Generation {generation} training fitness: average={average:.6f}, best={best:.6f}{change}", flush=True)
    rows = []
    for index, track in enumerate(tracks):
        scores = {key: episodes[index].score for key, episodes in evaluated.items()}
        winner = max(scores, key=scores.get)
        row = {"generation": generation, "episode_slot": index+1, "track_name": track.name,
               "track_hash": track.geometry_hash, "difficulty": track.difficulty,
               "genomes": len(scores), "average_score": float(np.mean(list(scores.values()))),
               "best_score": scores[winner], "best_genome_id": winner}
        rows.append(row)
        print(f"  Episode {index+1}/{len(tracks)} [{track.difficulty}] {track.name}: "
              f"average={row['average_score']:.6f}, best={row['best_score']:.6f} (genome {winner})", flush=True)
    append_csv(output / "episode-fitness.csv", rows, list(rows[0]))


def train(output: str | Path, suite_path: str | Path, settings: TrainConfig | None = None,
          robot: RobotConfig | None = None, episode: EpisodeConfig | None = None,
          neat_path: str | Path | None = None, resume: str | Path | None = None,
          stop_after: int | None = None, geometry_lock: str | Path = "configs/geometry-freeze.json",
          live_preview: bool = False, initial_model: str | Path | None = None,
          quality_reference: str | Path | None = None) -> Path:
    """Train without ever loading test track files or evaluating their outcomes."""
    output = Path(output)
    suite_path = Path(suite_path)
    settings, robot, episode = settings or TrainConfig(), robot or RobotConfig(), episode or EpisodeConfig()
    condition = f"neat_{settings.strategy}" + (f"_{settings.track_profile}" if settings.track_profile != "standard" else "")
    if initial_model is not None and (resume is not None or neat_path is not None):
        raise ValueError("Warm starting uses the saved model's config and a new run directory")
    if settings.speed_schedule:
        condition += "_speed"
    selected_geometry = verify_geometry(geometry_lock, robot)
    environment = environment_record()
    environment["fitness_version"] = episode.fitness_version
    manifest = read_suite(suite_path)
    suite_hash = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    validation = load_split(suite_path, "validation")
    fixed = load_split(suite_path, "training")[0]
    forbidden = {entry["geometry_hash"] for split in ("validation", "test") for entry in manifest["splits"][split]}
    forbidden.update(manifest.get("protected_geometry_hashes", ()))
    fast = settings.track_profile == "fast"
    reference_hash = None
    if fast:
        from .fast_study import fast_training_batch, quality_gate
        if quality_reference is None:
            raise ValueError("Fast study requires a frozen quality reference")
        reference_hash = hashlib.sha256(Path(quality_reference).read_bytes()).hexdigest()
        reference = json.loads(Path(quality_reference).read_text(encoding="utf-8"))
        validation_groups = {row["geometry_hash"]: row["group"] for row in manifest["splits"]["validation"]}
        training_tracks = load_split(suite_path, "training")
        bank_by_hash = {track.geometry_hash: track for track in training_tracks}
        bank = [bank_by_hash[row["geometry_hash"]] for row in manifest["splits"]["training"] if row["group"] == "speed"]
        if len(bank) != 20:
            raise ValueError("Fast study requires a bank of twenty speed courses")
    generator = GeneratorConfig(**{**manifest.get("training_generator_config", manifest["generator_config"]),
                                  "excluded_features": settings.excluded_features})
    if settings.strategy == "adaptive" and settings.track_profile != "corner" and generator.generator_version != "2.0":
        raise ValueError("Adaptive training requires a separately versioned progressive suite")
    curriculum_state = CurriculumState()
    empty_key = (-1e9, -1, -1e9) if fast else (-1.0, -float("inf"))
    best_key = empty_key
    seen_hashes: set[str] = set()
    total_physics_steps = 0
    if resume:
        with gzip.open(resume, "rb") as stream:
            saved = pickle.load(stream)
        if saved["checkpoint_version"] != CHECKPOINT_VERSION or saved["neat_version"] != NEAT_VERSION:
            raise ValueError("Unsupported checkpoint version")
        if saved["settings"] != settings or saved["robot"] != robot or saved["episode"] != episode:
            raise ValueError("Resume must retain saved training, robot, and episode settings")
        if saved["suite_hash"] != suite_hash:
            raise ValueError("Suite changed since checkpoint creation")
        if saved.get("source_sha256") != environment["source_sha256"]:
            raise ValueError("Source changed since checkpoint creation; preserve the original code to resume reproducibly")
        if saved.get("dependencies") != environment["dependencies"]:
            raise ValueError("Dependencies changed since checkpoint creation")
        if fast and saved.get("quality_reference_sha256") != reference_hash:
            raise ValueError("Quality reference changed since checkpoint creation")
        population = saved["population"]
        random.setstate(saved["random_state"])
        best_key = saved["best_key"]
        seen_hashes = saved["seen_hashes"]
        total_physics_steps = saved["total_physics_steps"]
        curriculum_state = saved.get("curriculum_state", CurriculumState())
        if not (output / "run.json").exists():
            raise ValueError("Resume in the original run directory, with its model and logs")
        _restore_logs(output, population.generation)
        if saved["best_model_bytes"] is not None:
            temporary = output / "best.pkl.restore.tmp"
            temporary.write_bytes(saved["best_model_bytes"])
            temporary.replace(output / "best.pkl")
        else:
            # A new continuation phase has no model scored under its new task.
            (output / "best.pkl").unlink(missing_ok=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        random.seed(settings.seed)
        if initial_model is None:
            neat_config = load_neat_config(neat_path or files("line_follower").joinpath("configs/neat.txt"))
            neat_config.pop_size = settings.population
            population = neat.Population(neat_config)
        else:
            population = warm_start_population(initial_model, settings.population)
        population.config.save(str(output / "neat-config.txt"))
        (output / "frozen-validation").mkdir()
        for index, track in enumerate(validation):
            track.save(output / "frozen-validation" / f"{index:03d}.json")
        fixed.save(output / "single-training-track.json")
        write_json(output / "run.json", {"settings": asdict(settings), "robot": asdict(robot), "episode": asdict(episode),
                   "quality_reference_sha256": reference_hash,
                   "live_population_observer": live_preview,
                   "geometry_freeze": selected_geometry,
                   "suite_sha256": suite_hash, "suite_path": str(suite_path.resolve()),
                   "validation_hashes": [track.geometry_hash for track in validation],
                   "generator": asdict(generator), "environment": environment,
                   "backend_details": {"physics": settings.backend, "gpu_experimental": settings.backend == "gpu",
                       "gpu_batch_layout": "track-major" if settings.backend == "gpu" else None,
                       "gpu_finished_world_compaction": settings.backend == "gpu",
                       "gpu_diagnostic_difference": "AUTORESET disable flag omitted; invalid states and overflow raise explicitly" if settings.backend == "gpu" else None},
                   "claim_scope": "Track generalization only; no hardware calibration or fault robustness claims",
                   "single_track_repeat_semantics": f"{settings.episodes_per_genome} budget slots; deterministic repeats are not independent trials"})
        snapshot_sources(output / "source.zip")
    (output / "checkpoints").mkdir(exist_ok=True)
    start_time = time.perf_counter()
    generations_this_call = 0
    def make_evaluator(phase_robot):
        return Evaluator(phase_robot, episode, population.config, settings.workers, settings.time_limit,
                         settings.backend, settings.gpu_batch_size)

    with ExitStack() as resources:
        phase_robot = replace(robot, max_speed=settings.speed_at(population.generation, robot.max_speed))
        evaluator = resources.enter_context(make_evaluator(phase_robot))
        if resume and evaluator.gpu is not None:
            evaluator.gpu.executed_world_steps = saved.get("gpu_world_steps_total", 0)
            evaluator.gpu.padding_world_steps = saved.get("gpu_padding_world_steps_total", 0)
        while population.generation < settings.generations:
            generation_started = time.perf_counter()
            generation = population.generation
            speed = settings.speed_at(generation, robot.max_speed)
            if speed != phase_robot.max_speed:
                resources.close()
                phase_robot = replace(robot, max_speed=speed)
                evaluator = resources.enter_context(make_evaluator(phase_robot))
            if settings.speed_schedule and (generation == 0 or speed != settings.speed_at(generation - 1, robot.max_speed)):
                best_key = empty_key
                population.best_genome = None
                for species in population.species.species.values():
                    species.fitness_history.clear()
                    species.last_improved = generation
                print(f"Speed phase at generation {generation}: wheel target cap {speed:.2f} m/s", flush=True)
            generation_stage = curriculum_state.stage
            generation_failures: list[dict] = []
            if fast:
                tracks = fast_training_batch(settings.seed, generation, bank, forbidden, seen_hashes)
            else:
                tracks = training_batch(settings.seed, settings.strategy, generation, settings.generations,
                                    fixed, forbidden, seen_hashes, generator, settings.episodes_per_genome,
                                    settings.track_profile, generation_failures=generation_failures,
                                    curriculum_state=curriculum_state, curriculum_config=settings.curriculum)
            track_generation_wall = time.perf_counter()-generation_started
            generation_dir = output / "tracks" / f"generation-{generation:04d}"
            for index, track in enumerate(tracks):
                track.save(generation_dir / f"{index}.json")
            if settings.track_profile == "corner":
                generator_parameters = [None] * 3 + [asdict(corner_transfer_geometry(generator, generation_stage, tracks[3].difficulty))]
            else:
                generator_parameters = [asdict(stage_geometry(generator, generation_stage, t.difficulty, settings.curriculum))
                                        if settings.strategy == "adaptive" else asdict(generator) for t in tracks]
            write_json(generation_dir / "batch.json", {"generation": generation, "condition": settings.strategy,
                       "track_profile": settings.track_profile,
                       "generation_failures": generation_failures,
                       "curriculum_stage": curriculum_state.stage, "curriculum_config": asdict(settings.curriculum),
                       "generator_parameters": generator_parameters,
                       "track_lengths_m": [t.route.length for t in tracks],
                       "difficulty_batch": [track.difficulty for track in tracks],
                       "tracks": [f"{i}.json" for i in range(len(tracks))], "hashes": [track.geometry_hash for track in tracks]})
            if generation_failures:
                print(f"Retried {len(generation_failures)} infeasible track seeds; recorded in {generation_dir / 'batch.json'}", flush=True)
            print(f"Starting generation {generation}: {len(population.population)} genomes x {len(tracks)} episodes = "
                  f"{len(population.population)*len(tracks)} training episodes; {len(validation)} champion validation episodes.", flush=True)
            if not (output / "preview.json").exists():
                candidate = next(iter(population.population.values()))
                candidate_path = Path("champions") / f"candidate-{generation:04d}.pkl"
                (output / candidate_path).parent.mkdir(exist_ok=True)
                save_model(output / candidate_path, candidate, population.config,
                            {"robot": asdict(phase_robot), "episode": asdict(episode), "generation": generation,
                            "condition": f"{condition}_unscored_candidate", "evolution_seed": settings.seed,
                            "track_profile": settings.track_profile,
                            "training_backend": settings.backend, "time_limit_override": settings.time_limit,
                            "aggregation": settings.aggregation, "cvar_fraction": settings.cvar_fraction})
                write_json(output / "preview.json", {"generation": generation, "model": str(candidate_path),
                           "phase": "Unscored candidate replay", "tracks": [str((generation_dir/f"{i}.json").relative_to(output)) for i in range(len(tracks))]})
            metrics: dict = {}
            if settings.track_profile == "corner" or settings.speed_schedule:
                species_sizes = {sid: len(species.members) for sid, species in population.species.species.items()}

            def evaluate_generation(genomes, _neat_config):
                live = None
                if live_preview:
                    live = {"directory": str((output / "live").resolve()), "token": uuid4().hex}
                    # Every worker has finished before the next roster is made.
                    # Bound disposable telemetry files across long research runs.
                    for snapshot in (output / "live").glob("*.json"):
                        if snapshot.name != "population.json":
                            try:
                                snapshot.unlink(missing_ok=True)
                            except PermissionError:
                                pass  # Old tokens are ignored if a Windows reader still holds this file.
                    initial = Simulation(phase_robot)
                    initial.reset(tracks[0])
                    atomic_snapshot(output / "live" / "population.json", {
                        **live, "generation": generation, "backend": settings.backend,
                        "wheel_speed_cap_m_s": phase_robot.max_speed,
                        "genomes": [key for key, _ in genomes], "initial_qpos": initial.data.qpos.tolist(),
                        "tracks": [str((generation_dir/f"{i}.json").resolve()) for i in range(len(tracks))]})
                training_started = time.perf_counter()
                evaluated = evaluator.evaluate([genome for _, genome in genomes], tracks,
                                               _progress_reporter(output, generation, "training"), live=live)
                metrics["training_eval_wall_seconds"] = time.perf_counter()-training_started
                if evaluator.gpu is not None:
                    append_csv(output / "gpu-profile.csv", [dict(generation=generation, phase="training", **p)
                               for p in evaluator.gpu.batch_profiles], ["generation", "phase"]+list(evaluator.gpu.batch_profiles[0]))
                for key, genome in genomes:
                    genome.fitness = aggregate_fitness([result.score for result in evaluated[key]], settings.aggregation, settings.cvar_fraction)
                _report_training_fitness(output, generation, evaluated, tracks, settings.aggregation, settings.cvar_fraction)
                training_rows = [{"generation": generation, "genome_id": key, **asdict(result)}
                                 for key, results in evaluated.items() for result in results]
                append_csv(output / "training-episodes.csv", training_rows,
                           ["generation", "genome_id"] + list(EpisodeResult.__dataclass_fields__))
                winner = max((genome for _, genome in genomes), key=lambda genome: genome.fitness)
                metrics["training_counts"] = training_competence(evaluated, {g.key:g.fitness for _, g in genomes}, settings.curriculum.elite_fraction)
                validation_started = time.perf_counter()
                if fast:
                    held_out = evaluator.validate(winner, validation, _progress_reporter(output, generation, "validation"),
                                                  detailed=True)
                else:
                    held_out = evaluator.validate(winner, validation, _progress_reporter(output, generation, "validation"))
                metrics["validation_eval_wall_seconds"] = time.perf_counter()-validation_started
                if evaluator.gpu is not None:
                    append_csv(output / "gpu-profile.csv", [dict(generation=generation, phase="validation", **p)
                               for p in evaluator.gpu.batch_profiles], ["generation", "phase"]+list(evaluator.gpu.batch_profiles[0]))
                diagnostic_rows = episode_summaries(generation, "training_population", [r for results in evaluated.values() for r in results])
                diagnostic_rows += episode_summaries(generation, "training_champion", evaluated[winner.key])
                diagnostic_rows += episode_summaries(generation, "validation_champion", held_out)
                append_csv(output / "diagnostics.csv", diagnostic_rows, list(diagnostic_rows[0]))
                completion = float(np.mean([result.completed for result in held_out]))
                validation_score = aggregate_fitness([result.score for result in held_out], settings.aggregation, settings.cvar_fraction)
                metrics.update({"winner": winner, "validation_key": (completion, validation_score),
                                "train_scores": [genome.fitness for _, genome in genomes],
                                "physics_steps": sum(result.physics_steps for episodes in evaluated.values() for result in episodes)
                                                 + sum(result.physics_steps for result in held_out),
                                "episodes": len(genomes) * len(tracks) + len(validation)})
                if fast:
                    selection_key, quality = quality_gate(held_out, reference, validation_groups)
                    final_phase = generation >= settings.speed_schedule[-1][0]
                    metrics["selection_key"] = selection_key if final_phase else None
                    network = NEATController(winner, population.config).network
                    active = {node for node, *_ in network.node_evals}
                    outputs = population.config.genome_config.output_keys
                    append_csv(output / "quality.csv", [{"generation": generation, "final_phase": final_phase,
                               "drive_evaluated": outputs[0] in active, "steering_evaluated": outputs[1] in active,
                               **quality}], ["generation", "final_phase", "drive_evaluated", "steering_evaluated", *quality])
                    from .fitness import fitness_components
                    for role, records in (("training", [r for episodes in evaluated.values() for r in episodes]),
                                          ("validation", held_out)):
                        parts = [fitness_components(r.progress, r.completed, r.elapsed_s, r.time_limit_s, r.legacy_return,
                                                    r.control_steps, episode.completion_speed_bonus, r.command_jitter,
                                                    episode.command_jitter_penalty, r.route_length_m, episode.fitness_version)
                                 for r in records]
                        component_row = {"generation": generation, "role": role,
                                         **{name: float(np.mean([p[name] for p in parts])) for name in parts[0]}}
                        append_csv(output / "fitness-components.csv", [component_row], list(component_row))
                rows = [dict(generation=generation, role=role, **asdict(result))
                        for role, results in (("training_champion", evaluated[winner.key]), ("validation_champion", held_out))
                        for result in results]
                append_csv(output / "champion-episodes.csv", rows, ["generation", "role"] + list(EpisodeResult.__dataclass_fields__))

            # Exactly one library generation: selection and reproduction stay NEAT's.
            population.run(evaluate_generation, 1)
            if settings.track_profile == "corner" or settings.speed_schedule:
                winner = metrics["winner"]
                nodes, connections = winner.size()
                evolution = {"generation": generation, "species_sizes": json.dumps(species_sizes, sort_keys=True),
                             "champion_id": winner.key, "champion_nodes": nodes,
                             "champion_enabled_connections": connections}
                append_csv(output / "evolution.csv", [evolution], list(evolution))
            if settings.strategy == "adaptive":
                progression = curriculum_state.observe(metrics["training_counts"], settings.curriculum)
                append_csv(output / "curriculum.csv", [dict(generation=generation, **progression)], ["generation"]+list(progression))
                print(f"Curriculum stage {progression['stage']} -> {progression['next_stage']}: {progression['reason']} "
                      f"(training-only rolling completion: B={progression['rolling_beginner_completion']:.2f}, "
                      f"I={progression['rolling_intermediate_completion']:.2f}, A={progression['rolling_advanced_completion']:.2f})", flush=True)
            total_physics_steps += metrics["physics_steps"]
            champion_path = Path("champions") / f"generation-{generation:04d}.pkl"
            (output / champion_path).parent.mkdir(exist_ok=True)
            save_model(output / champion_path, metrics["winner"], population.config,
                       {"condition": condition, "generation": generation, "track_profile": settings.track_profile,
                        "robot": asdict(phase_robot), "episode": asdict(episode), "time_limit_override": settings.time_limit,
                        "aggregation": settings.aggregation, "cvar_fraction": settings.cvar_fraction})
            write_json(output / "preview.json", {"generation": generation, "model": str(champion_path), "phase": "Champion replay",
                       "tracks": [str((generation_dir / f"{i}.json").relative_to(output)) for i in range(len(tracks))]})
            candidate_key = metrics.get("selection_key") if fast else metrics["validation_key"]
            if candidate_key is not None and candidate_key > best_key:
                best_key = candidate_key
                save_model(output / "best.pkl", metrics["winner"], population.config,
                           {"condition": condition, "evolution_seed": settings.seed, "track_profile": settings.track_profile,
                            "generation": generation, "validation_key": best_key, "robot": asdict(phase_robot),
                            "training_backend": settings.backend, "validation_backend": settings.backend,
                            "episode": asdict(episode), "time_limit_override": settings.time_limit,
                            "aggregation": settings.aggregation, "cvar_fraction": settings.cvar_fraction,
                            "suite_sha256": suite_hash, "fitness_version": episode.fitness_version})
            log = {"generation": generation, "condition": settings.strategy, "seed": settings.seed,
                   "training_mean": float(np.mean(metrics["train_scores"])), "training_best": float(max(metrics["train_scores"])),
                   "validation_completion_rate": metrics["validation_key"][0], "validation_score": metrics["validation_key"][1],
                   "physics_steps": metrics["physics_steps"], "episode_slots": metrics["episodes"],
                   "wall_seconds": time.perf_counter()-generation_started, "backend": settings.backend,
                   "track_generation_wall_seconds": track_generation_wall,
                   "training_eval_wall_seconds": metrics["training_eval_wall_seconds"],
                   "validation_eval_wall_seconds": metrics["validation_eval_wall_seconds"],
                   "curriculum_stage": generation_stage, "curriculum_stage_next": curriculum_state.stage,
                   "gpu_world_steps_total": evaluator.gpu.executed_world_steps if evaluator.gpu else 0,
                   "gpu_padding_world_steps_total": evaluator.gpu.padding_world_steps if evaluator.gpu else 0}
            if settings.speed_schedule:
                log["wheel_speed_cap_m_s"] = speed
            append_csv(output / "generations.csv", [log], list(log))
            _save_checkpoint(output / "checkpoints" / "latest.pkl.gz",
                             {"checkpoint_version": CHECKPOINT_VERSION, "neat_version": NEAT_VERSION,
                              "population": population, "random_state": random.getstate(), "settings": settings,
                              "robot": robot, "episode": episode, "suite_hash": suite_hash, "best_key": best_key,
                              "source_sha256": environment["source_sha256"],
                              "dependencies": environment["dependencies"],
                              "quality_reference_sha256": reference_hash,
                              "best_model_bytes": (output / "best.pkl").read_bytes() if (output / "best.pkl").exists() else None,
                              "curriculum_state": curriculum_state,
                              "gpu_world_steps_total": evaluator.gpu.executed_world_steps if evaluator.gpu else 0,
                              "gpu_padding_world_steps_total": evaluator.gpu.padding_world_steps if evaluator.gpu else 0,
                              "seen_hashes": seen_hashes, "total_physics_steps": total_physics_steps})
            generations_this_call += 1
            print(f"generation={generation} condition={settings.strategy} validation_completion={metrics['validation_key'][0]:.3f} validation_score={metrics['validation_key'][1]:.4f}", flush=True)
            if stop_after is not None and generations_this_call >= stop_after:
                break
    write_json(output / "status.json", {"next_generation": population.generation, "target_generations": settings.generations,
               "complete": population.generation >= settings.generations, "best_validation_key": best_key,
               "physics_steps_total": total_physics_steps, "wall_seconds_this_invocation": time.perf_counter() - start_time})
    return output / "best.pkl"


def evaluate(output: str | Path, tracks: list[TrackSpec], robot: RobotConfig | None = None,
             episode: EpisodeConfig | None = None, model_path: str | Path | None = None,
             pid: PIDConfig | None = None, repeats: int = 1, telemetry: bool = False,
             time_limit: float | None = None, evaluation_label: str | None = None) -> Path:
    """Use the same episode loop for PID and NEAT; save all attempted episodes."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {output}")
    if repeats < 1 or not tracks:
        raise ValueError("Evaluation needs tracks and at least one repeat")
    metadata = {}
    if model_path:
        controller, metadata = load_model(model_path)
        trained_robot = RobotConfig(**metadata["robot"])
        robot = robot or trained_robot
        if robot.sensor_layout != trained_robot.sensor_layout:
            raise ValueError("Observation layout differs from the saved policy")
        condition = metadata["condition"]
    else:
        robot = robot or RobotConfig()
        controller = PIDController(pid or PIDConfig(sensor_layout=robot.sensor_layout))
        condition = "pid_original" if pid is None else "pid_configured"
    episode = episode or EpisodeConfig(**metadata.get("episode", {}))
    if time_limit is None:
        time_limit = metadata.get("time_limit_override")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation.json", {"condition": condition, "model_path": str(model_path) if model_path else None,
               "evaluation_backend": "cpu",
               "model_metadata": metadata, "robot": asdict(robot), "episode": asdict(episode),
               "pid": asdict(controller.config) if isinstance(controller, PIDController) else None,
               "repeats": repeats, "repeat_semantics": "Deterministic checks, not independent trials",
               "environment": environment_record(), "time_limit_override": time_limit,
               "track_hashes": [track.geometry_hash for track in tracks]})
    simulation = Simulation(robot)
    results_path = output / "episodes.csv"
    errors = []
    for index, track in enumerate(tracks):
        track.save(output / "tracks" / f"{index:03d}.json")
        mask = TrackMask(track)
        for repeat in range(repeats):
            trace = output / "telemetry" / f"track-{index:03d}-repeat-{repeat}.csv" if telemetry else None
            result = run_episode(simulation, track, controller, episode, mask, trace, time_limit)
            row = {"condition": condition, "policy_id": metadata.get("evolution_seed", "pid"),
                   "evolution_seed": metadata.get("evolution_seed", ""), "repeat_id": repeat,
                   "evaluation_speed_limit_m_s": robot.max_speed, **asdict(result)}
            if evaluation_label:
                row["split"] = evaluation_label
            append_csv(results_path, [row], SUMMARY_FIELDS)
            if result.error:
                errors.append(result.error)
    if errors:
        raise RuntimeError(f"Evaluation saved explicit error rows: {errors}")
    return results_path


def seen_tracks(run: str | Path, count: int = 30) -> list[TrackSpec]:
    """A reproducible diagnostic sample; never used for held-out model selection."""
    unique = {}
    for path in sorted((Path(run) / "tracks").glob("generation-*/*.json")):
        if path.name == "batch.json":
            continue
        track = TrackSpec.load(path)
        unique.setdefault(track.geometry_hash, track)
    rng = np.random.default_rng(derived_seed(0, "seen-diagnostic"))
    indices = rng.permutation(len(unique))[:count]
    tracks = list(unique.values())
    return [tracks[index] for index in indices]
