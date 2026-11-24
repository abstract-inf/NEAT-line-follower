"""User-operated training entry point. Importing this file never starts training.

Edit the clearly named settings below, finalize geometry, then run this file.
Use --dry-run to inspect settings without creating a population or evaluating it.
"""

from dataclasses import asdict
import gzip
import json
import multiprocessing
from pathlib import Path
import pickle
import sys

from line_follower.config import EpisodeConfig, RobotConfig, verify_geometry
from line_follower.experiments import TrainConfig, train
from line_follower.viewer import training_preview

# Paths are relative to this project, regardless of the shell's working folder.
PROJECT_ROOT = Path(__file__).resolve().parent
ROBOT_CONFIG = "configs/robot-competition.json"
GEOMETRY_LOCK = "configs/geometry-freeze.json"
TRACK_SUITE = "benchmarks/progressive-v2/suite.json"
OUTPUT_DIRECTORY = "runs/28-10-2026"  # Fresh sessions add -02, -03, ... if this folder exists.

# Research controls for NEW runs. A resume restores the checkpoint's settings.
STRATEGY = "adaptive"  # Training-based progression; "curriculum" retains the generation schedule.
EVOLUTION_SEED = 0
POPULATION_SIZE = 200
GENERATIONS = 100
EPISODES_PER_GENOME = 10  # Ten fresh shared tracks per network, per generation.
EPISODE_TIME_LIMIT = None  # Route-based: 45-120 s, using a conservative 0.10 m/s average.
TRACK_PROFILE = "progressive"
AGGREGATION = "mean_min"  # Optional "mean_cvar"; preserve the original baseline by default.
CVAR_FRACTION = .30
WORKERS = 4  # Persistent Windows-compatible CPU workers; start here and measure.
BACKEND = "gpu"  # "gpu" selects optional experimental CUDA batching, not CPU workers.
GPU_BATCH_SIZE = POPULATION_SIZE  # All genomes evaluate one track together; reduce if CUDA memory is insufficient.

HEADLESS = True  # False observes the entire population; rendering never paces training.
PREVIEW_POPULATION_GRID = True  # Separate arenas for every genome; G toggles a shared-track overlay.
PREVIEW_REPLAY_SPEED = 10  # For older runs with champion-only previews; live population always follows actual workers.

# EXAMPLE:  RESUME_CHECKPOINT = "runs/my-curriculum-run/checkpoints/resume-eacb3caf4cb8.pkl.gz"
RESUME_CHECKPOINT = None  # Recover last generation; set None for a fresh session.


def new_run_directory(base: Path, *, create: bool = False) -> Path:
    """Choose an unused folder; atomic creation also protects concurrent launches.

    Dry runs only inspect paths. Resuming bypasses this function and keeps the
    configured original output directory, as required by checkpoint validation.
    """
    candidate = base
    number = 1
    while True:
        if create:
            try:
                candidate.mkdir(parents=True)
                return candidate
            except FileExistsError:
                if not candidate.exists():
                    raise  # A parent path is a file; numbering cannot repair that.
        elif not candidate.exists():
            return candidate
        number += 1
        candidate = base.with_name(f"{base.name}-{number:02d}")


def session_training_settings(requested: TrainConfig, resume: Path | None) -> tuple[TrainConfig, dict]:
    """Match CLI resume semantics without changing settings of the saved study.

    User-edited globals remain available for the next fresh session. Return
    overridden values for explicit reporting rather than silently ignoring them.
    Robot/episode compatibility is still checked independently by train().
    """
    if resume is None:
        return requested, {}
    with gzip.open(resume, "rb") as stream:
        saved = pickle.load(stream)["settings"]
    requested_values, saved_values = asdict(requested), asdict(saved)
    overrides = {name: {"requested": value, "checkpoint": saved_values[name]}
                 for name, value in requested_values.items() if value != saved_values[name]}
    return saved, overrides


def main() -> None:
    robot = RobotConfig.load(PROJECT_ROOT / ROBOT_CONFIG)
    settings = TrainConfig(strategy=STRATEGY, seed=EVOLUTION_SEED, generations=GENERATIONS,
                           population=POPULATION_SIZE, workers=WORKERS, backend=BACKEND,
                           gpu_batch_size=GPU_BATCH_SIZE, episodes_per_genome=EPISODES_PER_GENOME,
                           time_limit=EPISODE_TIME_LIMIT, track_profile=TRACK_PROFILE,
                           aggregation=AGGREGATION, cvar_fraction=CVAR_FRACTION)
    base_output = PROJECT_ROOT / OUTPUT_DIRECTORY
    resume = PROJECT_ROOT / RESUME_CHECKPOINT if RESUME_CHECKPOINT else None
    settings, overrides = session_training_settings(settings, resume)
    episode = EpisodeConfig(maximum_time_limit=120, time_reference_speed=.10)
    if resume:
        with gzip.open(resume, "rb") as stream:
            episode = pickle.load(stream)["episode"]
    if "--dry-run" in sys.argv:
        print(json.dumps({"training": asdict(settings), "training_episodes_per_generation": settings.population*settings.episodes_per_genome,
                          "resume_checkpoint": str(resume) if resume else None,
                          "checkpoint_overrides": overrides,
                          "output_directory": str(base_output if resume else new_run_directory(base_output)),
                          "robot": asdict(robot), "headless": HEADLESS,
                          "geometry_selected": (PROJECT_ROOT / GEOMETRY_LOCK).exists(),
                          "suite_exists": (PROJECT_ROOT / TRACK_SUITE).exists(), "training_started": False}, indent=2))
        return
    verify_geometry(PROJECT_ROOT / GEOMETRY_LOCK, robot)
    if overrides:
        print("Resuming with checkpoint settings. Edited research settings apply to fresh sessions:", flush=True)
        for name, values in overrides.items():
            print(f"  {name}: requested {values['requested']!r}; using checkpoint {values['checkpoint']!r}", flush=True)
    output = base_output if resume else new_run_directory(base_output, create=True)
    print(f"{'Resuming' if resume else 'New training session'}: {output}", flush=True)
    with training_preview(output, enabled=not HEADLESS, robot=robot,
                          replay_speed=PREVIEW_REPLAY_SPEED, grid=PREVIEW_POPULATION_GRID):
        print(train(output, PROJECT_ROOT / TRACK_SUITE, settings, robot,
                    episode,
                    resume=resume, geometry_lock=PROJECT_ROOT / GEOMETRY_LOCK, live_preview=not HEADLESS))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
