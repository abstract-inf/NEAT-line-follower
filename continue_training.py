"""Continue an evolved population in a separately recorded training phase.

Run --dry-run to inspect, --prepare-only to save the phase without training,
or run this script normally to train/resume it. Never initializes a population.
Only load trusted local checkpoints: Python pickle can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import shutil
import zipfile

from line_follower.config import freeze_geometry, verify_geometry
from line_follower.controllers import NEAT_VERSION
from line_follower.experiments import (CHECKPOINT_VERSION, _save_checkpoint, append_csv, environment_record,
                                       snapshot_sources, train, write_json)
from line_follower.fitness import FITNESS_VERSION
from line_follower.suites import read_suite, progressive_suite
from line_follower.curriculum import CurriculumConfig, CurriculumState
from line_follower.viewer import training_preview

# Editable controls for PREPARING a new phase. Once prepared, saved settings win.
PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_RUN = "runs/28-10-2026"
OUTPUT_DIRECTORY = "runs/28-10-2026-hard"
ADDITIONAL_GENERATIONS = 100  # Generation labels continue from 100 through 199.
MAX_SPEED = 0.35  # m/s wheel-rim target ceiling; slowing/reversing remain possible.
COMPLETION_SPEED_BONUS = 0.0  # Preserve the original episode score; time was not the main failure.
TRACK_PROFILE = "progressive"  # Performance-based length/complexity progression.
EPISODES_PER_GENOME = 10  # Distinct, fresh geometries shared across the population.
AGGREGATION = "mean_min"  # Baseline; optional "mean_cvar" uses a lower-tail mean.
CVAR_FRACTION = 0.30  # Three worst scores when there are ten episodes.
CURRICULUM = CurriculumConfig()  # Editable rolling window, thresholds and geometry scales.
EPISODE_TIME_LIMIT = None  # None uses the bounded route-based allowance below.
MINIMUM_EPISODE_TIME = 45.0  # simulated seconds
MAXIMUM_EPISODE_TIME = 120.0
CONSERVATIVE_ROUTE_SPEED = .10  # m/s average, including turns; not a motor command.
HEADLESS = True  # Set False, or pass --visualize, to observe the live population.


def archived_source(archive_path: Path) -> tuple[str, list[dict]]:
    """Verify parent provenance and report every package source change.

    This is an explicit scientific phase transition, unlike prepare_resume.py's
    observer-only recovery. The old archive must match the parent checkpoint;
    new source differences are recorded rather than disguised as exact resume.
    """
    suffixes = {".py", ".xml", ".json", ".txt"}
    current = {p.relative_to(PROJECT_ROOT).as_posix(): p.read_bytes()
               for p in (PROJECT_ROOT / "line_follower").rglob("*") if p.suffix in suffixes}
    digest, changes = hashlib.sha256(), []
    with zipfile.ZipFile(archive_path) as archive:
        old = {name: archive.read(name) for name in archive.namelist()
               if name.startswith("line_follower/") and Path(name).suffix in suffixes}
    for name in sorted(old, key=lambda n: Path(n).relative_to("line_follower")):
        digest.update(str(Path(name).relative_to("line_follower")).encode())
        digest.update(old[name])
    for name in sorted(old.keys() | current.keys()):
        before, after = old.get(name), current.get(name)
        if before != after:
            changes.append({"file": name,
                            "before_sha256": hashlib.sha256(before).hexdigest() if before is not None else None,
                            "after_sha256": hashlib.sha256(after).hexdigest() if after is not None else None})
    return digest.hexdigest(), changes


def prepare_continuation(parent: Path, output: Path, *, additional_generations: int = ADDITIONAL_GENERATIONS,
                         max_speed: float = MAX_SPEED, completion_speed_bonus: float = COMPLETION_SPEED_BONUS,
                         track_profile: str = TRACK_PROFILE, episodes_per_genome: int = EPISODES_PER_GENOME,
                         aggregation: str = AGGREGATION, curriculum: CurriculumConfig = CURRICULUM,
                         dry_run: bool = False) -> dict:
    """Fork a completed checkpoint without mutation, reproduction or evaluation.

    Preserve genomes, topology, species membership, ancestry, innovation/index
    counters, RNG and seen-track hashes. Reset fitness/stagnation histories and
    model selection because scores from different tasks are not comparable.
    """
    parent, output = parent.resolve(), output.resolve()
    if parent == output or parent in output.parents:
        raise ValueError("Use a separate output directory outside the parent run")
    if output.exists():
        raise FileExistsError(f"Phase already exists: {output}; resume it instead")
    if type(additional_generations) is not int or additional_generations < 1:
        raise ValueError("additional_generations must be a positive integer")
    checkpoint = parent / "checkpoints/latest.pkl.gz"
    with gzip.open(checkpoint, "rb") as stream:
        saved = pickle.load(stream)
    if saved["checkpoint_version"] != CHECKPOINT_VERSION or saved["neat_version"] != NEAT_VERSION:
        raise ValueError("Unsupported parent checkpoint version")
    start = saved["population"].generation
    if start < saved["settings"].generations:
        raise ValueError("Parent training has not finished; use ordinary resume first")
    record = json.loads((parent / "run.json").read_text(encoding="utf-8"))
    suite = Path(record["suite_path"])
    if hashlib.sha256(suite.read_bytes()).hexdigest() != saved["suite_hash"]:
        raise ValueError("Parent suite changed")
    read_suite(suite)
    if record["geometry_freeze"]["robot"] != asdict(saved["robot"]):
        raise ValueError("Parent robot provenance does not match checkpoint")
    old_hash, source_changes = archived_source(parent / "source.zip")
    if old_hash != saved["source_sha256"]:
        raise ValueError("Parent source.zip does not match checkpoint")
    environment = environment_record()
    if environment["dependencies"] != saved["dependencies"]:
        raise ValueError("Dependencies changed since parent checkpoint")
    adaptive = track_profile == "progressive"
    settings = replace(saved["settings"], strategy="adaptive" if adaptive else "multi", generations=start + additional_generations,
                       track_profile=track_profile, episodes_per_genome=episodes_per_genome if adaptive else saved["settings"].episodes_per_genome,
                       time_limit=EPISODE_TIME_LIMIT if adaptive else saved["settings"].time_limit,
                       aggregation=aggregation, cvar_fraction=CVAR_FRACTION, curriculum=curriculum)
    robot = replace(saved["robot"], max_speed=max_speed)
    episode = replace(saved["episode"], completion_speed_bonus=completion_speed_bonus,
                      minimum_time_limit=MINIMUM_EPISODE_TIME if adaptive else saved["episode"].minimum_time_limit,
                      maximum_time_limit=MAXIMUM_EPISODE_TIME if adaptive else saved["episode"].maximum_time_limit,
                      time_reference_speed=CONSERVATIVE_ROUTE_SPEED if adaptive else saved["episode"].time_reference_speed)
    if track_profile == "zigzag" and "zigzag" in settings.excluded_features:
        raise ValueError("Cannot focus on an excluded feature")
    new_manifest = progressive_suite(suite, output / "suite.json", write=False) if adaptive else read_suite(suite)
    suite_bytes = json.dumps(new_manifest, indent=2).encode("utf-8") if adaptive else suite.read_bytes()
    next_suite = output / "suite.json" if adaptive else suite
    curriculum_state = CurriculumState()
    initialization = {"source": "No prior training diagnostics; begin at stage 0"}
    champions = parent / "champion-episodes.csv"
    if adaptive and champions.exists():
        with champions.open(newline="", encoding="utf-8") as stream:
            training_rows = [r for r in csv.DictReader(stream) if r["role"] == "training_champion"
                             and start-curriculum.window <= int(r["generation"]) < start]
        beginner = [r for r in training_rows if r["difficulty"] == "beginner"]
        rate = sum(r["completed"].lower() == "true" for r in beginner)/max(1, len(beginner))
        if len(beginner) >= curriculum.window and rate >= curriculum.beginner_threshold:
            curriculum_state.stage = 1  # New longer levels must still demonstrate competence.
        initialization = {"source": "Parent training champions only", "beginner_completion": rate,
                          "episodes": len(beginner), "csv_sha256": hashlib.sha256(champions.read_bytes()).hexdigest()}
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "parent_run": str(parent),
             "parent_checkpoint": str(checkpoint),
             "parent_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             "parent_source_sha256": old_hash, "source_changes": source_changes,
             "parent_best_validation_key": saved["best_key"], "start_generation": start,
             "target_generation": settings.generations, "population_count": len(saved["population"].population),
             "inherited_track_hashes": len(saved["seen_hashes"]),
             "parent_physics_steps_total": saved["total_physics_steps"],
             "parent_gpu_world_steps_total": saved.get("gpu_world_steps_total", 0),
             "parent_gpu_padding_world_steps_total": saved.get("gpu_padding_world_steps_total", 0),
             "changes": {"strategy": [saved["settings"].strategy, settings.strategy],
                         "track_profile": [saved["settings"].track_profile, track_profile],
                         "max_speed_m_s": [saved["robot"].max_speed, robot.max_speed],
                         "completion_speed_bonus": [saved["episode"].completion_speed_bonus, episode.completion_speed_bonus]},
             "fitness_history_reset": True, "validation_model_selection_reset": True,
             "initial_curriculum_stage": curriculum_state.stage, "curriculum_initialization": initialization,
             "settings_before": asdict(saved["settings"]), "settings_after": asdict(settings),
             "parent_suite_sha256": saved["suite_hash"], "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
             "training_started": False}
    result = {"output_directory": str(output), "settings": asdict(settings), "robot": asdict(robot),
              "episode": asdict(episode), "transition": audit}
    if dry_run:
        return result
    # No creation until all parent checks pass. Never overwrite an existing run.
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    if adaptive:
        (output / "suite.json").write_bytes(suite_bytes)
    shutil.copyfile(checkpoint, output / "parent-checkpoint.pkl.gz")
    shutil.copyfile(parent / "source.zip", output / "parent-source.zip")
    shutil.copyfile(parent / "run.json", output / "parent-run.json")
    (output / "parent-best.pkl").write_bytes(saved["best_model_bytes"])
    shutil.copytree(parent / "frozen-validation", output / "frozen-validation")
    shutil.copyfile(parent / "single-training-track.json", output / "single-training-track.json")
    saved["population"].config.save(str(output / "neat-config.txt"))
    freeze_geometry(output / "geometry-freeze.json", robot)
    selected_geometry = verify_geometry(output / "geometry-freeze.json", robot)
    population = saved["population"]
    population.best_genome = None
    for genome in population.population.values():
        genome.fitness = None
    for species in population.species.species.values():
        species.fitness_history = []
        species.last_improved = start
        species.fitness = species.adjusted_fitness = None
    saved.update(settings=settings, robot=robot, episode=episode, source_sha256=environment["source_sha256"],
                 best_key=(-1.0, -float("inf")), best_model_bytes=None, total_physics_steps=0,
                 gpu_world_steps_total=0, gpu_padding_world_steps_total=0, curriculum_state=curriculum_state,
                 suite_hash=hashlib.sha256(suite_bytes).hexdigest())
    record.update(settings=asdict(settings), robot=asdict(robot), episode=asdict(episode),
                  geometry_freeze=selected_geometry, environment=environment,
                  fitness_version=FITNESS_VERSION, continuation=audit, live_population_observer=False,
                  suite_path=str(next_suite), suite_sha256=saved["suite_hash"],
                  generator=new_manifest.get("training_generator_config", new_manifest["generator_config"]))
    write_json(output / "run.json", record)
    write_json(output / "robot.json", asdict(robot))
    write_json(output / "episode.json", asdict(episode))
    write_json(output / "phase.json", audit)
    snapshot_sources(output / "source.zip")
    _save_checkpoint(output / "checkpoints/latest.pkl.gz", saved)
    write_json(output / "status.json", {"next_generation": start, "target_generations": settings.generations,
                                       "complete": False, "training_started": False})
    return result


def phase_checkpoint(output: Path) -> dict:
    """Validate a prepared phase before opening a window or starting evaluation."""
    audit = json.loads((output / "phase.json").read_text(encoding="utf-8"))
    with gzip.open(output / "checkpoints/latest.pkl.gz", "rb") as stream:
        saved = pickle.load(stream)
    environment = environment_record()
    if saved["source_sha256"] != environment["source_sha256"] or saved["dependencies"] != environment["dependencies"]:
        raise ValueError("Phase source/dependencies changed; restore its archived environment before resuming")
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    suite = Path(record["suite_path"])
    if saved["suite_hash"] != hashlib.sha256(suite.read_bytes()).hexdigest():
        raise ValueError("Phase suite changed")
    verify_geometry(output / "geometry-freeze.json", saved["robot"])
    if not audit["start_generation"] <= saved["population"].generation <= saved["settings"].generations:
        raise ValueError("Phase generation is outside recorded bounds")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PROJECT_ROOT / PARENT_RUN)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / OUTPUT_DIRECTORY)
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; no writes or training")
    parser.add_argument("--prepare-only", action="store_true", help="Save checkpoint/configuration only; no training")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--visualize", dest="headless", action="store_false")
    display.add_argument("--headless", dest="headless", action="store_true")
    parser.set_defaults(headless=HEADLESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.exists():
        result = prepare_continuation(args.parent, output, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    saved = phase_checkpoint(output) if output.exists() else None
    if args.dry_run or args.prepare_only:
        if saved:
            print(json.dumps({"output_directory": str(output), "next_generation": saved["population"].generation,
                              "settings": asdict(saved["settings"]), "robot": asdict(saved["robot"]),
                              "episode": asdict(saved["episode"]), "training_started": False}, indent=2))
        return
    print(f"Resuming full population at generation {saved['population'].generation}: {output}", flush=True)
    print("Using saved phase settings; editable globals apply when preparing a new phase.", flush=True)
    if saved["population"].generation >= saved["settings"].generations:
        print("Phase is complete. Prepare another output directory to extend it.")
        return
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    launch = {"started_utc": datetime.now(timezone.utc).isoformat(), "next_generation": saved["population"].generation,
              "headless": args.headless, "workers": saved["settings"].workers, "backend": saved["settings"].backend,
              "gpu_batch_size": saved["settings"].gpu_batch_size}
    append_csv(output / "launches.csv", [launch], list(launch))
    with training_preview(output, enabled=not args.headless, robot=saved["robot"]):
        print(train(output, record["suite_path"], saved["settings"], saved["robot"], saved["episode"],
                    resume=output / "checkpoints/latest.pkl.gz", geometry_lock=output / "geometry-freeze.json",
                    live_preview=not args.headless))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
