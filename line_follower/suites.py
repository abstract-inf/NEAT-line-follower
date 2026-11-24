"""Frozen held-out manifests and independent, stateless procedural seed streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .tracks import (DIFFICULTIES, GENERATOR_VERSION, GeneratorConfig, Section,
                     TrackGenerationError, TrackSpec, generate_track, validate_track)
from .curriculum import CurriculumConfig, CurriculumState, stage_batch, stage_geometry

SUITE_VERSION = 1
DEFAULT_SUITE_SEED = 20261027


def derived_seed(root: int, namespace: str, *indices: int) -> int:
    """No Python hash() or process scheduling participates in seed derivation."""
    digest = hashlib.sha256(json.dumps([root, namespace, *indices]).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def create_suite(directory: str | Path, seed: int = DEFAULT_SUITE_SEED,
                 validation_count: int = 30, test_count: int = 90,
                 config: GeneratorConfig | None = None, protected_hashes: set[str] | None = None) -> Path:
    """Freeze all held-outs before experiments. Never overwrite an existing suite."""
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Suite directory is not empty: {root}")
    if validation_count < 1 or test_count < 1:
        raise ValueError("Both validation and test sets must be non-empty")
    config = config or GeneratorConfig()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"suite_version": SUITE_VERSION, "generator_version": GENERATOR_VERSION,
                "root_seed": seed, "generator_config": asdict(config), "splits": {}}
    if config.generator_version == "2.0":
        manifest.update(suite_version=2, generator_version="2.0", training_generator_version="2.0",
                        training_generator_config=asdict(config))
    manifest["protected_geometry_hashes"] = sorted(protected_hashes or ())
    manifest["generation_failures"] = []
    identities: set[str] = set(protected_hashes or ())
    for split, count in (("training", 1), ("validation", validation_count), ("test", test_count)):
        entries = []
        for index in range(count):
            difficulty = "advanced" if split == "training" else DIFFICULTIES[index % len(DIFFICULTIES)]
            for attempt in range(100):
                track_seed = derived_seed(seed, split, index, attempt)
                geometry = stage_geometry(config, 4, difficulty) if config.generator_version == "2.0" else config
                try:
                    track = replace(generate_track(track_seed, difficulty, geometry), split=split)
                except TrackGenerationError as error:
                    if config.generator_version == "1.0":
                        raise
                    manifest["generation_failures"].append({"split": split, "index": index, "attempt": attempt,
                                                             "seed": track_seed, "reason": str(error)})
                    continue
                identity = track.geometry_hash
                if identity not in identities:
                    break
            else:
                raise ValueError(f"Could not find distinct {split} geometry at index {index}")
            relative_path = f"{split}/{index:03d}.json"
            track.save(root / relative_path)
            identities.add(identity)
            entries.append({"path": relative_path, "geometry_hash": identity, "seed": track_seed,
                            "difficulty": difficulty, "features": track.features})
        manifest["splits"][split] = entries
    path = root / "suite.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_suite(path: str | Path) -> dict:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if values["suite_version"] not in {1, 2} or values["generator_version"] not in {"1.0", "2.0"}:
        raise ValueError("Unsupported suite/generator version")
    if values["suite_version"] == 2 and values.get("training_generator_version") != "2.0":
        raise ValueError("Unsupported progressive training distribution")
    identities = [entry["geometry_hash"] for entries in values["splits"].values() for entry in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("Suite contains duplicate geometry across or within splits")
    return values


def load_split(path: str | Path, split: str) -> list[TrackSpec]:
    """Verify the actual files against the frozen manifest before using them."""
    manifest = read_suite(path)
    tracks = []
    for entry in manifest["splits"][split]:
        track = TrackSpec.load(Path(path).parent / entry["path"])
        if track.geometry_hash != entry["geometry_hash"] or track.split != split:
            raise ValueError(f"Track was modified or has the wrong split: {entry['path']}")
        tracks.append(track)
    return tracks


def progressive_suite(parent_path: Path, destination: Path, *, write: bool = True) -> dict:
    """Version training separately while reusing the identical frozen held-outs."""
    parent_path, destination = Path(parent_path).resolve(), Path(destination).resolve()
    manifest = read_suite(parent_path)
    manifest.update(suite_version=2, training_generator_version="2.0",
                    parent_suite_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                    training_generator_config=asdict(GeneratorConfig(generator_version="2.0", max_attempts=1600)))
    for entries in manifest["splits"].values():
        for entry in entries:
            entry["path"] = Path(os.path.relpath(parent_path.parent / entry["path"], destination.parent)).as_posix()
    if write:
        if destination.exists():
            raise FileExistsError("Never overwrite a versioned suite")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def difficulty_batch(strategy: str, generation: int, generations: int,
                     episodes_per_genome: int = 4, stage: int = 0,
                     track_profile: str = "standard") -> tuple[str, ...]:
    """The curriculum depends only on the saved generation count, not validation."""
    final = ("beginner", "intermediate", "advanced", "advanced")
    if episodes_per_genome < 1:
        raise ValueError("episodes_per_genome must be positive")
    if track_profile == "corner":
        drills = (("beginner",)*3, ("beginner", "beginner", "intermediate"),
                  ("intermediate", "intermediate", "beginner"),
                  ("intermediate", "intermediate", "advanced"),
                  ("intermediate", "advanced", "advanced"))[stage]
        fraction = generation / generations
        transfer = "beginner" if fraction < .20 else "intermediate" if fraction < .55 else "advanced"
        return (*drills, transfer)
    if strategy == "adaptive":
        return stage_batch(stage, episodes_per_genome, generation)
    if strategy == "single":
        mixture = ("advanced",)*4
    elif strategy == "multi":
        mixture = final
    elif strategy == "curriculum":
        fraction = generation / generations
        mixture = (("beginner",)*4 if fraction < .20 else
                   ("beginner", "beginner", "intermediate", "intermediate") if fraction < .60 else final)
    else:
        raise ValueError("strategy must be single, multi, or curriculum")
    # Whole four-slot mixtures retain their exact proportions. Smaller batches
    # rotate deterministically across generations rather than dropping a level.
    start = generation*episodes_per_genome % 4 if episodes_per_genome % 4 else 0
    return tuple(mixture[(start+i) % 4] for i in range(episodes_per_genome))


def corner_transfer_geometry(config: GeneratorConfig, stage: int, difficulty: str) -> GeneratorConfig:
    motifs, low, high = {"beginner": (3, 1., 2.4), "intermediate": (4, 1.6, 3.8),
                         "advanced": (6, 2.8, 5.5)}[difficulty]
    return replace(stage_geometry(config, stage, difficulty),
                   target_motifs=motifs, route_length_min=low, route_length_max=high)


def corner_track(seed: int, stage: int, slot: int, difficulty: str, config: GeneratorConfig) -> TrackSpec:
    """Short randomized drills plus a distinct slot for procedural transfer."""
    if slot == 3:
        return generate_track(seed, difficulty, corner_transfer_geometry(config, stage, difficulty))

    rng = np.random.default_rng(seed)
    if stage < 2:
        turn_count = 1 if slot < 2 else 2
        limits = ((35, 50), (45, 60)) if stage == 0 else ((65, 85), (75, 90))
        angle = math.radians(rng.uniform(*limits[0 if slot < 2 else 1]))
    else:
        turn_count = 3
        if slot == 2:
            turn_count = 1 if stage == 2 else 4
        if stage == 4 and slot == 1:
            turn_count = 4
        angle = math.pi / 2
    direction = (1, -1, int(rng.choice([-1, 1])))[slot]
    width = float(rng.uniform(.014, .026))
    feature = "corner" if turn_count == 1 else "zigzag"
    sections = [Section("straight", length=float(rng.uniform(.30, .46)), width=width)]
    for i in range(turn_count):
        sections.append(Section("corner", angle=direction * (-1)**i * angle, width=width, feature=feature))
        bounds = (.34, .46) if i == turn_count - 1 else (.21, .33)
        sections.append(Section("straight", length=float(rng.uniform(*bounds)), width=width, feature=feature))
    track = TrackSpec(f"corner-drill-{seed}", tuple(sections), start=(-.82, -.15, 0.), seed=seed,
                      difficulty=difficulty, generator_version="2.0", arena_size=config.arena_size,
                      resolution=config.resolution)
    errors = validate_track(track, config)
    if errors:
        raise TrackGenerationError("; ".join(errors))
    return track


def training_batch(seed: int, strategy: str, generation: int, generations: int,
                   fixed_track: TrackSpec, forbidden_hashes: set[str], seen_hashes: set[str],
                   config: GeneratorConfig, episodes_per_genome: int = 4,
                   track_profile: str = "standard", *,
                   generation_failures: list[dict] | None = None,
                   curriculum_state: CurriculumState | None = None,
                   curriculum_config: CurriculumConfig = CurriculumConfig()) -> list[TrackSpec]:
    """Share fresh tracks across genomes; optionally target zigzag weaknesses.

    With the multi strategy, each four-slot group is beginner, intermediate
    with a guaranteed zigzag, advanced mixed, advanced with two zigzag motifs.
    Additional groups retain that allocation. Default historical batches match.
    """
    if track_profile not in {"standard", "zigzag", "progressive", "corner"}:
        raise ValueError("Unknown training track profile")
    if track_profile == "zigzag" and strategy != "multi":
        raise ValueError("Zigzag continuation uses multi strategy, without restarting curriculum")
    if track_profile == "corner" and (strategy != "adaptive" or episodes_per_genome != 4):
        raise ValueError("Corner drills require four adaptive episode slots")
    if strategy == "single":
        return [fixed_track] * episodes_per_genome
    tracks = []
    stage = curriculum_state.stage if curriculum_state else 0
    for index, difficulty in enumerate(difficulty_batch(strategy, generation, generations, episodes_per_genome,
                                                        stage, track_profile)):
        geometry = stage_geometry(config, stage, difficulty, curriculum_config) if strategy == "adaptive" else config
        for attempt in range(100):
            track_seed = derived_seed(seed, "training", generation, index, attempt)
            focus = "zigzag" if track_profile == "zigzag" and index % 4 in {1, 3} else None
            if track_profile == "progressive" and difficulty != "beginner" and index % 2:
                focus = "zigzag"
            try:
                if track_profile == "corner":
                    generated = corner_track(track_seed, stage, index, difficulty, config)
                else:
                    generated = generate_track(track_seed, difficulty, geometry, required_feature=focus,
                                               required_feature_count=2 if focus and difficulty == "advanced"
                                               and (track_profile == "zigzag" or stage >= 3) else 1)
                track = replace(generated, split="training")
            except TrackGenerationError as error:
                if track_profile == "standard" and strategy != "adaptive":
                    raise  # Preserve the original protocol's failure semantics.
                if generation_failures is not None:
                    generation_failures.append({"slot": index, "attempt": attempt, "seed": track_seed,
                                                "difficulty": difficulty, "reason": str(error)})
                continue  # A new deterministic seed, never relaxed feasibility.
            identity = track.geometry_hash
            if identity not in forbidden_hashes and identity not in seen_hashes:
                seen_hashes.add(identity)
                tracks.append(track)
                break
        else:
            raise ValueError(f"Unable to generate training geometry after 100 seeds: generation={generation}, "
                             f"slot={index}, last_seed={track_seed}; check geometry/held-out constraints")
    return tracks
