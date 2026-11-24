"""Course construction, sampling, and validation selection for the fast study."""

from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .suites import corner_track, create_suite, derived_seed, read_suite
from .tracks import GeneratorConfig, Section, TrackGenerationError, TrackSpec, generate_track, validate_track

SUITE_SEED = 2026111801
SCHEDULE = ((0, .65), (50, .80), (100, 1.00), (150, 1.25), (200, 1.50))
FITNESS_VERSION = "route-speed-1.0"


def centered_course(sections, name, seed, split):
    track = TrackSpec(name, tuple(sections), start=(0, 0, 0), generator_version="2.0", resolution=.002,
                      seed=seed, split=split, difficulty="speed")
    route = track.route
    runout = route.points[-1] + .25 * np.array([math.cos(route.heading_at(route.length)),
                                             math.sin(route.heading_at(route.length))])
    points = np.vstack((route.points, runout))
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    arena = max(4, math.ceil(float(max(upper - lower)) + 1))
    return replace(track, start=(-float(center[0]), -float(center[1]), 0), arena_size=arena)


def speed_course(seed, slot, split):
    rng = np.random.default_rng(seed)
    kind = slot % 20
    length = rng.uniform(3, 6) if split == "training" else rng.uniform(4, 6)
    sections = [Section("straight", length=length, feature="speed-straight")]
    sign = 1 if slot % 2 else -1
    if 6 <= kind < 16:
        radius, angle = rng.uniform(1, 3), sign * rng.uniform(math.pi/12, math.pi/4)
        sections.append(Section("arc", radius=radius, angle=angle, feature="gentle-curve"))
        if kind >= 12:
            sections.append(Section("arc", radius=radius, angle=-angle, feature="broad-s-curve"))
        sections.append(Section("straight", length=rng.uniform(.7, 1.2), feature="speed-straight"))
    elif kind >= 16:
        angle = sign * (math.pi/4 if kind < 18 else math.pi/2)
        sections += [Section("corner", angle=angle, feature="braking-corner"),
                     Section("straight", length=rng.uniform(1, 2), feature="speed-straight")]
    return centered_course(sections, f"speed-{seed}", seed, split)


def prepare_suite(directory, protected):
    suite = create_suite(directory, seed=SUITE_SEED, validation_count=30, test_count=90,
                         protected_hashes=protected)
    manifest = read_suite(suite)
    used = protected | {r["geometry_hash"] for rows in manifest["splits"].values() for r in rows}
    for rows in manifest["splits"].values():
        for row in rows:
            row["group"] = "mixed"
    for split, group, count in (("training", "speed", 20), ("validation", "speed", 20),
                                ("validation", "corner", 10), ("test", "speed", 30), ("test", "corner", 30)):
        for slot in range(count):
            for attempt in range(100):
                seed = derived_seed(SUITE_SEED, f"{split}-{group}", slot, attempt)
                track = (speed_course(seed, slot * 20 // count, split) if group == "speed"
                         else replace(corner_track(seed, 4, slot % 3, "advanced", GeneratorConfig()), split=split))
                if not validate_track(track) and track.geometry_hash not in used:
                    break
            else:
                raise ValueError(f"Could not generate disjoint {split} {group} course {slot}")
            used.add(track.geometry_hash)
            path = f"{split}/{group}-{slot:03d}.json"
            track.save(suite.parent / path)
            manifest["splits"][split].append({"path": path, "geometry_hash": track.geometry_hash, "seed": seed,
                                              "difficulty": track.difficulty, "group": group, "features": track.features,
                                              "dedicated_straight": group == "speed" and all(s.kind == "straight" for s in track.sections)})
    manifest["training_bank"] = [row["path"] for row in manifest["splits"]["training"] if row["group"] == "speed"]
    suite.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return suite


def speed_bank_slots(seed, generation):
    cycle, position = divmod(2 * generation, 20)
    rng = np.random.default_rng(derived_seed(seed, "speed-bank-cycle", cycle))
    return rng.permutation(20)[position:position+2].tolist()


def fast_training_batch(seed, generation, bank, forbidden, seen):
    tracks = [bank[index] for index in speed_bank_slots(seed, generation)]
    for slot in range(4):
        difficulty = "intermediate" if slot == 0 else "advanced"
        for attempt in range(100):
            track_seed = derived_seed(seed, "fast-training", generation, slot, attempt)
            try:
                if slot < 2:
                    track = generate_track(track_seed, difficulty)
                else:
                    # Reverse the order each generation; one three-turn drill in each direction.
                    track = corner_track(track_seed, 3, (generation + slot) % 2, difficulty, GeneratorConfig())
                track = replace(track, split="training")
            except TrackGenerationError:
                continue
            if track.geometry_hash not in forbidden and track.geometry_hash not in seen:
                seen.add(track.geometry_hash)
                tracks.append(track)
                break
        else:
            raise ValueError("Could not generate a fresh fast-study course")
    return tracks


def quality_gate(rows, reference, groups):
    candidate = {row.track_hash: row for row in rows}
    if set(candidate) != set(reference) or set(candidate) != set(groups):
        raise ValueError("Quality gate needs identical validation tracks")
    reasons = []
    for group in ("mixed", "speed", "corner"):
        keys = [key for key in groups if groups[key] == group]
        if sum(candidate[key].completed for key in keys) < sum(reference[key]["completed"] for key in keys):
            reasons.append(f"{group}_completion")
    shared = [key for key in candidate if candidate[key].completed and reference[key]["completed"]]
    for keys, label in ((list(candidate), "all"), (shared, "shared")):
        if not keys:
            reasons.append(f"{label}_empty")
            continue
        for metric in ("geometry_lateral_rmse_m", "command_jitter"):
            values = [getattr(candidate[key], metric) for key in keys]
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("Missing detailed validation metrics")
            if mean(values) > mean(reference[key][metric] for key in keys) + 1e-12:
                reasons.append(f"{label}_{metric}")
    paired_change = mean(candidate[key].completion_time_s - reference[key]["completion_time_s"] for key in shared) if shared else None
    rmse = mean(row.geometry_lateral_rmse_m for row in rows)
    eligible = not reasons
    key = (-paired_change, sum(row.completed for row in rows), -rmse) if eligible else None
    straights = [row for row in rows if reference[row.track_hash].get("dedicated_straight", False)]
    straight_goal = bool(straights) and all(row.completed and row.straight_speed_m_s is not None
                                          and row.straight_speed_m_s >= .95 for row in straights)
    return key, {"eligible": eligible, "reasons": ";".join(reasons),
                 "shared_finishes": len(shared), "paired_time_change_s": paired_change,
                 "geometric_rmse_m": rmse, "command_jitter": mean(row.command_jitter for row in rows),
                 "straight_goal": straight_goal, "completed": sum(row.completed for row in rows)}
