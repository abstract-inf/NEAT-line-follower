"""Geometry, reflectance, and split leakage checks with independent expectations."""

from dataclasses import replace
import math

import numpy as np
import pytest

from line_follower.suites import create_suite, difficulty_batch, load_split, read_suite, training_batch
from line_follower.tracks import Branch, GeneratorConfig, Section, TrackMask, TrackSpec, generate_track, straight_track, validate_track


def test_seed_reproduction_and_feature_coverage():
    observed = set()
    identities = set()
    for difficulty in ("beginner", "intermediate", "advanced"):
        for seed in range(12):
            track = generate_track(seed, difficulty)
            assert not validate_track(track)
            assert track == generate_track(seed, difficulty)
            observed.update(track.features)
            identities.add(track.geometry_hash)
    assert len(identities) == 36
    assert {"straight", "gentle_curve", "tight_curve", "corner", "zigzag", "s_curve", "hairpin", "u_turn", "gap", "crossing", "branch"} <= observed


def test_sharp_corner_and_gap_geometry_matches_reflectance():
    track = TrackSpec("corner-gap", (Section("straight", length=0.5), Section("corner", angle=math.pi / 2),
                                     Section("straight", length=0.5)), start=(-0.5, -0.25, 0), gaps=((0.20, 0.26),))
    route = track.route
    assert route.length == pytest.approx(1)
    assert route.corners == (0.5,)
    assert np.any(np.all(np.isclose(route.points, [0, -0.25]), axis=1))
    mask = TrackMask(track)
    points = np.array([[-0.4, -0.25], [-0.27, -0.25], [0, -0.1], [-0.04, -0.21], [0.1, -0.25]])
    assert mask.sample(points).tolist() == [1, 0, 1, 0, 0]
    assert np.all(mask.sample(np.array([[9, 9], [-9, -9]])) == 0)


def test_width_and_distractor_sensing():
    track = replace(straight_track(1.5), branches=(Branch(0.75, crossing=True),))
    mask = TrackMask(track)
    assert mask.sample(np.array([[0, 0.10], [0.02, 0.10], [0.2, 0.008], [0.2, 0.018]])).tolist() == [1, 0, 1, 0]


def test_hash_groups_reflections_and_rigid_transforms():
    track = generate_track(5, "intermediate")
    reflected = replace(track, start=(0, 0, 0), sections=tuple(replace(s, angle=-s.angle if s.angle else 0.0) for s in track.sections))
    moved = replace(track, start=(0.2, 0.3, 1.1))
    assert track.geometry_hash == reflected.geometry_hash == moved.geometry_hash
    split_straight = TrackSpec("same", (Section("straight", length=0.5), Section("straight", length=0.5)))
    assert split_straight.geometry_hash == straight_track().geometry_hash


def test_invalid_geometry_is_rejected():
    crossing = TrackSpec("overlap", (Section("straight", length=0.8), Section("corner", angle=math.pi),
                                     Section("straight", length=0.8)))
    assert any("intersection" in error for error in validate_track(crossing))
    assert validate_track(replace(straight_track(), gaps=((0.3, 0.2),)))
    with pytest.raises(ValueError, match="exhausted"):
        generate_track(0, config=GeneratorConfig(arena_size=0.35, max_attempts=10))


def test_frozen_suite_and_training_exclude_held_outs(tmp_path):
    path = create_suite(tmp_path / "suite", validation_count=3, test_count=3)
    manifest = read_suite(path)
    hashes = {entry["geometry_hash"] for split in ("validation", "test") for entry in manifest["splits"][split]}
    fixed = load_split(path, "training")[0]
    seen = set()
    first = training_batch(0, "multi", 0, 500, fixed, hashes, seen, GeneratorConfig())
    second = training_batch(0, "multi", 1, 500, fixed, hashes, seen, GeneratorConfig())
    assert len({t.geometry_hash for t in first + second}) == 8
    assert not hashes & {t.geometry_hash for t in first + second}
    assert all(t == fixed for t in training_batch(0, "single", 0, 500, fixed, hashes, seen, GeneratorConfig()))
    target = path.parent / manifest["splits"]["validation"][0]["path"]
    track = TrackSpec.load(target)
    replace(track, gaps=((0.3, 0.35),)).save(target)
    with pytest.raises(ValueError, match="modified"):
        load_split(path, "validation")


def test_curriculum_boundaries_retain_easy_tracks():
    assert difficulty_batch("curriculum", 99, 500) == ("beginner",) * 4
    assert difficulty_batch("curriculum", 100, 500) == ("beginner", "beginner", "intermediate", "intermediate")
    assert difficulty_batch("curriculum", 300, 500) == difficulty_batch("multi", 0, 500)
    assert "beginner" in difficulty_batch("curriculum", 499, 500)


def test_editable_episode_count_keeps_reproducible_difficulty_mixture():
    assert difficulty_batch("multi", 0, 500, 8) == difficulty_batch("multi", 0, 500)*2
    small_batches = tuple(difficulty_batch("multi", generation, 500, 1)[0] for generation in range(4))
    assert small_batches == difficulty_batch("multi", 0, 500)
    fixed = straight_track()
    assert training_batch(0, "single", 0, 500, fixed, set(), set(), GeneratorConfig(), 2) == [fixed, fixed]
    with pytest.raises(ValueError, match="positive"):
        difficulty_batch("multi", 0, 500, 0)
