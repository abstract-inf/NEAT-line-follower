"""Adaptive training gates and reproducibility, without running evolution."""

from dataclasses import asdict, replace
import pickle

import numpy as np
import pytest

from line_follower.config import EpisodeConfig
from line_follower.curriculum import CurriculumConfig, CurriculumState, stage_batch, stage_geometry
from line_follower.fitness import aggregate_fitness
from line_follower.suites import training_batch
from line_follower.tracks import GeneratorConfig, generate_track, straight_track, validate_track


def test_advancement_uses_rolling_training_counts_and_has_no_demotions():
    config = CurriculumConfig(window=3, minimum_generations=4, maximum_generations=8)
    state = CurriculumState()
    for _ in range(3):
        assert state.observe({"beginner": [9, 10]}, config)["next_stage"] == 0
    assert state.observe({"beginner": [9, 10]}, config)["reason"] == "competence"
    assert state.stage == 1 and state.history == []
    for _ in range(4):
        assert state.observe({"intermediate": [0, 10]}, config)["next_stage"] == 1
    restored = pickle.loads(pickle.dumps(state))
    for _ in range(4):
        assert state.observe({"intermediate": [0, 10]}, config) == restored.observe({"intermediate": [0, 10]}, config)
    assert state.stage == 2


def test_plateau_transition_is_distinguished_from_mastery_and_final_stage_is_bounded():
    config = CurriculumConfig(window=2, minimum_generations=2, maximum_generations=3)
    state = CurriculumState()
    for stage in range(4):
        for age in range(3):
            record = state.observe({}, config)
        assert record["reason"] == "plateau_exposure" and state.stage == stage+1
    for _ in range(20):
        assert state.observe({}, config)["reason"] == "final_stage"
    assert state.stage == 4


def test_ten_tracks_are_distinct_shared_reproducible_and_isolated_across_generations():
    config = GeneratorConfig(generator_version="2.0", max_attempts=1600)
    state = CurriculumState(stage=1)
    kwargs = dict(seed=2, strategy="adaptive", generations=200, fixed_track=straight_track(),
                  forbidden_hashes=set(), config=config, episodes_per_genome=10,
                  track_profile="progressive", curriculum_state=state)
    first = training_batch(generation=100, seen_hashes=set(), **kwargs)
    assert first == training_batch(generation=100, seen_hashes=set(), **kwargs)
    hashes = {t.geometry_hash for t in first}
    assert len(hashes) == 10
    second = training_batch(generation=101, seen_hashes=hashes.copy(), **kwargs)
    assert not hashes & {t.geometry_hash for t in second}
    assert [t.difficulty for t in first] == list(stage_batch(1, 10, 100))
    assert all(t.generator_version == "2.0" for t in first)
    assert all(not validate_track(t, stage_geometry(config, 1, t.difficulty)) for t in first)


def test_length_and_complexity_can_be_configured_independently_in_four_meter_arena():
    base = GeneratorConfig(generator_version="2.0", max_attempts=1600)
    early, late = stage_geometry(base, 1, "advanced"), stage_geometry(base, 4, "advanced")
    assert early.target_motifs == 6 and late.target_motifs == 12
    for seed in (1000, 1001):
        track = generate_track(seed, "advanced", late)
        assert 6 <= track.route.length <= 10 and track.arena_size == 4
        assert not validate_track(track, late)
    custom = replace(base, target_motifs=6, route_length_min=2.8, route_length_max=5.5)
    generated = generate_track(1000, "advanced", stage_geometry(custom, 4, "advanced"))
    assert 2.8 <= generated.route.length <= 5.5
    assert stage_geometry(base, 2, "advanced", CurriculumConfig(length_scale=1.1)).target_motifs == 8
    assert stage_geometry(base, 2, "advanced", CurriculumConfig(motif_scale=1.2)).route_length_max == 6.8


def test_optional_cvar_reduces_one_outlier_influence_but_does_not_hide_failures():
    scores = [2.0]*9+[-1.0]
    assert aggregate_fitness(scores) == pytest.approx(.35)
    assert aggregate_fitness(scores, "mean_cvar", .3) == pytest.approx(1.35)
    assert aggregate_fitness([2]*10, "mean_cvar", .3) > aggregate_fitness(scores, "mean_cvar", .3)
    assert aggregate_fitness(scores, "mean_cvar", .1) == aggregate_fitness(scores)
    assert aggregate_fitness([0]*10, "mean_cvar", .3) == 0
    assert aggregate_fitness([0, 1, 2, 3], "mean_cvar", .4) == pytest.approx(.9375)
    with pytest.raises(ValueError):
        aggregate_fitness(scores, "unknown")


def test_route_based_time_limits_are_bounded_without_changing_stall_rules():
    config = EpisodeConfig(time_reference_speed=.10, maximum_time_limit=120)
    assert [config.time_limit(length, .35) for length in (1, 5, 9, 15)] == [45, 50, 90, 120]
    assert config.stall_seconds == 5 and config.departure_seconds == 1
    assert EpisodeConfig().time_limit(20, .3) == 200  # Original formula stays available.


def test_version_two_suite_keeps_legacy_geometry_protected(tmp_path):
    from line_follower.suites import create_suite, load_split, read_suite
    old = create_suite(tmp_path / "old", validation_count=3, test_count=3)
    original = old.read_bytes()
    manifest = read_suite(old)
    protected = {e["geometry_hash"] for split in ("validation", "test") for e in manifest["splits"][split]}
    new = create_suite(tmp_path / "new", config=GeneratorConfig(generator_version="2.0", max_attempts=1600),
                       validation_count=3, test_count=3, protected_hashes=protected)
    revised = read_suite(new)
    assert revised["suite_version"] == 2 and revised["generator_version"] == "2.0"
    assert not protected & {e["geometry_hash"] for entries in revised["splits"].values() for e in entries}
    assert old.read_bytes() == original
    validation = load_split(new, "validation")
    assert 6 <= next(t for t in validation if t.difficulty == "advanced").route.length <= 10


def test_diagnostics_keep_failures_and_do_not_claim_feature_causation():
    from types import SimpleNamespace
    from line_follower.diagnostics import episode_summaries
    common = dict(features="zigzag;corner", difficulty="advanced", route_length_m=7.0,
                  time_limit_s=70, average_forward_speed_m_s=.12, lateral_rmse_m=.02)
    results = [SimpleNamespace(**common, completed=True, completion_time_s=55, termination="completed", progress=1),
               SimpleNamespace(**common, completed=False, completion_time_s=None, termination="off_route", progress=.4)]
    rows = episode_summaries(100, "training_population", results)
    feature = next(r for r in rows if r["category"] == "contains_feature" and r["value"] == "zigzag")
    assert feature["completion_rate"] == .5 and feature["fail_off_route"] == 1
    assert feature["mean_progress_before_failure"] == .4
