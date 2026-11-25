"""Objective invariants and invalid-measurement checks, without evolution."""
from copy import deepcopy
import math

import pytest

from neat_simple_study.fitness import VERSION, score_batch


def batch():
    return [
        {'track_hash': str(index), 'completed': True, 'progress': 1.,
         'command_jitter': .01, 'completion_time_s': 5.,
         'route_length_m': 4., 'cap_m_s': 1., 'error': None}
        for index in range(3)
    ]


def test_exact_completed_formula_and_auditable_components():
    rows = batch()
    rows[1].update(route_length_m=6., completion_time_s=10., command_jitter=.04)
    rows[2].update(route_length_m=2., completion_time_s=4., command_jitter=.09)
    result = score_batch(rows, smoothness_weight=10.)
    utility = (.8 / 2 + .6 / 3 + .5 / 4) / 3
    assert result['fitness_version'] == VERSION == 'completion-speed-rms-1.0'
    assert result['completed'] == result['courses'] == 3
    assert result['tier'] == 'completed'
    assert result['mean_progress'] == 1.
    assert result['speed_utility'] == pytest.approx(utility)
    assert result['fitness'] == pytest.approx(3 + utility / (1 + utility))
    assert result['mean_command_change_rms'] == pytest.approx(.2)
    assert result['smoothness_weight'] == 10.


def test_every_additional_completion_outranks_residual_changes():
    scores = []
    for count in range(4):
        rows = batch()
        for index, row in enumerate(rows):
            row.update(completed=index < count, progress=1. if index < count else .999999,
                       completion_time_s=1e100 if index < count else 0.)
        score = score_batch(rows)['fitness']
        assert count <= score < count + 1
        scores.append(score)
    assert scores == sorted(scores)
    assert all(a < b for a, b in zip(scores, scores[1:]))


def test_extreme_finite_speed_stays_inside_completion_band():
    rows = batch()
    for row in rows:
        row.update(completion_time_s=1e-100, command_jitter=0.)
    assert 3 <= score_batch(rows)['fitness'] < 4


def test_incomplete_batches_use_progress_and_ignore_speed_and_jitter():
    rows = batch()
    rows[0].update(completed=False, progress=.2, completion_time_s=.001)
    before = deepcopy(rows)
    result = score_batch(rows)
    progress = 2.2 / 3
    assert result['tier'] == 'incomplete'
    assert result['fitness'] == pytest.approx(2 + progress / (1 + progress))
    assert result['speed_utility'] == 0.
    assert rows == before
    rows[0]['progress'] = .3
    assert score_batch(rows)['fitness'] > result['fitness']
    rows[0]['progress'] = .2
    rows[0]['completion_time_s'] = None
    rows[1].update(completion_time_s=1000., command_jitter=1.)
    assert score_batch(rows)['fitness'] == result['fitness']


def test_completed_speed_and_jitter_have_expected_directions():
    normal = batch()
    faster, smoother = deepcopy(normal), deepcopy(normal)
    for row in faster:
        row['completion_time_s'] /= 2
    for row in smoother:
        row['command_jitter'] = 0.
    baseline = score_batch(normal)['fitness']
    assert score_batch(faster)['fitness'] > baseline
    assert score_batch(smoother)['fitness'] > baseline
    assert score_batch(smoother, smoothness_weight=0.)['fitness'] == score_batch(normal, smoothness_weight=0.)['fitness']


def test_reference_cannot_change_fitness_or_be_used_as_a_target():
    rows = batch()
    assert score_batch(rows) == score_batch(rows, [{'invalid': 'reference'}])
    assert score_batch(rows) == score_batch(rows, object())


@pytest.mark.parametrize('key,value', [
    ('completed', 1), ('completed', 'False'), ('progress', -.01), ('progress', 1.01),
    ('progress', math.nan), ('command_jitter', -.001), ('command_jitter', 1.001),
    ('command_jitter', math.inf), ('command_jitter', '0.1'),
    ('completion_time_s', 0.), ('completion_time_s', -1.), ('completion_time_s', math.inf),
    ('route_length_m', 0.), ('route_length_m', math.nan), ('cap_m_s', 0.),
    ('cap_m_s', math.inf), ('error', 'physics failed'), ('track_hash', ''),
])
def test_invalid_measurements_are_rejected(key, value):
    rows = batch()
    rows[0][key] = value
    with pytest.raises(ValueError):
        score_batch(rows)


def test_empty_duplicate_and_mixed_cap_batches_are_rejected():
    with pytest.raises(ValueError):
        score_batch([])
    rows = batch()
    rows[1]['track_hash'] = rows[0]['track_hash']
    with pytest.raises(ValueError):
        score_batch(rows)
    rows = batch()
    rows[1]['cap_m_s'] = .5
    with pytest.raises(ValueError):
        score_batch(rows)


@pytest.mark.parametrize('value', [-1., math.nan, math.inf, True, '1'])
def test_invalid_tradeoff_weight_is_rejected(value):
    with pytest.raises(ValueError):
        score_batch(batch(), smoothness_weight=value)


def test_jitter_roundoff_is_clamped_only_within_tiny_tolerance():
    rows, exact = batch(), batch()
    rows[0]['command_jitter'] = -1e-13
    exact[0]['command_jitter'] = 0.
    rows[1]['command_jitter'] = 1 + 1e-13
    exact[1]['command_jitter'] = 1.
    assert score_batch(rows) == score_batch(exact)


def test_nonfinite_failed_episode_time_is_rejected_even_without_speed_reward():
    rows = batch()
    rows[0].update(completed=False, completion_time_s=math.nan)
    with pytest.raises(ValueError):
        score_batch(rows)
