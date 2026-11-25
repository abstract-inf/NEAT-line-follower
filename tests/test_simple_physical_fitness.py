"""Conditional objective arithmetic and priority checks; no physics/evolution."""
from copy import deepcopy
import math
from pathlib import Path

import pytest

from neat_simple_study.physical_fitness import VERSION, score_batch


def batch(speed=.95, yaw=.1, cap=1.):
    return [{'track_hash': str(index), 'completed': True, 'progress': 1.,
             'command_jitter': .01, 'cap_m_s': cap, 'route_length_m': 4.,
             'completion_time_s': 5./cap, 'error': None,
             'straight_windows': [{'observed': True, 'samples': 120,
                                  'speed_m_s': speed*cap, 'yaw_rms_rad_s': yaw}] if index == 0 else []}
            for index in range(3)]


def test_exact_feasible_formula_and_duration_activity():
    rows = batch()
    rows[1]['completion_time_s'] = 10.
    result = score_batch(rows, smoothness_weight=2.)
    normalized_time = (5/4 + 10/4 + 5/4) / 3
    cost = normalized_time * (1 + 2*.2)
    boundary = .95/1.95
    assert result['fitness_version'] == VERSION == 'completion-straight-motion-1.0'
    assert result['fitness'] == pytest.approx(3 + boundary + (1-boundary)/(1+cost))
    assert result['theta_rad'] == pytest.approx(.2)
    assert result['normalized_lap_cost'] == pytest.approx(normalized_time)
    assert result['motion_adjusted_lap_cost'] == pytest.approx(cost)
    assert result['straight_target_met'] and result['tier'] == 'completed_target_met'


def test_completion_bands_and_incomplete_progress_direction():
    scores = []
    for count in range(4):
        rows = batch(yaw=1e100)
        for index in range(count, 3):
            rows[index].update(completed=False, progress=.999, completion_time_s=None,
                               straight_windows=[{'observed': False}])
        score = score_batch(rows)['fitness']
        assert count <= score < count+1
        scores.append(score)
    assert all(a < b for a, b in zip(scores, scores[1:]))
    rows = batch()
    rows[0].update(completed=False, progress=.1, completion_time_s=0.)
    before = score_batch(rows)['fitness']
    rows[0]['progress'] = .2
    assert score_batch(rows)['fitness'] > before


def test_target_priority_survives_extreme_noise_and_just_below_target_speed():
    below = batch(speed=math.nextafter(.95, 0.), yaw=0.)
    feasible = batch(yaw=1e200)
    assert score_batch(feasible)['fitness'] > score_batch(below)['fitness']
    for row in feasible:
        row['completion_time_s'] = 1e90
    assert score_batch(feasible)['fitness'] > score_batch(below)['fitness']


def test_feasible_physical_motion_and_lap_time_directions():
    normal = batch(yaw=.4)
    smoother, faster = deepcopy(normal), deepcopy(normal)
    smoother[0]['straight_windows'][0]['yaw_rms_rad_s'] = .1
    for row in faster:
        row['completion_time_s'] /= 2
    baseline = score_batch(normal)['fitness']
    assert score_batch(smoother)['fitness'] > baseline
    assert score_batch(faster)['fitness'] > baseline
    assert score_batch(normal, smoothness_weight=0.)['fitness'] == score_batch(smoother, smoothness_weight=0.)['fitness']


def test_below_target_only_straight_speed_changes_fitness():
    smooth, noisy = batch(speed=.8, yaw=0.), batch(speed=.8, yaw=100.)
    for row in noisy:
        row['completion_time_s'] *= 100
    assert score_batch(smooth)['fitness'] == score_batch(noisy)['fitness']
    assert score_batch(batch(speed=.9, yaw=100.))['fitness'] > score_batch(smooth)['fitness']


def test_command_jitter_and_reference_are_diagnostic_only():
    rows = batch()
    before = deepcopy(rows)
    original = score_batch(rows)
    assert rows == before
    for row in rows:
        row['command_jitter'] = 1.
    changed = score_batch(rows, object())
    assert original['fitness'] == changed['fitness']
    assert original['mean_command_change_rms'] < changed['mean_command_change_rms']


def test_theta_uses_maximum_duration_weighted_window_and_correct_control_rate():
    rows = batch(yaw=.2)
    rows[1]['straight_windows'] = [{'observed': True, 'samples': 600,
                                   'speed_m_s': .97, 'yaw_rms_rad_s': .1}]
    result = score_batch(rows)
    assert result['theta_rad'] == pytest.approx(1.)
    assert score_batch(rows, control_hz=120.)['theta_rad'] == pytest.approx(.5)
    assert result['minimum_straight_speed_fraction'] == .95


def test_target_fraction_scales_with_half_speed_cap():
    assert score_batch(batch(cap=.5))['straight_target_met']
    assert score_batch(batch(cap=.5))['fitness'] == score_batch(batch())['fitness']
    assert not score_batch(batch(speed=.949, cap=.5))['straight_target_met']
    rows = batch(speed=.8, cap=.5)
    assert score_batch(rows, target_fraction=.8)['straight_target_met']


@pytest.mark.parametrize('change', ['missing-list', 'empty-batch-windows', 'unobserved',
                                  'zero-samples', 'float-samples', 'bool-samples',
                                  'negative-yaw', 'nonfinite-yaw', 'negative-speed', 'nonfinite-speed'])
def test_invalid_successful_straight_measurements_raise(change):
    rows = batch()
    window = rows[0]['straight_windows'][0]
    if change == 'missing-list':
        del rows[0]['straight_windows']
    elif change == 'empty-batch-windows':
        rows[0]['straight_windows'] = []
    elif change == 'unobserved':
        window['observed'] = False
    else:
        key, value = {'zero-samples': ('samples', 0), 'float-samples': ('samples', 1.),
                      'bool-samples': ('samples', True), 'negative-yaw': ('yaw_rms_rad_s', -.1),
                      'nonfinite-yaw': ('yaw_rms_rad_s', math.nan),
                      'negative-speed': ('speed_m_s', -.1),
                      'nonfinite-speed': ('speed_m_s', math.inf)}[change]
        window[key] = value
    with pytest.raises(ValueError):
        score_batch(rows)


@pytest.mark.parametrize('key,value', [('progress', math.nan), ('progress', 1.1),
    ('completed', 1), ('error', 'physics failed'), ('cap_m_s', .5), ('route_length_m', 0.),
    ('completion_time_s', 0.), ('command_jitter', math.inf), ('command_jitter', 1.01),
    ('track_hash', '1')])
def test_invalid_batch_measurements_raise(key, value):
    rows = batch()
    rows[0][key] = value
    with pytest.raises(ValueError):
        score_batch(rows)


@pytest.mark.parametrize('kwargs', [{'smoothness_weight': -1.}, {'control_hz': 0.},
    {'control_hz': math.nan}, {'target_fraction': 0.}, {'target_fraction': 1.},
    {'smoothness_weight': True}])
def test_invalid_objective_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        score_batch(batch(), **kwargs)


def test_empty_batches_raise_and_standalone_exec_loading_works():
    with pytest.raises(ValueError):
        score_batch([])
    namespace = {}
    source = Path(__file__).resolve().parents[1] / 'neat_simple_study/physical_fitness.py'
    exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), namespace)
    assert namespace['score_batch'](batch()) == score_batch(batch())
