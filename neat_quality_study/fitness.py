"""Completion-first task fitness for stock NEAT; no controller-side corrections.

Each genome is evaluated on the same batch as a frozen reference. Completion
counts occupy disjoint score bands. Within fully completed batches, satisfying
quality constraints outranks all violations; acceptable candidates compete on
speed. Infeasible candidates retain graded fitness and are never discarded.
"""
import math
from statistics import mean


VERSION = 'completion-quality-speed-1.0'
LIMITS = {
    'straight_yaw_rms_rad_s': .15,
    'straight_steering_rms': .02,
    'geometric_rmse_floor_m': .005,
    'command_jitter_floor': .0005,
    'straight_speed_target_m_s': .95,
    'route_speed_reference_m_s': 1.,
}


def excess(value, limit):
    if not math.isfinite(value) or value < 0 or not math.isfinite(limit) or limit <= 0:
        raise ValueError('Quality measurements must be finite and nonnegative, with positive limits.')
    return max(0., value / limit - 1.) if value > limit + 1e-12 else 0.


def score_batch(episodes, reference):
    """Return a scalar fitness and its complete auditable decomposition.

    The 5 mm and 0.0005 floors apply only to training references, preventing a
    perfectly centered reference from imposing a numerically zero tolerance.
    Original held-out acceptance thresholds remain unchanged.
    """
    if not episodes or len(episodes) != len(reference):
        raise ValueError('Candidate and reference require the same nonempty course batch.')
    expected = {row['track_hash'] for row in reference}
    actual = {row['track_hash'] for row in episodes}
    if actual != expected or len(actual) != len(episodes):
        raise ValueError('Candidate/reference geometry must match exactly, without duplicates.')
    complete = sum(bool(row['completed']) for row in episodes)
    for row in episodes + reference:
        for key in ('geometry_lateral_rmse_m', 'command_jitter'):
            if row.get(key) is None or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f'Missing or invalid independent quality metric: {key}')
        if not math.isfinite(row['progress']) or not 0 <= row['progress'] <= 1:
            raise ValueError('Progress must be a finite fraction.')
    error_limit = max(LIMITS['geometric_rmse_floor_m'], mean(r['geometry_lateral_rmse_m'] for r in reference))
    jitter_limit = max(LIMITS['command_jitter_floor'], mean(r['command_jitter'] for r in reference))
    error = mean(r['geometry_lateral_rmse_m'] for r in episodes)
    jitter = mean(r['command_jitter'] for r in episodes)
    windows = [w for row in episodes for w in row['straight_windows']]
    observed = [w for w in windows if w['observed']]
    if not windows:
        raise ValueError('The quality-speed profile requires dedicated straight measurement windows.')
    for window in observed:
        for key in ('speed_m_s', 'yaw_rms_rad_s', 'steering_rms'):
            if window.get(key) is None or not math.isfinite(window[key]):
                raise ValueError(f'Invalid straight measurement: {key}')
    max_yaw = max((w['yaw_rms_rad_s'] for w in observed), default=0.)
    max_steering = max((w['steering_rms'] for w in observed), default=0.)
    missing = len(windows) - len(observed)
    violations = {
        'geometric_rmse': excess(error, error_limit),
        'command_jitter': excess(jitter, jitter_limit),
        'straight_yaw': excess(max_yaw, LIMITS['straight_yaw_rms_rad_s']),
        'straight_steering': excess(max_steering, LIMITS['straight_steering_rms']),
        'missing_straights': missing / len(windows),
    }
    violation = sum(violations.values())
    feasible = violation == 0.
    minimum_speed = min((max(0., w['speed_m_s']) if w['observed'] else 0. for w in windows))
    straight_utility = min(1., minimum_speed / LIMITS['straight_speed_target_m_s'])
    route_speeds = []
    for row in episodes:
        if row['completed']:
            elapsed = row.get('completion_time_s')
            if elapsed is None or elapsed <= 0 or not math.isfinite(elapsed):
                raise ValueError('Successful courses require a positive finite completion time.')
            route_speeds.append(min(1., max(0., row['route_length_m'] / elapsed / LIMITS['route_speed_reference_m_s'])))
        else:
            route_speeds.append(0.)
    speed_utility = .75 * straight_utility + .25 * mean(route_speeds)
    quality_fraction = 1. / (1. + violation)
    if complete < len(episodes):
        # One additional completion always outranks improvements within a band.
        fitness = 10. * complete + 9. * (.8 * mean(r['progress'] for r in episodes) + .2 * quality_fraction)
        tier = 'incomplete'
    elif not feasible:
        fitness = 10. * complete + 9. * quality_fraction
        tier = 'completed_quality_violation'
    else:
        fitness = 10. * complete + 10. + 9. * speed_utility
        tier = 'completed_quality_feasible'
    return {
        'fitness_version': VERSION, 'fitness': fitness, 'tier': tier,
        'completed': complete, 'courses': len(episodes),
        'quality_feasible': feasible, 'quality_violation': violation, 'violations': violations,
        'geometric_rmse_m': error, 'geometric_rmse_limit_m': error_limit,
        'command_jitter': jitter, 'command_jitter_limit': jitter_limit,
        'maximum_straight_yaw_rms_rad_s': max_yaw,
        'maximum_straight_steering_rms': max_steering,
        'missing_windows': missing, 'minimum_straight_speed_m_s': minimum_speed,
        'straight_speed_utility': straight_utility, 'mean_route_speed_utility': mean(route_speeds),
        'speed_utility': speed_utility,
    }
