"""Conditional, unselected physical-motion objective; no training is wired here.

Theta is yaw RMS times sampled duration, in radians: angular activity rather
than net heading or oscillation alone. Command-change RMS is diagnostic only.
"""
import math
from collections.abc import Mapping
from numbers import Real
from statistics import mean


VERSION = 'completion-straight-motion-1.0'


def _number(value, label, *, positive=False, minimum=0.):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite number.')
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'{label} must be finite.') from None
    if not math.isfinite(value) or value < minimum or (positive and value <= 0):
        raise ValueError(f'{label} must be finite and within its valid range.')
    return value


def score_batch(episodes, reference=None, *, smoothness_weight=1.0,
                control_hz=60., target_fraction=.95):
    """Prioritize completion, near-cap straight speed, then physical lap cost.

    The reference is intentionally unused. The target fraction is an application
    requirement, and the weight multiplies angular activity in radians. Neither
    choice is claimed validated by this prospective implementation.
    """
    weight = _number(smoothness_weight, 'smoothness_weight')
    hz = _number(control_hz, 'control_hz', positive=True)
    target = _number(target_fraction, 'target_fraction', positive=True)
    if target >= 1:
        raise ValueError('target_fraction must lie in (0, 1).')
    rows = list(episodes)
    if not rows:
        raise ValueError('A nonempty course batch is required.')
    hashes, progresses, command_changes, lap_costs = set(), [], [], []
    completed, cap = 0, None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError('Every episode must be a measurement mapping.')
        identity = row.get('track_hash')
        if not isinstance(identity, str) or not identity.strip() or identity in hashes:
            raise ValueError('Episodes require nonempty unique track hashes.')
        hashes.add(identity)
        if row.get('error') or not isinstance(row.get('completed'), bool):
            raise ValueError('Episodes require bool completion and no evaluation error.')
        completed += row['completed']
        progress = _number(row.get('progress'), 'progress')
        if progress > 1:
            raise ValueError('progress must lie in [0, 1].')
        progresses.append(progress)
        jitter = _number(row.get('command_jitter'), 'command_jitter', minimum=-1e-12)
        if jitter > 1 + 1e-12:
            raise ValueError('command_jitter must lie in [0, 1].')
        command_changes.append(math.sqrt(max(0., min(1., jitter))))
        row_cap = _number(row.get('cap_m_s'), 'cap_m_s', positive=True)
        if cap is not None and row_cap != cap:
            raise ValueError('All episodes must use the same cap_m_s.')
        cap = row_cap
        length = _number(row.get('route_length_m'), 'route_length_m', positive=True)
        elapsed = row.get('completion_time_s')
        if row['completed']:
            elapsed = _number(elapsed, 'completion_time_s', positive=True)
            lap_costs.append(_number(elapsed * cap / length, 'normalized lap cost', positive=True))
        elif elapsed is not None:
            _number(elapsed, 'completion_time_s')
    progress = mean(progresses)
    ratio = theta = lap_cost = adjusted_cost = None
    target_met = False
    utility, tier = progress / (1 + progress), 'incomplete'
    if completed == len(rows):
        windows = []
        for row in rows:
            if not isinstance(row.get('straight_windows'), list):
                raise ValueError('Completed episodes require a straight_windows measurement list.')
            windows.extend(row['straight_windows'])
        if not windows:
            raise ValueError('A completed batch requires at least one measured straight window.')
        fractions, activity = [], []
        for window in windows:
            if not isinstance(window, Mapping) or window.get('observed') is not True:
                raise ValueError('Every required straight window must be observed on a completed batch.')
            samples = window.get('samples')
            if type(samples) is not int or samples <= 0:
                raise ValueError('Straight-window samples must be a positive integer.')
            speed = _number(window.get('speed_m_s'), 'straight speed')
            yaw = _number(window.get('yaw_rms_rad_s'), 'straight yaw RMS')
            duration = _number(_number(samples, 'samples', positive=True) / hz, 'window duration', positive=True)
            fractions.append(_number(speed / cap, 'straight speed fraction'))
            activity.append(_number(yaw * duration, 'straight angular activity'))
        ratio, theta = min(fractions), max(activity)
        lap_cost = mean(lap_costs)
        adjusted_cost = _number(lap_cost * (1 + weight * theta), 'motion-adjusted lap cost', positive=True)
        target_met = ratio >= target
        boundary = target / (1 + target)
        utility = (boundary + (1 - boundary) / (1 + adjusted_cost) if target_met
                   else ratio / (1 + ratio))
        tier = 'completed_target_met' if target_met else 'completed_below_target'
    fitness = completed + utility
    if completed == len(rows):
        boundary_score = completed + target / (1 + target)
        fitness = (max(fitness, math.nextafter(boundary_score, math.inf)) if target_met else
                   max(float(completed), min(fitness, math.nextafter(boundary_score, -math.inf))))
    fitness = min(fitness, math.nextafter(float(completed + 1), -math.inf))
    return {'fitness_version': VERSION, 'fitness': fitness, 'completed': completed, 'courses': len(rows),
            'mean_progress': progress, 'minimum_straight_speed_fraction': ratio,
            'straight_target_met': target_met, 'theta_rad': theta, 'normalized_lap_cost': lap_cost,
            'motion_adjusted_lap_cost': adjusted_cost, 'smoothness_weight': weight,
            'mean_command_change_rms': mean(command_changes), 'control_hz': hz,
            'target_fraction': target, 'cap_m_s': cap, 'utility': utility, 'tier': tier}
