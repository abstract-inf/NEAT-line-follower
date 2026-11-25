"""Completion priority with one declared route-speed/command-change tradeoff."""
import math
from collections.abc import Mapping
from numbers import Real
from statistics import mean


VERSION = 'completion-speed-rms-1.0'
JITTER_TOLERANCE = 1e-12


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite number.')
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'{label} must be finite.') from None
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        requirement = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{label} must be finite and {requirement}.')
    return value


def score_batch(episodes, reference=None, *, smoothness_weight=1.0):
    """Score one common course batch; reference is intentionally unused.

    Incomplete batches compete on mean progress within their completion band.
    Fully completed batches compete on cap-normalized route speed divided by
    ``1 + smoothness_weight * RMS(command change)``. RMS is per simulation
    step, so comparisons require the same timestep and command normalization.
    """
    weight = _number(smoothness_weight, 'smoothness_weight')
    rows = list(episodes)
    if not rows:
        raise ValueError('A nonempty course batch is required.')

    hashes = set()
    progresses, command_changes, speed_utilities = [], [], []
    completed = 0
    cap = None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError('Every episode must be a measurement mapping.')
        track_hash = row.get('track_hash')
        if not isinstance(track_hash, str) or not track_hash.strip() or track_hash in hashes:
            raise ValueError('Episodes require nonempty unique track hashes.')
        hashes.add(track_hash)
        if row.get('error'):
            raise ValueError(f'Episode {track_hash} contains an evaluation error.')
        success = row.get('completed')
        if not isinstance(success, bool):
            raise ValueError('completed must be a bool.')
        completed += success

        progress = _number(row.get('progress'), 'progress')
        if progress > 1:
            raise ValueError('progress must lie in [0, 1].')
        progresses.append(progress)
        jitter = row.get('command_jitter')
        # Permit only roundoff outside the mathematically normalized range.
        if isinstance(jitter, Real) and not isinstance(jitter, bool) and -JITTER_TOLERANCE <= jitter < 0:
            jitter = 0.
        jitter = _number(jitter, 'command_jitter')
        if jitter > 1 + JITTER_TOLERANCE:
            raise ValueError('command_jitter must lie in [0, 1].')
        rms = math.sqrt(min(1., jitter))
        command_changes.append(rms)

        row_cap = _number(row.get('cap_m_s'), 'cap_m_s', positive=True)
        if cap is not None and row_cap != cap:
            raise ValueError('All episodes must use the same cap_m_s.')
        cap = row_cap
        length = _number(row.get('route_length_m'), 'route_length_m', positive=True)
        elapsed = row.get('completion_time_s')
        if success:
            elapsed = _number(elapsed, 'completion_time_s', positive=True)
            utility = length / elapsed / cap / (1. + weight * rms)
            speed_utilities.append(_number(utility, 'normalized speed utility'))
        else:
            if elapsed is not None:
                _number(elapsed, 'completion_time_s')
            speed_utilities.append(0.)

    complete_batch = completed == len(rows)
    mean_progress = mean(progresses)
    speed_utility = mean(speed_utilities) if complete_batch else 0.
    utility = speed_utility if complete_batch else mean_progress
    fitness = completed + utility / (1. + utility)
    # Extreme finite utility can round the residual to one in floating point.
    fitness = min(fitness, math.nextafter(float(completed + 1), -math.inf))
    return {
        'fitness_version': VERSION,
        'fitness': fitness,
        'completed': completed,
        'courses': len(rows),
        'mean_progress': mean_progress,
        'speed_utility': speed_utility,
        'mean_command_change_rms': mean(command_changes),
        'smoothness_weight': weight,
        'tier': 'completed' if complete_batch else 'incomplete',
        'utility': utility,
        'cap_m_s': cap,
    }
