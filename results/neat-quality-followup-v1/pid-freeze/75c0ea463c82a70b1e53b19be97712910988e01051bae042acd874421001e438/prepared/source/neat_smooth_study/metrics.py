"""Physical straight-window metrics, bounded reward and winner eligibility."""
import math
from statistics import mean

import numpy as np


def straight_windows(track):
    distance, windows = 0., []
    for section in track.sections:
        length = section.length if section.kind == 'straight' else abs(section.radius*section.angle) if section.kind == 'arc' else 0.
        if section.kind == 'straight' and length >= 5-1e-9:
            middle = distance+length/2
            windows.append((middle-1, middle+1))
        distance += length
    return windows


def window_metrics(samples, finished):
    if not finished or len(samples) < 21:
        return {'observed':False, 'samples':len(samples), 'speed_m_s':None, 'yaw_rms_rad_s':None, 'steering_rms':None}
    a = np.asarray(samples)
    return {'observed':True, 'samples':len(samples), 'speed_m_s':float(a[:,0].mean()),
            'yaw_rms_rad_s':float(np.sqrt(np.mean(a[:,1]**2))),
            'steering_rms':float(np.sqrt(np.mean(a[:,2]**2)))}


def cost(value, target):
    if not math.isfinite(value) or value < 0:
        raise ValueError('A normalized physical cost must be finite and nonnegative')
    return value*value/(value*value+target*target)


def added_costs(windows, route_rmse):
    # Empty means this course has no eligible straight, not an unobserved window.
    def window_cost(field, target):
        return mean(cost(w[field],target) if w['observed'] else 1. for w in windows) if windows else 0.
    deficit = mean(min(1.,max(0.,1-w['speed_m_s']/.95))**2 if w['observed'] else 1.
                   for w in windows) if windows else 0.
    return {'straight_yaw_term':-.15*window_cost('yaw_rms_rad_s',.15),
            'straight_steering_term':-.15*window_cost('steering_rms',.02),
            'tracking_term':-.10*cost(route_rmse,.02), 'straight_speed_deficit_term':-.10*deficit}


def quality(primary, retention, reference, low_reference):
    """Every generation's champion uses identical tracks and the same gates."""
    reasons = []
    def by_hash(rows):
        output={r['track_hash']:r for r in rows}
        if len(output)!=len(rows):raise ValueError('Duplicate validation geometry')
        return output
    ref, low = by_hash(reference), by_hash(low_reference)
    if set(by_hash(primary))!=set(ref) or set(by_hash(retention))!=set(low):
        raise ValueError('Validation and reference must contain identical geometry')
    for label, rows, baseline in (('primary',primary,ref),('retention',retention,low)):
        for group in ('mixed','speed','corner'):
            current=[r for r in rows if r['group']==group]
            if sum(r['completed'] for r in current)<sum(baseline[r['track_hash']]['completed'] for r in current):
                reasons.append(f'{label}_{group}_completion')
    shared=[r for r in primary if r['completed'] and ref[r['track_hash']]['completed']]
    if not shared:
        reasons.append('no_shared_finishes')
        ratio=None
    else:
        ratio=mean(r['completion_time_s'] for r in shared)/mean(ref[r['track_hash']]['completion_time_s'] for r in shared)
        if ratio>1.05+1e-12:reasons.append('paired_time')
        for field in ('geometry_lateral_rmse_m','command_jitter'):
            if any(r.get(field) is None or not math.isfinite(r[field]) for r in shared):
                raise ValueError(f'Missing validation metric: {field}')
            if mean(r[field] for r in shared)>mean(ref[r['track_hash']][field] for r in shared)+1e-12:
                reasons.append(field)
    windows=[w for r in primary for w in r['straight_windows']]
    if not windows:reasons.append('no_required_straights')
    for w in windows:
        if not w['observed']:reasons.append('unfinished_straight');continue
        if w['speed_m_s']<.95-1e-12:reasons.append('straight_speed')
        if w['yaw_rms_rad_s']>.15+1e-12:reasons.append('straight_yaw')
        if w['steering_rms']>.02+1e-12:reasons.append('straight_steering')
    observed=[w for w in windows if w['observed']]
    return {'eligible':not reasons, 'reasons':sorted(set(reasons)), 'completed':sum(r['completed'] for r in primary),
            'paired_time_ratio':ratio, 'shared_finishes':len(shared),
            'straight_yaw_rms':mean(w['yaw_rms_rad_s'] for w in observed) if observed else None,
            'command_jitter':mean(r['command_jitter'] for r in primary),
            'geometric_rmse_m':mean(r['geometry_lateral_rmse_m'] for r in shared) if shared else None}


def select(records):
    candidates=[r for r in records if r['eligible']]
    if not candidates:return None
    maximum=max(r['completed'] for r in candidates)
    candidates=[r for r in candidates if r['completed']==maximum]
    fastest=min(r['paired_time_ratio'] for r in candidates)
    candidates=[r for r in candidates if r['paired_time_ratio']<=1.02*fastest+1e-12]
    return min(candidates,key=lambda r:(r['straight_yaw_rms'],r['command_jitter'],r['geometric_rmse_m'],
                                        r['paired_time_ratio'],r.get('branch','old')!='old',r.get('generation',0)))
