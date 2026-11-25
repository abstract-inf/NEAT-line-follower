"""Describe every pilot arm on consulted validation; never promote a test winner."""
import csv
import gzip
import json
from pathlib import Path
import pickle
from statistics import mean

from neat_smooth_study.common import digest, read, write_json
from .fitness import score_batch


def summary(rows):
    windows = [window for row in rows for window in row['straight_windows']]
    seen = [window for window in windows if window['observed']]
    return {'completed': sum(row['completed'] for row in rows), 'courses': len(rows),
            'mean_progress': mean(row['progress'] for row in rows),
            'mean_successful_finish_time_s': mean(row['completion_time_s'] for row in rows if row['completed'])
                if any(row['completed'] for row in rows) else None,
            'mean_geometric_rmse_m': mean(row['geometry_lateral_rmse_m'] for row in rows),
            'mean_command_jitter': mean(row['command_jitter'] for row in rows),
            'mean_command_change_rms': mean(row['command_jitter']**.5 for row in rows),
            'minimum_required_straight_speed_m_s': min((max(0., window['speed_m_s']) if window['observed'] else 0.
                                                       for window in windows), default=None),
            'missing_straight_windows': len(windows) - len(seen),
            'maximum_observed_straight_yaw_rms_rad_s': max((window['yaw_rms_rad_s'] for window in seen), default=None)}


def report(run):
    run = Path(run).resolve()
    plan = read(run / 'prepared/protocol.json')
    protocol_hash = digest(run / 'prepared/protocol.json')
    destination = run / 'report'
    inputs, arms = {}, {}
    for arm, specification in plan['arms'].items():
        directory = run / 'development' / arm
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        if checkpoint.exists():
            with gzip.open(checkpoint, 'rb') as stream:
                next_generation = pickle.load(stream)['population'].generation
            inputs[str(checkpoint.relative_to(run))] = digest(checkpoint)
        else:
            next_generation = 0
        candidates = []
        for path in sorted((directory / 'validation').glob('generation-*.json')):
            if int(path.stem.split('-')[1]) >= next_generation:
                continue
            inputs[str(path.relative_to(run))] = digest(path)
            for candidate in read(path)['candidates']:
                cache_path = directory / 'validation-cache' / (candidate['genotype_sha256'] + '.json')
                cache = read(cache_path)
                if (cache['signature']['protocol_sha256'] != protocol_hash
                        or cache['signature']['arm'] != arm):
                    raise ValueError('Validation result belongs to a different frozen pilot.')
                inputs[str(cache_path.relative_to(run))] = digest(cache_path)
                model_path = directory / candidate['model_file']
                inputs[str(model_path.relative_to(run))] = digest(model_path)
                score = score_batch(cache['primary'], smoothness_weight=specification['smoothness_weight'])
                candidates.append({**candidate, 'development_validation_fitness': score['fitness'],
                    'primary': summary(cache['primary']), 'retention': summary(cache['retention'])})
        log = directory / 'generations.csv'
        generations = list(csv.DictReader(log.open(encoding='utf-8', newline=''))) if log.exists() else []
        generations = [row for row in generations if int(row['generation']) < next_generation]
        if log.exists():
            inputs[str(log.relative_to(run))] = digest(log)
        # Own-objective development selection is explicitly disclosed. Fitness
        # values from different coefficients are not treated as comparable units.
        selected = max(candidates, key=lambda row: (row['development_validation_fitness'], -row['generation'], -row['candidate_rank'])) if candidates else None
        arms[arm] = {'smoothness_weight': specification['smoothness_weight'],
                     'committed_generations': len(generations), 'complete': (directory / 'complete.json').exists(),
                     'selected_development_candidate': selected, 'candidates': candidates,
                     'actual_training_episodes': sum(int(row['training_episodes']) for row in generations),
                     'new_validation_episodes_logged': sum(int(row['newly_evaluated_validation_episodes']) for row in generations),
                     'cached_validation_slots_logged': sum(int(row['cached_validation_episodes']) for row in generations)}
    result = {'scope': 'Related warm-start development sensitivity; consulted validation, no final assessment or independent-seed confirmation.',
              'protocol_sha256': protocol_hash, 'arms': arms, 'input_hashes': inputs,
              'report_source_sha256': digest(Path(__file__)),
              'selection': 'Within each arm, highest new-objective fitness on 60 primary validation courses; ties earliest generation/rank. Report low-speed retention separately. Historical eligibility is diagnostic only.',
              'limitations': ['One existing ancestry; changes cannot establish from-scratch learning.',
                              'All-episode error/jitter means include partial failures; finish-time means have differing success sets.',
                              'Cached-slot log counts can omit pre-interruption work; not a full physical-compute audit.',
                              'No geometry/yaw gate enters the new task fitness; these remain measured quality outcomes.']}
    write_json(destination / 'report.json', result)
    lines = ['# Simple NEAT pilot — development results', '', result['scope'], '',
             result['selection'], '',
             '| Arm | Generations | Selected generation/rank | Primary completion | Retention | Min straight m/s | RMS command change | Geometric RMSE mm |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for arm, data in arms.items():
        candidate = data['selected_development_candidate']
        if candidate is None:
            lines.append(f'| {arm} | {data["committed_generations"]} | pending | — | — | — | — | — |')
            continue
        primary, retention = candidate['primary'], candidate['retention']
        speed = primary['minimum_required_straight_speed_m_s']
        speed_text = 'n/a' if speed is None else f'{speed:.4f}'
        lines.append(f'| {arm} | {data["committed_generations"]} | {candidate["generation"]}/{candidate["candidate_rank"]} | '
                     f'{primary["completed"]}/{primary["courses"]} | {retention["completed"]}/{retention["courses"]} | '
                     f'{speed_text} | {primary["mean_command_change_rms"]:.6f} | {primary["mean_geometric_rmse_m"]*1000:.3f} |')
    lines += ['', 'All arms and candidates, original failed-gate reasons and measured straight yaw remain in `report.json`.',
              'This report does not claim a final winner or five-seed repeatability. Fitness values under different lambda coefficients are not directly comparable.', '']
    (destination / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    selected = [(arm, data['selected_development_candidate']) for arm, data in arms.items()
                if data['selected_development_candidate'] is not None]
    if selected:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for arm, candidate in selected:
            primary = candidate['primary']
            speed = primary['minimum_required_straight_speed_m_s']
            for axis, field, scale in ((axes[0], 'mean_command_change_rms', 1), (axes[1], 'mean_geometric_rmse_m', 1000)):
                if speed is not None:
                    axis.scatter(primary[field]*scale, speed, label=f'{arm}: {primary["completed"]}/60', s=70)
        for axis, label in zip(axes, ('Mean RMS normalized command change / step', 'Mean geometric RMSE (mm)')):
            axis.set(xlabel=label, ylabel='Minimum required straight speed (m/s)')
            axis.axhline(.95, color='#777', linestyle='--', label='Declared straight-speed target')
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.suptitle('Consulted validation — one related development ancestry')
        for extension in ('png', 'svg'):
            figure.savefig(destination / f'speed-quality.{extension}', dpi=200)
        plt.close(figure)
    print(json.dumps({arm: {'generations': data['committed_generations'], 'complete': data['complete']}
                      for arm, data in arms.items()}))
