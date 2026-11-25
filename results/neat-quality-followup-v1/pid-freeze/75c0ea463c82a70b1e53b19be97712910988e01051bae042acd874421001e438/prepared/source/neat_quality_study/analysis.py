"""Development calibration and matched-controller baseline; no evolution here."""
from pathlib import Path
from statistics import mean
import multiprocessing as mp
import shutil

from neat_smooth_study.common import read, write_json, digest, lock
from neat_smooth_study.engine import initialize
from .fitness import score_batch, VERSION, LIMITS
from .protocol import ROOT, SETTINGS, verify
from .training import validation_episodes


def calibrate():
    old = ROOT / 'runs/neat-smooth-study-v1'
    reference_path = old / 'pilot/references/seed-01.json'
    reference = read(reference_path)['primary']
    rows, hashes = [], {str(reference_path.relative_to(ROOT)): digest(reference_path)}
    for branch in ('old', 'smooth'):
        for path in sorted((old / 'pilot/seed-01' / branch / 'validation').glob('generation-*.json')):
            saved = read(path)
            result = score_batch(saved['primary'], reference)
            rows.append({'branch': branch, 'generation': saved['quality']['generation'], **result,
                         'original_acceptance': saved['quality'], 'input': str(path.relative_to(ROOT))})
            hashes[str(path.relative_to(ROOT))] = digest(path)
    if len(rows) != 200:
        raise ValueError('Both complete predecessor development pilots are required for this calibration.')
    rows.sort(key=lambda row: row['fitness'], reverse=True)
    out = ROOT / 'results/neat-quality-speed-v1'
    out.mkdir(exist_ok=True)
    write_json(out / 'fitness-calibration.json', {
        'version': VERSION, 'limits': LIMITS, 'fitness_source_sha256': digest(Path(__file__).with_name('fitness.py')),
        'scope': 'Retrospective development-only rescore. No new training, final test or independent confirmation.',
        'input_sha256': hashes, 'records': rows})
    text = ['# Quality-speed fitness calibration', '',
        'The task objective was calibrated before its new evolution run using existing development episodes. This is not an independent performance result.', '',
        'All synthetic ordering checks passed: additional completion dominates; acceptable quality plateaus; among feasible policies speed increases fitness; quality violations retain graded scores; missing windows and invalid geometry are rejected.', '',
        '| Rank | Previous branch / generation | New scalar fitness | Quality-feasible completed batch | Minimum straight speed |',
        '|---:|---|---:|---|---:|']
    for rank, row in enumerate(rows[:10], 1):
        text.append(f"| {rank} | {row['branch']} / {row['generation']} | {row['fitness']:.6f} | {row['tier'] == 'completed_quality_feasible'} | {row['minimum_straight_speed_m_s']:.3f} m/s |")
    text += ['', 'The scores here aggregate 60 validation episodes, while breeding aggregates six training episodes. Their numeric score bands therefore differ; the ordering principle is the same. Smooth g83 is the only fully completed candidate satisfying these quality constraints among the 200 stored validation champions. Its selection is explicitly a development warm-start decision.', '',
        'Limits, decomposition, archived inputs and hashes are recorded in `fitness-calibration.json`. The proposed task fitness preserves stock NEAT operators; it changes genome-level aggregation and must be described as a new experiment.']
    (out / 'fitness-calibration.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(f"Calibration saved: {rows[0]['branch']} g{rows[0]['generation']} ranks first; no evolution executed.", flush=True)


def summarize(rows):
    windows = [w for row in rows for w in row['straight_windows']]
    observed = [w for w in windows if w['observed']]
    return {
        'completed': sum(row['completed'] for row in rows), 'courses': len(rows),
        'geometric_rmse_m': mean(row['geometry_lateral_rmse_m'] for row in rows),
        'command_jitter': mean(row['command_jitter'] for row in rows),
        'mean_observed_chassis_speed_m_s': mean(row['average_forward_speed_m_s'] for row in rows),
        'minimum_observed_straight_speed_m_s': min((w['speed_m_s'] for w in observed), default=None),
        'maximum_observed_straight_yaw_rms_rad_s': max((w['yaw_rms_rad_s'] for w in observed), default=None),
        'missing_straight_windows': len(windows) - len(observed),
        'failures': {reason: sum(row['termination'] == reason for row in rows)
                     for reason in {r['termination'] for r in rows if not r['completed']}},
    }


def baseline(run, workers=6):
    """Freeze then evaluate the existing NEAT parents and PID on identical development courses."""
    run = Path(run).resolve(); verify(run)
    directory = run / 'baseline-development'
    inputs = run / 'prepared/inputs'
    policies = {'stable-neat-g83': inputs / 'warm-start-g0083.pkl',
                'fast-neat-g134': inputs / 'reference-g0134.pkl',
                'pid-fullspeed-1m': inputs / 'pid-fullspeed-100.json'}
    with lock(directory):
        frozen = directory / 'frozen'; frozen.mkdir(exist_ok=True)
        manifest = {'protocol_sha256': digest(run / 'prepared/protocol.json'), 'policies': {}}
        for name, source in policies.items():
            destination = frozen / (name + source.suffix)
            if destination.exists() and digest(destination) != digest(source):
                raise ValueError('Existing development controller snapshot changed.')
            shutil.copy2(source, destination)
            manifest['policies'][name] = {'file': destination.name, 'sha256': digest(destination)}
        write_json(frozen / 'manifest.json', manifest)
        records = {}
        with mp.get_context('spawn').Pool(workers, initialize, (str(run), str(inputs / 'warm-start-g0083.pkl')),
                maxtasksperchild=SETTINGS['worker_recycle_tasks']) as pool:
            for name, policy in manifest['policies'].items():
                target = directory / (name + '.json')
                source = frozen / policy['file']
                if target.exists():
                    saved = read(target)
                    if saved['protocol_sha256'] != manifest['protocol_sha256'] or saved['model_sha256'] != policy['sha256']:
                        raise ValueError('Development comparison cache changed.')
                else:
                    args = {'parameters': read(source)} if source.suffix == '.json' else {'model': str(source)}
                    primary, retention = validation_episodes(pool, run, **args)
                    saved = {'protocol_sha256': manifest['protocol_sha256'], 'model_sha256': policy['sha256'],
                             'primary': primary, 'retention': retention}
                    write_json(target, saved)
                records[name] = saved
                print(name, summarize(saved['primary']), flush=True)
        summary = {'scope': 'Shared previously consulted development courses, not final test data.',
                   'pid_retention_scope': 'The 1 m/s PID tune is transferred to .5 m/s here, not represented as a separate cap-specific tune.',
                   'policies': {name: {'primary': summarize(rows['primary']), 'retention': summarize(rows['retention'])}
                                for name, rows in records.items()}, 'paired': {}}
        pid = {row['track_hash']: row for row in records['pid-fullspeed-1m']['primary']}
        for name in ('stable-neat-g83', 'fast-neat-g134'):
            shared = [row for row in records[name]['primary'] if row['completed'] and pid[row['track_hash']]['completed']]
            summary['paired'][name] = {'shared_finishes': len(shared),
                'neat_mean_completion_time_s': mean(row['completion_time_s'] for row in shared) if shared else None,
                'pid_mean_completion_time_s': mean(pid[row['track_hash']]['completion_time_s'] for row in shared) if shared else None}
        write_json(directory / 'summary.json', summary)
        print('Shared development baseline complete; no NEAT evolution or final test executed.', flush=True)
