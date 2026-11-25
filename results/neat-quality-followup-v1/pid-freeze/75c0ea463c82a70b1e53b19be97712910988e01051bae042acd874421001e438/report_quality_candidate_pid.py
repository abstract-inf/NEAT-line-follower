"""Compare the saved report's named development candidate; never runs physics or evolution."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import pickle
from statistics import mean
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
COLORS = {'mixed': '#007f86', 'speed': '#c45a28', 'corner': '#7756a0'}
LABELS = {'pid': 'Full-speed filtered PID', 'fast': 'Fast NEAT g134'}
FILES = {'training_report': 'training-report.json', 'candidate': 'candidate-cache.json',
         'pid': 'pid-episodes.json', 'fast': 'fast-parent-episodes.json',
         'candidate_model': 'candidate-model.pkl', 'pid_model': 'pid-parameters.json',
         'fast_model': 'fast-parent-model.pkl', 'baseline_manifest': 'baseline-manifest.json',
         'protocol': 'protocol.json', 'suite': 'suite.json', 'report_source': 'report-source.py'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def load_inputs(run, report_path):
    blobs, paths = {}, {}

    def capture(role, path, expected=None):
        path = path.resolve()
        data = path.read_bytes()
        if expected is not None and sha(data) != expected:
            raise ValueError(f'Saved input hash differs: {path}')
        blobs[role], paths[role] = data, str(path)
        return json.loads(data) if path.suffix == '.json' else data

    report = capture('training_report', report_path)
    gate = report['fastest_speed_only_nonqualifier']
    check_candidate(gate)
    directory = run / 'development/seed-01'
    cache = directory / 'validation-cache' / (gate['genotype_sha256'] + '.json')
    capture('candidate', cache, report['input_sha256'][str(cache.resolve())])
    for role, name in (('pid', 'pid-fullspeed-1m'), ('fast', 'fast-neat-g134')):
        path = run / 'baseline-development' / (name + '.json')
        capture(role, path, report['input_sha256'][str(path.resolve())])
    manifest = capture('baseline_manifest', run / 'baseline-development/frozen/manifest.json')
    for role, name in (('pid', 'pid-fullspeed-1m'), ('fast', 'fast-neat-g134')):
        model = manifest['policies'][name]
        capture(role + '_model', run / 'baseline-development/frozen' / model['file'], model['sha256'])
    capture('candidate_model', directory / gate['model_file'])
    plan = capture('protocol', run / 'prepared/protocol.json', report['protocol_sha256'])
    capture('suite', run / 'prepared/suite/suite.json', plan['prepared_files']['suite/suite.json'])
    capture('report_source', Path(__file__))
    return blobs, paths


def check_candidate(gate):
    if not gate or not gate.get('genotype_sha256') or 'generation' not in gate or 'candidate_rank' not in gate:
        raise ValueError('The consumed training report must explicitly name a candidate and its identity.')
    if gate['eligible'] or gate['reasons'] != ['straight_speed']:
        raise ValueError('Expected the unchanged speed-only nonqualifier.')


def snapshot_inputs(blobs, paths, output):
    manifest = {'format': 1, 'inputs': {role: {'file': FILES[role], 'sha256': sha(data),
                'original_path': paths[role]} for role, data in sorted(blobs.items())}}
    manifest_bytes = encode(manifest)
    parent = output / 'snapshots'
    parent.mkdir(parents=True, exist_ok=True)
    snapshot = parent / sha(manifest_bytes)
    if snapshot.exists():
        restored, _ = replay_inputs(snapshot)
        if restored != blobs or (snapshot / 'manifest.json').read_bytes() != manifest_bytes:
            raise ValueError('Existing immutable snapshot differs.')
    else:
        with tempfile.TemporaryDirectory(prefix='staging-', dir=parent) as temporary:
            staging = Path(temporary)
            for role, data in blobs.items():
                (staging / FILES[role]).write_bytes(data)
            (staging / 'manifest.json').write_bytes(manifest_bytes)
            staging.rename(snapshot)
    return snapshot


def replay_inputs(snapshot):
    manifest_bytes = (snapshot / 'manifest.json').read_bytes()
    if snapshot.name != sha(manifest_bytes):
        raise ValueError('Snapshot address does not match its manifest.')
    manifest = json.loads(manifest_bytes)
    blobs, paths = {}, {}
    for role, entry in manifest['inputs'].items():
        if entry['file'] != FILES[role]:
            raise ValueError('Unexpected snapshot filename.')
        data = (snapshot / entry['file']).read_bytes()
        if sha(data) != entry['sha256']:
            raise ValueError(f'Snapshot input changed: {role}')
        blobs[role], paths[role] = data, entry['original_path']
    if set(blobs) != set(FILES):
        raise ValueError('Snapshot is incomplete.')
    return blobs, paths


def validated_rows(blobs):
    saved = {role: json.loads(data) for role, data in blobs.items() if FILES[role].endswith('.json')}
    gate = saved['training_report']['fastest_speed_only_nonqualifier']
    check_candidate(gate)
    protocol_hash = sha(blobs['protocol'])
    if protocol_hash != saved['training_report']['protocol_sha256']:
        raise ValueError('Report protocol fingerprint differs.')
    manifest = saved['baseline_manifest']
    if manifest['protocol_sha256'] != protocol_hash:
        raise ValueError('Baseline protocol differs.')
    models = {}
    for role, expected_generation in (('candidate', gate['generation']), ('fast', 134)):
        artifact = pickle.loads(blobs[role + '_model'])
        metadata = artifact['metadata']
        if metadata['generation'] != expected_generation:
            raise ValueError('Unexpected saved model generation.')
        models[role] = {'sha256': sha(blobs[role + '_model']), 'generation': metadata['generation'],
                        'controller_semantics': metadata.get('controller_semantics')}
        if role == 'candidate':
            clone = deepcopy(artifact['genome']); clone.key, clone.fitness = 0, None
            if (sha(pickle.dumps(clone, protocol=4)) != gate['genotype_sha256'] or
                    metadata['genotype_sha256'] != gate['genotype_sha256'] or metadata['protocol_sha256'] != protocol_hash):
                raise ValueError('Candidate model, genotype and protocol differ.')
    if saved['pid_model']['base'] != 1.:
        raise ValueError('PID is not the frozen full-speed tune.')
    models['pid'] = {'sha256': sha(blobs['pid_model']), 'parameters': saved['pid_model']}
    suite_entries = saved['suite']['splits']['validation']
    expected = {e['geometry_hash']: e['group'] for e in suite_entries}
    if len(suite_entries) != 60 or len(expected) != 60:
        raise ValueError('Expected 60 distinct primary development courses.')
    rows = {}
    for role in ('candidate', 'pid', 'fast'):
        record = saved[role]
        if record['protocol_sha256'] != protocol_hash:
            raise ValueError('Episode protocol differs.')
        if role == 'candidate':
            if record['genotype_sha256'] != gate['genotype_sha256']:
                raise ValueError('Candidate cache identity differs.')
        else:
            name = 'pid-fullspeed-1m' if role == 'pid' else 'fast-neat-g134'
            if record['model_sha256'] != models[role]['sha256'] or manifest['policies'][name]['sha256'] != models[role]['sha256']:
                raise ValueError('Frozen baseline model hash differs.')
        data = record['primary']
        by_hash = {r['track_hash']: r for r in data}
        if len(data) != 60 or set(by_hash) != set(expected):
            raise ValueError('Controller episode geometries differ from the identical 60-course suite.')
        for row in data:
            if row['cap_m_s'] != 1. or row['group'] != expected[row['track_hash']]:
                raise ValueError('Recorded cap or course group differs.')
            for field in ('geometry_lateral_rmse_m', 'command_jitter'):
                if not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError('Invalid independent observed metric.')
            if row['completed'] and (row['completion_time_s'] is None or not math.isfinite(row['completion_time_s']) or row['completion_time_s'] <= 0):
                raise ValueError('Invalid successful finish time.')
            for window in row['straight_windows']:
                if window['observed'] and any(window.get(field) is None or not math.isfinite(window[field])
                        for field in ('speed_m_s', 'yaw_rms_rad_s', 'steering_rms')):
                    raise ValueError('Invalid observed straight-window measurement.')
        rows[role] = by_hash
    if any(len(rows['candidate'][h]['straight_windows']) != len(rows[role][h]['straight_windows'])
           for h in expected for role in ('pid', 'fast')):
        raise ValueError('Required straight-window identities differ.')
    return gate, rows, models


def observed_summary(rows):
    rows = list(rows.values())
    windows = [w for r in rows for w in r['straight_windows']]
    observed = [w for w in windows if w['observed']]
    if not windows:
        raise ValueError('Required straight windows are absent.')
    return {'courses': len(rows), 'completed': sum(r['completed'] for r in rows),
            'failures': dict(Counter(r['termination'] for r in rows if not r['completed'])),
            'failure_rows': [r for r in rows if not r['completed']],
            'observed_mean_geometric_rmse_m': mean(r['geometry_lateral_rmse_m'] for r in rows),
            'observed_mean_command_jitter': mean(r['command_jitter'] for r in rows),
            'required_windows': len(windows), 'missing_windows': len(windows) - len(observed),
            'minimum_required_window_speed_m_s': min(w['speed_m_s'] if w['observed'] else 0. for w in windows),
            'maximum_observed_raw_yaw_rms_rad_s': max((w['yaw_rms_rad_s'] for w in observed), default=None),
            'maximum_observed_steering_rms': max((w['steering_rms'] for w in observed), default=None)}


def paired_summary(left, right):
    shared = sorted(h for h in left if left[h]['completed'] and right[h]['completed'])
    if not shared:
        raise ValueError('No shared successful courses.')
    metrics = {field: {'left': mean(left[h][field] for h in shared),
                       'right': mean(right[h][field] for h in shared)}
               for field in ('completion_time_s', 'geometry_lateral_rmse_m', 'command_jitter')}
    return {'shared_successes': len(shared), 'total_courses': len(left), 'track_hashes': shared, **metrics}


def plot_pairs(rows, output, labels):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), layout='constrained')
    for row_axes, other in zip(axes, ('pid', 'fast')):
        shared = [h for h in rows['candidate'] if rows['candidate'][h]['completed'] and rows[other][h]['completed']]
        for ax, field, unit, scale in zip(row_axes, ('completion_time_s', 'geometry_lateral_rmse_m'), ('finish time (s)', 'geometric RMSE (mm)'), (1, 1000)):
            for group, color in COLORS.items():
                subset = [h for h in shared if rows['candidate'][h]['group'] == group]
                ax.scatter([rows[other][h][field] * scale for h in subset],
                           [rows['candidate'][h][field] * scale for h in subset], color=color, label=group, alpha=.8)
            upper = max(max(rows['candidate'][h][field], rows[other][h][field]) * scale for h in shared) * 1.05
            ax.plot([0, upper], [0, upper], '--', color='#555', linewidth=1)
            ax.set(xlabel=f'{labels[other]} {unit}', ylabel=f"{labels['candidate']} {unit}",
                   title=f'{len(shared)}/60 shared successful courses', xlim=(0, upper), ylim=(0, upper))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"DEVELOPMENT / EXPLORATORY: saved {labels['candidate']}\nCandidate remains unqualified for the 0.95 m/s straight-speed requirement", fontsize=12)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'matched-course-comparison.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/neat-quality-speed-v1')
    parser.add_argument('--report', type=Path, default=ROOT / 'results/neat-quality-speed-v1/training/report.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/neat-quality-speed-v1/candidate-pid-development')
    parser.add_argument('--snapshot', type=Path, help='Replay only the captured immutable inputs, without reading the live run.')
    args = parser.parse_args()
    output = args.out.resolve()
    if args.snapshot:
        snapshot = args.snapshot.resolve()
        blobs, paths = replay_inputs(snapshot)
    else:
        blobs, paths = load_inputs(args.run.resolve(), args.report.resolve())
        snapshot = snapshot_inputs(blobs, paths, output)
    gate, rows, models = validated_rows(blobs)
    labels = {**LABELS, 'candidate': f"NEAT g{gate['generation']} rank {gate['candidate_rank']}"}
    summaries = {role: observed_summary(data) for role, data in rows.items()}
    if summaries['candidate']['minimum_required_window_speed_m_s'] >= .95:
        raise ValueError('The captured candidate no longer matches its disclosed speed nonqualification.')
    pairs = {f'{left}_vs_{right}': paired_summary(rows[left], rows[right])
             for left, right in (('candidate', 'pid'), ('fast', 'pid'), ('candidate', 'fast'))}
    output.mkdir(parents=True, exist_ok=True)
    plot_pairs(rows, output, labels)
    command = f"& .\\.venv\\Scripts\\python.exe '{snapshot / FILES['report_source']}' --snapshot '{snapshot}' --out '{output}'"
    result = {'scope': 'DEVELOPMENT / EXPLORATORY: repeatedly consulted primary validation; no fresh test or official winner promotion.',
              'candidate': gate, 'candidate_remains_unqualified': True, 'recorded_cap_m_s': 1.,
              'snapshot': str(snapshot), 'snapshot_manifest_sha256': snapshot.name,
              'report_source_sha256': sha(Path(__file__).read_bytes()),
              'captured_report_source_sha256': sha(blobs['report_source']),
              'inputs': {role: {'original_path': paths[role], 'sha256': sha(data)} for role, data in blobs.items()},
              'models': models, 'all_episode_observed_summaries': summaries, 'shared_success_comparisons': pairs,
              'replay_command': command}
    (output / 'report.json').write_bytes(encode(result))
    text = [f"# DEVELOPMENT / EXPLORATORY: saved {labels['candidate']} versus PID", '',
            f"This uses the same 60 saved primary development courses at a verified 1 m/s cap. The candidate is generation {gate['generation']}, rank {gate['candidate_rank']}, named by the consumed training report. It remains unqualified: its minimum required-straight speed is below 0.95 m/s. No official winner is promoted; these reused selection data are not a fresh test.", '',
            '| Controller | Completion | Failures | Observed mean episode geometric RMSE | Observed mean command jitter |',
            '|---|---:|---|---:|---:|']
    for role, item in summaries.items():
        text.append(f"| {labels[role]} | {item['completed']}/60 | {item['failures'] or 'none'} | {item['observed_mean_geometric_rmse_m']*1000:.3f} mm | {item['observed_mean_command_jitter']:.7f} |")
    text += ['', 'Observed means include all 60 episodes, including partial exposure on failures. They do not replace the matched shared-success comparisons below.', '',
             '| Left / right controller | Shared successful tracks | Mean finish time left / right | Mean episode geometric RMSE left / right | Mean jitter left / right |',
             '|---|---:|---:|---:|---:|']
    for key, pair in pairs.items():
        left, right = key.split('_vs_'); t, e, j = (pair[f] for f in ('completion_time_s', 'geometry_lateral_rmse_m', 'command_jitter'))
        text.append(f"| {labels[left]} / {labels[right]} | {pair['shared_successes']}/60 | {t['left']:.3f} / {t['right']:.3f} s | {e['left']*1000:.3f} / {e['right']*1000:.3f} mm | {j['left']:.7f} / {j['right']:.7f} |")
    text += ['', '| Controller | Required / missing windows | Minimum required-window speed | Maximum raw yaw RMS | Maximum steering RMS |',
             '|---|---:|---:|---:|---:|']
    for role, item in summaries.items():
        text.append(f"| {labels[role]} | {item['required_windows']} / {item['missing_windows']} | {item['minimum_required_window_speed_m_s']:.4f} m/s | {item['maximum_observed_raw_yaw_rms_rad_s']:.4f} rad/s | {item['maximum_observed_steering_rms']:.5f} |")
    text += ['', 'Missing windows contribute zero required-window speed; yaw/steering maxima describe observed windows. Acceptance limits are speed ≥0.95 m/s, raw yaw RMS ≤0.15 rad/s and steering RMS ≤0.02.', '',
             '![Matched finish times and geometric errors](matched-course-comparison.png)', '',
             'Each point is one shared successful course, colored by course group. Below the diagonal means a smaller candidate value. Failed courses remain in the all-episode summary and exact captured JSON, and are excluded from shared-success finish-time plots. Lower command jitter and lower geometric error measure different behavior; no blanket superiority claim follows.', '',
             f"Input snapshot: `{snapshot.name}`. Exact consumed report, episode caches, frozen model/parameter bytes, suite/protocol, manifest and reporting source are preserved under `snapshots/`. `report.json` records input/model/source hashes, failure rows and paired course identities. Existing snapshots are verified and never overwritten.", '',
             'Replay from the project root with the recorded Python/dependencies (this reads the snapshot only):', '', '```powershell', command, '```', '']
    (output / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'snapshot': str(snapshot), 'candidate_generation': gate['generation'],
                      'completion': {role: item['completed'] for role, item in summaries.items()},
                      'shared_successes': {key: pair['shared_successes'] for key, pair in pairs.items()}}, indent=2))


if __name__ == '__main__':
    main()
