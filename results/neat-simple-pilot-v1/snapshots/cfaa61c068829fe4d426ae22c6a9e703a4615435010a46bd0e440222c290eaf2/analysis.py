"""Snapshot and compare the compact-reward pilot on consulted development data."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import pickle
from statistics import mean
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/neat-simple-pilot-v1'
OUTPUT = ROOT / 'results/neat-simple-pilot-v1'
PROTOTYPE = 'f520e5e3592fcd8563156ad1ab8a7def8554aa83c72f42997d8760410ec7c3fa'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def capture(run):
    blobs = {}
    def take(role, path):
        blobs[role] = path.read_bytes()
        return blobs[role]
    prepared = run / 'prepared'
    plan = json.loads(take('protocol', prepared / 'protocol.json'))
    if sha(blobs['protocol']) != (prepared / 'protocol.sha256').read_text().strip():
        raise ValueError('Pilot protocol fingerprint differs.')
    for role, relative in {'suite': 'suite/suite.json', 'scorer': 'source/neat_simple_study/fitness.py',
                           'native': 'inputs/reference-validation.json',
                           'native-model': 'inputs/reference-parent.pkl',
                           'prototype-model': 'inputs/accepted-prototype.pkl',
                           'pid-model': 'inputs/pid-100.json'}.items():
        data = take(role, prepared / relative)
        expected = plan['source_files'].get('neat_simple_study/fitness.py') if role == 'scorer' else plan['prepared_files'][relative]
        if sha(data) != expected:
            raise ValueError(f'Frozen report input differs: {role}')
    take('prototype', ROOT / f'runs/neat-joint-study-v1/development/joint/validation-cache/{PROTOTYPE}.json')
    pid_path = ROOT / 'results/neat-quality-followup-v1/pid-development-evaluation/episodes.json'
    pid = json.loads(take('pid', pid_path))
    pid_freeze = ROOT / Path(pid['freeze'].replace('\\', '/'))
    take('pid-freeze', pid_freeze / 'manifest.json')
    take('pid-frozen-config', pid_freeze / 'pid-fullspeed.json')
    take('preflight', run / 'preflight/physics-replay.json')
    for arm in plan['arms']:
        directory = run / 'development' / arm
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        if not checkpoint.exists():
            continue
        # The exact bytes used to establish the committed generation are also
        # the bytes hashed and retained, even if training commits a newer file.
        data = take(f'{arm}/checkpoint', checkpoint)
        next_generation = pickle.loads(gzip.decompress(data))['population'].generation
        for name in ('generations.csv', 'genome-fitness.csv'):
            path = directory / name
            if path.exists():
                take(f'{arm}/{name}', path)
        complete = directory / 'complete.json'
        if complete.exists():
            take(f'{arm}/complete', complete)
        for generation in range(next_generation):
            path = directory / 'validation' / f'generation-{generation:04d}.json'
            record = json.loads(take(f'{arm}/validation/{generation}', path))
            for candidate in record['candidates']:
                take(f'{arm}/model/{generation}-{candidate["candidate_rank"]}', directory / candidate['model_file'])
        # Count all retained unique caches, including completed physics from a
        # generation not yet committed. Do not call this all attempted work.
        for path in sorted((directory / 'validation-cache').glob('*.json')):
            take(f'{arm}/cache/{path.stem}', path)
    return blobs


def freeze(blobs, output):
    blobs = {**blobs, 'analysis-source': Path(__file__).read_bytes()}
    manifest = {'format': 1, 'dependencies': {name: importlib.metadata.version(name)
                for name in ('neat-python', 'numpy', 'matplotlib')},
                'inputs': {role: {'file': f'{index:04d}.bin', 'sha256': sha(data)}
                           for index, (role, data) in enumerate(sorted(blobs.items()))}}
    data = encoded(manifest)
    parent = output / 'snapshots'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / sha(data)
    if target.exists():
        restored, _ = replay(target)
        if restored != blobs:
            raise ValueError('Immutable snapshot differs.')
    else:
        with tempfile.TemporaryDirectory(prefix='staging-', dir=parent) as temporary:
            staging = Path(temporary)
            for role, entry in manifest['inputs'].items():
                (staging / entry['file']).write_bytes(blobs[role])
            (staging / 'analysis.py').write_bytes(blobs['analysis-source'])
            (staging / 'manifest.json').write_bytes(data)
            staging.rename(target)
    return target


def replay(snapshot):
    data = (snapshot / 'manifest.json').read_bytes()
    if sha(data) != snapshot.name:
        raise ValueError('Snapshot address differs from its manifest.')
    manifest = json.loads(data)
    for name, expected in manifest['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Snapshot dependency changed: {name}')
    blobs = {}
    for role, entry in manifest['inputs'].items():
        value = (snapshot / entry['file']).read_bytes()
        if sha(value) != entry['sha256']:
            raise ValueError(f'Snapshot bytes changed: {role}')
        blobs[role] = value
    if (snapshot / 'analysis.py').read_bytes() != blobs['analysis-source']:
        raise ValueError('Archived report source changed.')
    return blobs, manifest


def table(data, next_generation):
    rows = csv.DictReader(io.StringIO(data.decode('utf-8')))
    return [row for row in rows if row.get('generation', '').isdigit() and int(row['generation']) < next_generation]


def checked_rows(rows, expected, cap):
    by_hash = {row['track_hash']: row for row in rows}
    if len(by_hash) != len(rows) or set(by_hash) != set(expected):
        raise ValueError('Controllers did not use identical distinct development tracks.')
    for identity, row in by_hash.items():
        if row['cap_m_s'] != cap or row['group'] != expected[identity] or row['error']:
            raise ValueError('Episode cap, group or evaluation validity differs.')
        for field in ('geometry_lateral_rmse_m', 'command_jitter'):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f'Invalid quality measurement: {field}')
    return by_hash


def summary(rows):
    windows = [window for row in rows.values() for window in row['straight_windows']]
    observed = [window for window in windows if window['observed']]
    return {'courses': len(rows), 'completed': sum(row['completed'] for row in rows.values()),
            'failures': dict(Counter(row['termination'] for row in rows.values() if not row['completed'])),
            'completion_by_group': {group: {'completed': sum(row['completed'] for row in rows.values() if row['group'] == group),
                                          'courses': sum(row['group'] == group for row in rows.values())}
                                    for group in sorted({row['group'] for row in rows.values()})},
            'mean_geometric_rmse_m': mean(row['geometry_lateral_rmse_m'] for row in rows.values()),
            'mean_command_change_rms': mean(math.sqrt(row['command_jitter']) for row in rows.values()),
            'minimum_straight_speed_m_s': min((max(0., window['speed_m_s']) if window['observed'] else 0.
                                              for window in windows), default=None),
            'missing_windows': len(windows) - len(observed),
            'maximum_straight_yaw_rms_rad_s': max((window['yaw_rms_rad_s'] for window in observed), default=None)}


def paired(left, right):
    shared = sorted(identity for identity in left if left[identity]['completed'] and right[identity]['completed'])
    result = {'shared_successes': len(shared), 'total_courses': len(left), 'track_hashes': shared}
    if shared:
        result.update({field: {'left': mean(left[h][field] for h in shared), 'right': mean(right[h][field] for h in shared)}
                       for field in ('completion_time_s', 'geometry_lateral_rmse_m', 'command_jitter')})
    return result


def analyze(blobs, snapshot):
    plan = json.loads(blobs['protocol'])
    protocol_hash = sha(blobs['protocol'])
    suite = json.loads(blobs['suite'])['splits']
    primary = {row['geometry_hash']: row['group'] for row in suite['validation']}
    retention = {row['geometry_hash']: row['group'] for row in suite['retention']}
    if len(primary) != 60 or len(retention) != 20 or set(primary) & set(retention):
        raise ValueError('Unexpected development suite or split overlap.')
    scorer_namespace = {}
    exec(compile(blobs['scorer'], '<frozen compact fitness>', 'exec'), scorer_namespace)
    score_batch = scorer_namespace['score_batch']
    native, prototype, pid = (json.loads(blobs[name]) for name in ('native', 'prototype', 'pid'))
    if prototype['genotype_sha256'] != PROTOTYPE:
        raise ValueError('Accepted-prototype cache identity differs.')
    for role, cache in (('prototype', prototype),):
        artifact = pickle.loads(blobs[role + '-model'])
        genome = deepcopy(artifact['genome'])
        genome.key, genome.fitness = 0, None
        if sha(pickle.dumps(genome, protocol=4)) != cache['genotype_sha256']:
            raise ValueError('Prototype model and episode cache differ.')
    if native['parent_sha256'] != sha(blobs['native-model']):
        raise ValueError('Native reference model differs.')
    pid_manifest = json.loads(blobs['pid-freeze'])
    if (sha(blobs['pid-freeze']) != pid['freeze_sha256'] or sha(blobs['pid-model']) != pid['pid_sha256']
            or blobs['pid-model'] != blobs['pid-frozen-config']
            or sha(blobs['pid-model']) != pid_manifest['pid-fullspeed.json']
            or json.loads(blobs['pid-model'])['base'] != 1.):
        raise ValueError('PID results are not the same frozen full-base tune.')
    controls = {'native-g134': native['primary'], 'accepted-prototype': prototype['primary'], 'tuned-pid': pid['primary']}
    control_rows = {name: checked_rows(rows, primary, 1.) for name, rows in controls.items()}
    summaries = {name: summary(rows) for name, rows in control_rows.items()}
    arms = {}
    for arm, definition in plan['arms'].items():
        role = f'{arm}/checkpoint'
        if role not in blobs:
            arms[arm] = {'complete': False, 'committed_generations': 0, 'candidates': []}
            continue
        checkpoint = pickle.loads(gzip.decompress(blobs[role]))
        next_generation = checkpoint['population'].generation
        if checkpoint['signature']['protocol_sha256'] != protocol_hash or checkpoint['signature']['arm'] != arm:
            raise ValueError('Population checkpoint signature differs.')
        logs = table(blobs.get(f'{arm}/generations.csv', b'generation\n'), next_generation)
        if [int(row['generation']) for row in logs] != list(range(next_generation)):
            raise ValueError('Committed generation log is incomplete.')
        candidates, trajectories = [], []
        for generation in range(next_generation):
            gates = json.loads(blobs[f'{arm}/validation/{generation}'])['candidates']
            current = []
            for gate in gates:
                cache = json.loads(blobs[f'{arm}/cache/{gate["genotype_sha256"]}'])
                if (cache['signature']['protocol_sha256'] != protocol_hash or cache['signature']['arm'] != arm
                        or cache['genotype_sha256'] != gate['genotype_sha256']):
                    raise ValueError('Candidate cache identity differs.')
                artifact = pickle.loads(blobs[f'{arm}/model/{generation}-{gate["candidate_rank"]}'])
                metadata = artifact['metadata']
                genome = deepcopy(artifact['genome'])
                genome.key, genome.fitness = 0, None
                if (sha(pickle.dumps(genome, protocol=4)) != gate['genotype_sha256']
                        or metadata['protocol_sha256'] != protocol_hash
                        or metadata['fitness_version'] != plan['fitness_version']
                        or metadata['condition'] != 'simple-reward-development'):
                    raise ValueError('Candidate model provenance differs.')
                rows = checked_rows(cache['primary'], primary, 1.)
                low = checked_rows(cache['retention'], retention, .5)
                fitness = score_batch(list(rows.values()), smoothness_weight=definition['smoothness_weight'])['fitness']
                result = {'generation': generation, 'rank': gate['candidate_rank'],
                          'genotype_sha256': gate['genotype_sha256'], 'model_file': gate['model_file'],
                          'own_validation_fitness': fitness, 'primary': summary(rows), 'retention': summary(low),
                          'historical_gate_rejections': gate['reasons'],
                          'paired_vs_pid': paired(rows, control_rows['tuned-pid']),
                          'paired_vs_prototype': paired(rows, control_rows['accepted-prototype'])}
                candidates.append(result)
                current.append(result)
            trajectories.append(max(current, key=lambda row: (row['own_validation_fitness'], -row['rank'])))
        selected = max(candidates, key=lambda row: (row['own_validation_fitness'], -row['generation'], -row['rank'])) if candidates else None
        cache_roles = [name for name in blobs if name.startswith(arm + '/cache/')]
        retained = sum(len(json.loads(blobs[name])['primary']) + len(json.loads(blobs[name])['retention']) for name in cache_roles)
        arms[arm] = {'smoothness_weight': definition['smoothness_weight'],
                     'complete': next_generation == plan['settings']['generations'] and f'{arm}/complete' in blobs,
                     'committed_generations': next_generation, 'selected': selected, 'candidates': candidates,
                     'per_generation_selection': trajectories,
                     'cost': {'committed_training_episodes': sum(int(row['training_episodes']) for row in logs),
                              'logged_new_validation_episodes': sum(int(row['newly_evaluated_validation_episodes']) for row in logs),
                              'logged_cached_validation_slots': sum(int(row['cached_validation_episodes']) for row in logs),
                              'retained_unique_validation_episodes': retained}}
    return {'scope': 'Warm-start development sensitivity only. All validation data are consulted; one ancestry, no independent-seed or final-test claims.',
            'complete': all(arm['complete'] for arm in arms.values()), 'protocol_sha256': protocol_hash,
            'snapshot_sha256': snapshot.name, 'controls': summaries, 'arms': arms,
            'preflight_new_episodes': json.loads(blobs['preflight'])['episodes'],
            'selection': 'Highest own-coefficient fitness on 60 primary validation tracks; ties earliest generation/rank. This is development selection, not a new final winner.',
            'limitations': ['Own-coefficient fitness values are not comparable across coefficients.',
                            'All-episode quality means include partial failures; paired metrics use disclosed shared finishes.',
                            'RMS command change measures actuation commands per fixed step, not physical vibration.',
                            'Retained/logged episode counts cannot recover unknown discarded partial simulations.',
                            'No error/yaw requirement enters this compact breeding fitness.']}


def render(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_bytes(encoded(result))
    lines = ['# Compact NEAT reward — matched development comparison', '', result['scope'], '',
             '**' + ('Completed pilot.' if result['complete'] else 'Interim: pilot still running.') + '**', '',
             '| Arm | Generations | Primary / retention | Min straight m/s | Paired time vs PID | Shared finishes | RMS command change | RMSE mm |',
             '|---|---:|---|---:|---|---:|---:|---:|']
    for name, arm in result['arms'].items():
        selected = arm.get('selected')
        if not selected:
            lines.append(f'| {name} | 0 | pending | — | — | — | — | — |')
            continue
        p, low, pair = selected['primary'], selected['retention'], selected['paired_vs_pid']
        times = pair.get('completion_time_s')
        time_text = f'{times["left"]:.3f} / {times["right"]:.3f} s' if times else 'no shared finishes'
        lines.append(f'| {name} | {arm["committed_generations"]} | {p["completed"]}/60; {low["completed"]}/20 | '
                     f'{p["minimum_straight_speed_m_s"]:.4f} | {time_text} | {pair["shared_successes"]}/60 | '
                     f'{p["mean_command_change_rms"]:.5f} | {p["mean_geometric_rmse_m"]*1000:.2f} |')
    lines += ['', result['selection'], '',
              'Time comparisons use each pair\'s shared successful tracks; different rows can have different denominators. Error/RMS columns above are observed all-episode means. Full paired error/jitter, failures, physical yaw, per-group completion and all candidate records are in `report.json`.', '',
              'The frozen accepted prototype and tuned full-base PID are development controls, not independent training trials. No threshold is relaxed and no rejected historical model is promoted.', '',
              f'Immutable analysis snapshot: `snapshots/{result["snapshot_sha256"]}/`. Its `analysis.py` can replay all captured numerical inputs with `--snapshot <snapshot> --output <new-output>`.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'lambda-0': '#8d5a3b', 'lambda-1': '#167d8d', 'lambda-10': '#7856a0'}
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout='constrained')
    metrics = [('completed', 'Primary completion / 60', 1),
               ('minimum_straight_speed_m_s', 'Minimum required straight speed (m/s)', 1),
               ('mean_command_change_rms', 'Mean RMS command change / step', 1),
               ('mean_geometric_rmse_m', 'Mean geometric RMSE (mm)', 1000),
               ('maximum_straight_yaw_rms_rad_s', 'Maximum observed straight yaw RMS (rad/s)', 1)]
    for axis, (field, label, scale) in zip(axes.flat, metrics):
        for name, arm in result['arms'].items():
            rows = arm.get('per_generation_selection', [])
            axis.plot([row['generation'] for row in rows], [row['primary'][field]*scale for row in rows],
                      marker='.', color=colors[name], label=name)
        for control, style in (('accepted-prototype', '--'), ('tuned-pid', ':')):
            value = result['controls'][control][field]
            if value is not None:
                axis.axhline(value*scale, color='#555' if style == '--' else '#999', linestyle=style, label=control)
        if field == 'minimum_straight_speed_m_s':
            axis.axhline(.95, color='#c44343', linewidth=.8, label='Declared 0.95 target')
        axis.set(xlabel='Committed generation', ylabel=label)
        axis.grid(alpha=.2)
    axis = axes.flat[-1]
    for name, arm in result['arms'].items():
        rows = [row for row in arm.get('per_generation_selection', []) if row['paired_vs_pid']['shared_successes']]
        axis.plot([row['generation'] for row in rows],
                  [row['paired_vs_pid']['completion_time_s']['left']/row['paired_vs_pid']['completion_time_s']['right'] for row in rows],
                  marker='.', color=colors[name], label=name)
    axis.axhline(1., color='#999', linestyle=':')
    axis.set(xlabel='Committed generation', ylabel='Paired finish-time ratio to PID')
    axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    figure.suptitle(('Completed' if result['complete'] else 'Interim') + ' related development runs — no independent-seed uncertainty bands')
    for extension in ('png', 'svg'):
        figure.savefig(output / f'learning-and-quality.{extension}', dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve() if args.snapshot else freeze(capture(args.run.resolve()), args.output.resolve())
    blobs, _ = replay(snapshot)
    if Path(__file__).read_bytes() != blobs['analysis-source']:
        raise ValueError('Use the archived analysis.py to replay this snapshot.')
    result = analyze(blobs, snapshot)
    render(result, args.output.resolve())
    print(json.dumps({'complete': result['complete'], 'snapshot': str(snapshot),
                      'generations': {name: arm['committed_generations'] for name, arm in result['arms'].items()}}))


if __name__ == '__main__':
    main()
