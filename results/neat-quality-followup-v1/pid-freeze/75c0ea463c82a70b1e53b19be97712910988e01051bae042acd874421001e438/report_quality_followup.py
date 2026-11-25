"""Finished development report; optional one-time, 60-course frozen PID evaluation."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import io
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from report_quality_candidate_pid import sha, encode, observed_summary, paired_summary


ROOT = Path(__file__).resolve().parents[1]
FOLLOW = Path('runs/neat-quality-followup-v1')
STRICT = Path('runs/neat-quality-speed-v1')
OUT = Path('results/neat-quality-followup-v1')
PID_DATA = OUT / 'pid-development-evaluation'
PID_SHA = 'd6df8d4275b568583dcaf652a131f7c3f17f4d6923998828eb1322562c78b5cc'
STUDIES = {'strict': (STRICT, STRICT / 'development/seed-01'),
           'smooth': (FOLLOW, FOLLOW / 'development/smooth'), 'fast': (FOLLOW, FOLLOW / 'development/fast')}
COLORS = {'strict': '#555e68', 'smooth': '#007f86', 'fast': '#c45a28'}


def freeze_files(base, blobs):
    manifest = {name: sha(data) for name, data in sorted(blobs.items())}
    data = encode(manifest)
    target = base / sha(data)
    if target.exists():
        if (target / 'manifest.json').read_bytes() != data:
            raise ValueError('Existing immutable freeze differs.')
        for name, content in blobs.items():
            if (target / name).read_bytes() != content:
                raise ValueError(f'Existing frozen bytes differ: {name}')
    else:
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='staging-', dir=base) as temporary:
            staging = Path(temporary)
            for name, content in blobs.items():
                path = staging / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            (staging / 'manifest.json').write_bytes(data)
            staging.rename(target)
    return target


def evaluate_pid_once():
    """The attempt marker prevents accidentally spending a second episode budget."""
    result_path = ROOT / PID_DATA / 'episodes.json'
    if result_path.exists():
        return
    attempt = ROOT / PID_DATA / 'attempt.json'
    if attempt.exists():
        raise RuntimeError('A PID evaluation was already attempted; preserve partial work rather than rerunning the budget.')
    run = ROOT / FOLLOW
    plan_bytes = (run / 'prepared/protocol.json').read_bytes()
    plan = json.loads(plan_bytes)
    if sha(plan_bytes) != (run / 'prepared/protocol.sha256').read_text().strip():
        raise ValueError('Follow-up protocol signature differs.')
    if {name: importlib.metadata.version(name) for name in plan['dependencies']} != plan['dependencies']:
        raise ValueError('Frozen evaluator dependency versions differ.')
    frozen = {'prepared/protocol.json': plan_bytes}
    for name, expected in plan['source_files'].items():
        data = (run / 'prepared/source' / name).read_bytes()
        if sha(data) != expected:
            raise ValueError(f'Frozen evaluator source differs: {name}')
        frozen['prepared/source/' + name] = data
    suite_bytes = (run / 'prepared/suite/suite.json').read_bytes()
    if sha(suite_bytes) != plan['prepared_files']['suite/suite.json']:
        raise ValueError('Frozen suite differs.')
    frozen['prepared/suite/suite.json'] = suite_bytes
    entries = json.loads(suite_bytes)['splits']['validation']
    if len(entries) != 60 or len({e['geometry_hash'] for e in entries}) != 60:
        raise ValueError('Expected the identical 60 primary development courses.')
    for entry in entries:
        name = 'suite/' + entry['path']
        data = (run / 'prepared' / name).read_bytes()
        if sha(data) != plan['prepared_files'][name]:
            raise ValueError('Frozen validation course differs.')
        frozen['prepared/' + name] = data
    pid_run = ROOT / 'runs/pid-quality-tuning-v1'
    config = (pid_run / 'cap-100/pid-fullspeed.json').read_bytes()
    selection = (pid_run / 'cap-100/selection-freeze.json').read_bytes()
    pid_protocol = (pid_run / 'prepared/protocol.json').read_bytes()
    selected = json.loads(selection)
    if (sha(config) != PID_SHA or selected['sha256'] != PID_SHA or not selected['eligible'] or
            selected['signature']['cap'] != 1. or selected['signature']['protocol_sha256'] != sha(pid_protocol)):
        raise ValueError('Unexpected frozen retuned PID selection.')
    frozen.update({'pid-fullspeed.json': config, 'pid-selection.json': selection, 'pid-protocol.json': pid_protocol,
                   'report_quality_followup.py': Path(__file__).read_bytes(),
                   'report_quality_candidate_pid.py': (Path(__file__).parent / 'report_quality_candidate_pid.py').read_bytes(),
                   'runtime.json': encode({'python': sys.version, 'dependencies': plan['dependencies']}),
                   'evaluation.json': encode({'scope': 'Previously consulted primary development courses; no fresh test.',
                       'episodes': 60, 'workers': 2, 'cap_m_s': 1., 'pid_sha256': PID_SHA})})
    freeze = freeze_files(ROOT / OUT / 'pid-freeze', frozen)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with attempt.open('x', encoding='utf-8') as stream:
        json.dump({'freeze': str(freeze.relative_to(ROOT)), 'episodes_authorized': 60, 'workers': 2}, stream)
    sys.path.insert(0, str(freeze / 'prepared/source'))
    from neat_smooth_study.engine import evaluate_tasks
    tasks = [{'parameters': json.loads(config), 'cap': 1., 'detailed': True,
              'paths': [str(freeze / 'prepared/suite' / entry['path']) for entry in entries[start:start+4]],
              'groups': {entry['geometry_hash']: entry['group'] for entry in entries[start:start+4]}}
             for start in range(0, 60, 4)]
    rows = [row for chunk in evaluate_tasks(freeze, tasks, workers=2) for row in chunk]
    if len(rows) != 60 or {r['track_hash'] for r in rows} != {e['geometry_hash'] for e in entries} or any(r['cap_m_s'] != 1. for r in rows):
        raise ValueError('PID episode identities/caps differ.')
    result = {'scope': 'DEVELOPMENT ONLY: 60 consulted primary courses, no fresh test.',
              'freeze': str(freeze.relative_to(ROOT)), 'freeze_sha256': freeze.name,
              'pid_sha256': PID_SHA, 'new_episodes': 60, 'workers': 2, 'primary': rows}
    with result_path.open('xb') as stream:
        stream.write(encode(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-pid', action='store_true', help='Spend the one authorized 60-episode PID budget only if not attempted.')
    parser.add_argument('--snapshot', type=Path, help='Replay captured report inputs without physics or live files.')
    parser.add_argument('--out', type=Path, default=ROOT / OUT)
    args = parser.parse_args()
    if args.snapshot and args.evaluate_pid:
        raise ValueError('Snapshot replay never evaluates physics.')
    if args.evaluate_pid:
        evaluate_pid_once()
    captured = {}
    replay = None
    if args.snapshot:
        snapshot = args.snapshot.resolve()
        manifest_bytes = (snapshot / 'manifest.json').read_bytes()
        if sha(manifest_bytes) != snapshot.name:
            raise ValueError('Report snapshot address differs.')
        replay = json.loads(manifest_bytes)

    def content(path):
        name = Path(path).as_posix()
        if name not in captured:
            if replay is None:
                data = (ROOT / path).read_bytes()
            else:
                entry = replay[name]; data = (snapshot / entry['file']).read_bytes()
                if sha(data) != entry['sha256']:
                    raise ValueError(f'Report snapshot changed: {name}')
            captured[name] = data
        return captured[name]

    def read(path):
        return json.loads(content(path))

    def table(path):
        return list(csv.DictReader(io.StringIO(content(path).decode('utf-8'))))

    manifest = read(FOLLOW / 'prepared/suite/suite.json')['splits']
    expected_primary = {e['geometry_hash']: e['group'] for e in manifest['validation']}
    expected_retention = {e['geometry_hash']: e['group'] for e in manifest['retention']}
    studies, records, fixed = {}, {}, {}
    for name, (run, directory) in STUDIES.items():
        plan_bytes = content(run / 'prepared/protocol.json')
        plan = json.loads(plan_bytes)
        completed = read(directory / 'complete.json')
        generations = table(directory / 'generations.csv')
        fitness = table(directory / 'genome-fitness.csv')
        training = table(directory / 'training-episodes.csv')
        if (completed['generations'] != 50 or completed['signature']['protocol_sha256'] != sha(plan_bytes) or
                [int(r['generation']) for r in generations] != list(range(50)) or completed['qualified'] or
                len(training) != sum(int(r['training_episodes']) for r in generations) or
                {float(r['cap_m_s']) for r in training} != {1.}):
            raise ValueError(f'Incomplete or inconsistent finished study: {name}')
        data, identities = [], set()
        for generation in range(50):
            gates = read(directory / 'validation' / f'generation-{generation:04d}.json')['candidates']
            if len(gates) != 3:
                raise ValueError('Expected three candidates per generation.')
            for gate in gates:
                cache = read(directory / 'validation-cache' / (gate['genotype_sha256'] + '.json'))
                if cache['protocol_sha256'] != sha(plan_bytes) or cache['genotype_sha256'] != gate['genotype_sha256']:
                    raise ValueError('Validation cache fingerprint differs.')
                primary, retention = cache['primary'], cache['retention']
                if (len(primary) != 60 or len(retention) != 20 or
                        {r['track_hash']: r['group'] for r in primary} != expected_primary or
                        {r['track_hash']: r['group'] for r in retention} != expected_retention or
                        any(r['cap_m_s'] != 1. for r in primary) or any(r['cap_m_s'] != .5 for r in retention)):
                    raise ValueError('Validation count/caps differ.')
                observed = observed_summary({r['track_hash']: r for r in primary})
                row = {**gate, 'study': name,
                    'minimum_straight_speed_m_s': observed['minimum_required_window_speed_m_s'],
                    'maximum_raw_yaw_rms_rad_s': observed['maximum_observed_raw_yaw_rms_rad_s'],
                    'maximum_steering_rms': observed['maximum_observed_steering_rms'],
                    'missing_windows': observed['missing_windows'], 'retention_completed': sum(r['completed'] for r in retention),
                    'observed_geometric_rmse_m': observed['observed_mean_geometric_rmse_m']}
                data.append(row); identities.add(gate['genotype_sha256'])
                if (name, generation, gate['candidate_rank']) in (('strict', 40, 0), ('fast', 33, 2)):
                    model = content(directory / gate['model_file'])
                    artifact = pickle.loads(model); clone = deepcopy(artifact['genome']); clone.key, clone.fitness = 0, None
                    if sha(pickle.dumps(clone, protocol=4)) != gate['genotype_sha256']:
                        raise ValueError('Fixed representative model identity differs.')
                    fixed[name] = {'gate': row, 'primary': primary, 'model_sha256': sha(model)}
        if any(row['eligible'] for row in data):
            raise ValueError('Expected the completed studies with zero qualifiers.')
        only_speed = [r for r in data if r['reasons'] == ['straight_speed']]
        speed_pass = [r for r in data if not r['missing_windows'] and r['minimum_straight_speed_m_s'] >= .95 - 1e-12]
        population_counts = Counter(int(r['generation']) for r in fitness)
        studies[name] = {'training_episodes': len(training), 'evaluated_genomes': len(fitness),
            'population_sizes': dict(sorted(population_counts.items())), 'candidate_records': len(data),
            'distinct_validated_genotypes': len(identities), 'repeated_candidate_records': len(data) - len(identities),
            'known_validation_cache_episodes_used': 80 * len(identities),
            'new_validation_episode_counters': sum(int(r['newly_evaluated_validation_episodes']) for r in generations),
            'reference_episodes_new': sum(int(r.get('reference_episodes_new', r.get('reference_episodes', 0))) for r in generations),
            'reference_episodes_reused': sum(int(r.get('reference_episodes_reused', 0)) for r in generations),
            'training_quality_tiers': dict(Counter(r['tier'] for r in fitness)), 'qualified_records': 0,
            'rejection_counts': dict(Counter(reason for row in data for reason in row['reasons'])),
            'speed_pass_records': len(speed_pass), 'speed_pass_distinct_genotypes': len({r['genotype_sha256'] for r in speed_pass}),
            'fastest_speed_only_nonqualifier': max(only_speed, key=lambda r: r['minimum_straight_speed_m_s']) if only_speed else None}
        records[name] = data
        del training, fitness
    for generation in range(50):
        for filename in [*(f'{slot}.json' for slot in range(6)), 'batch.json', 'reference.json']:
            original = STRICT / 'development/seed-01/tracks' / f'generation-{generation:04d}' / filename
            copied = FOLLOW / 'prepared/batches' / f'generation-{generation:04d}' / filename
            if content(original) != content(copied):
                raise ValueError('Follow-up batches/reference bytes differ from strict V1.')
    suite_bytes = content(FOLLOW / 'prepared/suite/suite.json')
    if suite_bytes != content(STRICT / 'prepared/suite/suite.json'):
        raise ValueError('Consulted validation suites differ.')
    expected = {e['geometry_hash']: e['group'] for e in json.loads(suite_bytes)['splits']['validation']}
    pid_result = read(PID_DATA / 'episodes.json')
    if pid_result['pid_sha256'] != PID_SHA or pid_result['new_episodes'] != 60:
        raise ValueError('Unexpected saved PID evaluation.')
    freeze = Path(pid_result['freeze'])
    freeze_manifest = content(freeze / 'manifest.json')
    if sha(freeze_manifest) != pid_result['freeze_sha256'] or sha(content(freeze / 'pid-fullspeed.json')) != PID_SHA:
        raise ValueError('PID evaluation freeze differs.')
    policies = {'strict_g40': fixed['strict']['primary'], 'fast_g33': fixed['fast']['primary'], 'retuned_pid': pid_result['primary']}
    for rows in policies.values():
        if len(rows) != 60 or {r['track_hash']: r['group'] for r in rows} != expected or any(r['cap_m_s'] != 1. for r in rows):
            raise ValueError('Fixed comparisons require identical course geometry, groups and caps.')
    all_summaries = {name: observed_summary({r['track_hash']: r for r in rows}) for name, rows in policies.items()}
    pairs = {name + '_vs_retuned_pid': paired_summary({r['track_hash']: r for r in rows}, {r['track_hash']: r for r in policies['retuned_pid']})
             for name, rows in policies.items() if name != 'retuned_pid'}
    reference = read(FOLLOW / 'prepared/inputs/reference-validation.json')['primary']
    reference_summary = observed_summary({r['track_hash']: r for r in reference})
    output = args.out.resolve(); output.mkdir(parents=True, exist_ok=True)
    plot_results(records, reference_summary, fixed, output)
    for filename in ('report_quality_followup.py', 'report_quality_candidate_pid.py'):
        content(Path('research') / filename)
    if replay is None:
        blobs, manifest = {}, {}
        for index, (name, data) in enumerate(sorted(captured.items())):
            filename = Path(name).name if name.startswith('research/') else f'files/{index:04d}{Path(name).suffix}'
            blobs[filename] = data; manifest[name] = {'file': filename, 'sha256': sha(data)}
        address = sha(encode(manifest)); snapshot = output / 'snapshots' / address
        snapshot.parent.mkdir(exist_ok=True)
        if snapshot.exists():
            if (snapshot / 'manifest.json').read_bytes() != encode(manifest):
                raise ValueError('Existing report snapshot differs.')
            for filename, data in blobs.items():
                if (snapshot / filename).read_bytes() != data:
                    raise ValueError('Existing captured report bytes differ.')
        else:
            snapshot.mkdir()
            for filename, data in blobs.items():
                path = snapshot / filename; path.parent.mkdir(exist_ok=True); path.write_bytes(data)
            (snapshot / 'manifest.json').write_bytes(encode(manifest))
    command = f"& .\\.venv\\Scripts\\python.exe '{snapshot / 'report_quality_followup.py'}' --snapshot '{snapshot}' --out '{output}'"
    result = {'scope': 'Finished DEVELOPMENT study; no independent confirmation or fresh shared test; no official model promotion.',
        'studies': studies, 'identical_50_batch_and_reference_bytes': True, 'candidate_records': records,
        'fixed_representatives': {name: {key: value for key, value in item.items() if key != 'primary'} for name, item in fixed.items()},
        'retuned_pid_sha256': PID_SHA, 'all_episode_observed_summaries': all_summaries, 'shared_success_comparisons': pairs,
        'reference': reference_summary, 'snapshot': str(snapshot), 'input_sha256': {name: sha(data) for name, data in captured.items()},
        'report_source_sha256': sha(Path(__file__).read_bytes()), 'replay_command': command}
    (output / 'report.json').write_bytes(encode(result))
    write_report(result, output)
    print(json.dumps({'studies': {name: {key: value for key, value in item.items() if key in ('training_episodes', 'distinct_validated_genotypes', 'speed_pass_records', 'speed_pass_distinct_genotypes')} for name, item in studies.items()},
                      'pid_completion': all_summaries['retuned_pid']['completed'], 'snapshot': str(snapshot)}, indent=2))


def plot_results(records, reference, fixed, output):
    fields = [('minimum_straight_speed_m_s', 'Minimum required straight speed (m/s)', 1., .95),
              ('observed_geometric_rmse_m', 'Observed mean episode geometric RMSE (mm)', 1000., reference['observed_mean_geometric_rmse_m'] * 1000),
              ('command_jitter', 'Observed mean command jitter', 1., reference['observed_mean_command_jitter']),
              ('maximum_raw_yaw_rms_rad_s', 'Maximum observed raw straight yaw RMS (rad/s)', 1., .15)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, (field, label, scale, limit) in zip(axes.flat, fields):
        for name, rows in records.items():
            ax.scatter([r['generation'] for r in rows], [(r[field] if r[field] is not None else np.nan) * scale for r in rows], s=9, alpha=.2, color=COLORS[name])
            ax.plot(range(50), [np.nanmedian([(r[field] if r[field] is not None else np.nan) * scale for r in rows if r['generation'] == g]) for g in range(50)], color=COLORS[name], label=name)
        ax.axhline(limit, color='#b23b35', ls='--', linewidth=1)
        ax.set(xlabel='Generation', ylabel=label); ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Fixed consulted development validation: every training-ranked candidate\nLines are per-generation medians of three candidates; points and generations are not independent trials', fontsize=11)
    save_figure(fig, output, 'fixed-validation')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
    for ax, (field, label, scale, limit) in zip(axes, fields[1:]):
        for name, rows in records.items():
            unique = {r['genotype_sha256']: r for r in rows}.values()
            ax.scatter([r['minimum_straight_speed_m_s'] for r in unique], [(r[field] if r[field] is not None else np.nan) * scale for r in unique], s=16, alpha=.55, color=COLORS[name], label=name)
        for name, item in fixed.items():
            row = item['gate']
            ax.scatter(row['minimum_straight_speed_m_s'], row[field] * scale, marker='*' if name == 'strict' else 'D', s=85, edgecolor='black', color=COLORS[name], label='fixed ' + name)
        ax.axvline(.95, color='#555', ls='--', linewidth=1); ax.axhline(limit, color='#b23b35', ls='--', linewidth=1)
        ax.set(xlabel='Minimum required straight speed (m/s)', ylabel=label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Joint speed/quality development trade-offs: one point per distinct genotype within each arm\nMissing required windows give zero speed; observed means include failed-episode exposure', fontsize=11)
    save_figure(fig, output, 'speed-quality-tradeoffs')


def save_figure(fig, output, name):
    for extension in ('png', 'svg'):
        fig.savefig(output / f'{name}.{extension}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def write_report(result, output):
    studies = result['studies']
    text = ['# Finished quality-speed follow-up — DEVELOPMENT', '',
        'Both revised-fitness arms finished 50 generations with zero qualifying candidates. Strict V1 also had zero qualifiers. All 50 six-course batches and saved reference rows match byte-for-byte; consulted validation is the same 60 primary plus 20 retention courses. These results do not establish independent training repeatability or test generalization.', '',
        '| Study | Candidate training episodes | Evaluated genomes | Unique validated genotypes / 150 records | New validation counters / known cached episodes used | New / reused reference episodes |',
        '|---|---:|---:|---:|---:|---:|']
    for name, item in studies.items():
        text.append(f"| {name} | {item['training_episodes']:,} | {item['evaluated_genomes']:,} | {item['distinct_validated_genotypes']} / 150 | {item['new_validation_episode_counters']:,} / {item['known_validation_cache_episodes_used']:,} | {item['reference_episodes_new']} / {item['reference_episodes_reused']} |")
    text += ['', 'Population sizes may fluctuate under stock NEAT; JSON records every generation size, all candidate decisions, rejection counts, repeats and training-quality tiers. Training tier names describe completion/quality status; the continuous reward differs from V1, so raw scalar fitness must not be compared as a common outcome. Cached episode counts exclude unknown partial work; candidates and generations are not independent replicates.', '',
             '## Speed and quality together', '']
    for name in ('smooth', 'fast'):
        near = studies[name]['fastest_speed_only_nonqualifier']
        text.append(f"The {name} arm's fastest speed-only nonqualifier was g{near['generation']} rank {near['candidate_rank']}: minimum required straight speed {near['minimum_straight_speed_m_s']:.6f} m/s, still below 0.95 m/s.")
    fast = studies['fast']; rep = result['fixed_representatives']['fast']['gate']
    text += ['', f"The fast arm produced {fast['speed_pass_records']} speed-passing candidate records representing {fast['speed_pass_distinct_genotypes']} distinct genotypes, all rejected for other acceptance failures. They are not independent training repetitions. Fixed g33 rank 2 completed 60 primary and {rep['retention_completed']} retention courses and passed paired time, with {rep['minimum_straight_speed_m_s']:.6f} m/s minimum straight speed, but mean episode geometric RMSE {rep['geometric_rmse_m']*1000:.3f} mm, jitter {rep['command_jitter']:.7f}, maximum raw yaw {rep['maximum_raw_yaw_rms_rad_s']:.5f} rad/s and steering {rep['maximum_steering_rms']:.5f} exceed their original limits (14.493 mm, 0.0065299, 0.15 and 0.02). No official winner is promoted.", '',
             '![Fixed development validation](fixed-validation.png)', '', '![Joint speed and quality](speed-quality-tradeoffs.png)', '',
             'The error reference line is the full fast-parent mean; the actual acceptance gate uses each candidate’s shared successful tracks. Observed all-episode error includes partial exposure on failures. Read figures alongside completion/rejection data; median curves do not select a model.', '',
             '## Fixed representatives versus the newly frozen retuned PID', '',
             f"PID config SHA-256: `{PID_SHA}`. Exactly 60 new PID episodes were evaluated, using two workers and a source/protocol/suite/config freeze created before evaluation. Both NEAT representatives use existing saved caches. This is reused development validation, not a fresh shared test.", '',
             '| Controller | Completion / 60 | Failures | Observed mean episode error | Observed mean jitter | Min required speed | Max raw yaw | Max steering |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for name, item in result['all_episode_observed_summaries'].items():
        text.append(f"| {name} | {item['completed']} | {item['failures'] or 'none'} | {item['observed_mean_geometric_rmse_m']*1000:.3f} mm | {item['observed_mean_command_jitter']:.7f} | {item['minimum_required_window_speed_m_s']:.5f} m/s | {item['maximum_observed_raw_yaw_rms_rad_s']:.5f} rad/s | {item['maximum_observed_steering_rms']:.5f} |")
    text += ['', '| NEAT / retuned PID | Shared successes / 60 | Mean time NEAT / PID | Mean episode error NEAT / PID | Mean jitter NEAT / PID |',
             '|---|---:|---:|---:|---:|']
    for name, pair in result['shared_success_comparisons'].items():
        t, e, j = (pair[field] for field in ('completion_time_s', 'geometry_lateral_rmse_m', 'command_jitter'))
        text.append(f"| {name} | {pair['shared_successes']} | {t['left']:.3f} / {t['right']:.3f} s | {e['left']*1000:.3f} / {e['right']*1000:.3f} mm | {j['left']:.7f} / {j['right']:.7f} |")
    text += ['', 'All failures remain in JSON, including their full episode rows. Paired comparisons use identical shared-success course IDs and report the denominator; all-episode observed error/jitter means include failure exposure. Required-window minima include zero for missing windows; yaw/steering maxima use observed raw windows. Neither representative qualifies under every unchanged acceptance condition. An independent fresh shared test must follow the settled protocol and final PID/NEAT controller freeze.', '',
             f"Exact consumed bytes and both report helpers are preserved in hash-addressed snapshot `{Path(result['snapshot']).name}`. JSON records input/model/source hashes and all numeric evidence. Old strict-V1 g21/g40 reports remain unchanged.", '',
             'Replay this report without physics:', '', '```powershell', result['replay_command'], '```', '']
    (output / 'report.md').write_text('\n'.join(text), encoding='utf-8')


if __name__ == '__main__':
    main()
