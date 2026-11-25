"""Descriptive per-seed results and figures; no post-test controller selection."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
from statistics import mean

from neat_smooth_study.common import digest, read, write_json
from .confirmation import SEEDS, SPEED_SCHEDULE
from .fitness import score_batch


def summarize(rows):
    windows = [window for row in rows for window in row['straight_windows']]
    observed = [window for window in windows if window['observed']]
    return {'courses': len(rows), 'completed': sum(row['completed'] for row in rows),
            'mean_progress': mean(row['progress'] for row in rows),
            'mean_geometric_rmse_m': mean(row['geometry_lateral_rmse_m'] for row in rows),
            'mean_command_jitter': mean(row['command_jitter'] for row in rows),
            'mean_command_change_rms': mean(math.sqrt(row['command_jitter']) for row in rows),
            'mean_episode_forward_speed_m_s': mean(row['average_forward_speed_m_s'] for row in rows),
            'peak_forward_speed_m_s': max(row['peak_forward_speed_m_s'] for row in rows),
            'missing_straight_windows': len(windows) - len(observed),
            'minimum_straight_speed_m_s': min((max(0., window['speed_m_s']) if window['observed'] else 0.
                                              for window in windows), default=None),
            'maximum_straight_yaw_rms_rad_s': max((window['yaw_rms_rad_s'] for window in observed), default=None),
            'maximum_straight_steering_rms': max((window['steering_rms'] for window in observed), default=None),
            'failure_reasons': {reason: sum(not row['completed'] and row['termination'] == reason for row in rows)
                                for reason in sorted({row['termination'] for row in rows if not row['completed']})}}


def pair(rows, reference):
    left = {row['track_hash']: row for row in rows}
    right = {row['track_hash']: row for row in reference}
    if len(left) != len(rows) or len(right) != len(reference) or set(left) != set(right):
        raise ValueError('Paired results require identical distinct geometries.')
    shared = sorted(identity for identity in left if left[identity]['completed'] and right[identity]['completed'])
    result = {'shared_successes': len(shared), 'courses': len(left), 'track_hashes': shared}
    if shared:
        for field in ('completion_time_s', 'geometry_lateral_rmse_m', 'command_jitter'):
            result[field] = {'neat': mean(left[identity][field] for identity in shared),
                             'reference': mean(right[identity][field] for identity in shared)}
    return result


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'report'
    plan = read(run / 'prepared/protocol.json')
    protocol_hash = digest(run / 'prepared/protocol.json')
    hashes, training = {}, {}
    def captured(path):
        data = path.read_bytes()
        hashes[path.relative_to(run).as_posix()] = hashlib.sha256(data).hexdigest()
        return data
    for seed in SEEDS:
        directory = run / f'seed-{seed:02d}'
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        if not checkpoint.exists():
            training[str(seed)] = []
            continue
        state = pickle.loads(gzip.decompress(captured(checkpoint)))
        if state['signature']['protocol_sha256'] != protocol_hash or state['signature']['seed'] != seed:
            raise ValueError('Training report contains a different seed/protocol.')
        rows = []
        for generation in range(state['population'].generation):
            record = json.loads(captured(directory / 'validation' / f'generation-{generation:04d}.json'))
            actual = score_batch(record['primary'], smoothness_weight=plan['smoothness_weight'])
            if actual != record['primary_score']:
                raise ValueError('Logged validation fitness differs from its measured episodes.')
            rows.append({'generation': generation, 'cap_m_s': record['cap_m_s'],
                         'primary_fitness': actual['fitness'], 'primary': summarize(record['primary']),
                         'retention': summarize(record['retention']),
                         'drive_evaluated': record['drive_evaluated'], 'steering_evaluated': record['steering_evaluated']})
        training[str(seed)] = rows
    assessment_root = run / 'assessment'
    final = None
    if (assessment_root / 'complete.json').exists():
        complete = json.loads(captured(assessment_root / 'complete.json'))
        freeze = json.loads(captured(assessment_root / 'test-freeze.json'))
        if (complete['protocol_sha256'] != protocol_hash
                or complete['controllers_sha256'] != digest(assessment_root / 'frozen/manifest.json')
                or complete['suite_sha256'] != digest(assessment_root / 'suite/suite.json')
                or freeze['controllers_sha256'] != complete['controllers_sha256']):
            raise ValueError('Final report controller/test freeze differs.')
        manifest = read(assessment_root / 'frozen/manifest.json')
        if any(digest(assessment_root / 'frozen' / name) != row['sha256'] for name, row in manifest['controllers'].items()):
            raise ValueError('Final report controller bytes changed.')
        entries = read(assessment_root / 'suite/suite.json')['splits']['test']
        expected = [entry['geometry_hash'] for entry in entries]
        raw, summaries = {}, {}
        for name, expected_hash in complete['result_files'].items():
            path = assessment_root / 'episodes' / name
            data = captured(path)
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError('Completed assessment data changed.')
            record = json.loads(data)
            rows = record['episodes']
            if (record['controllers_sha256'] != complete['controllers_sha256']
                    or record['suite_sha256'] != complete['suite_sha256']
                    or [row['track_hash'] for row in rows] != expected
                    or any(row['cap_m_s'] != int(path.stem.rsplit('-', 1)[1])/100 or row['error'] for row in rows)):
                raise ValueError('Final result geometries or controller identities differ.')
            raw[path.stem] = rows
            summaries[path.stem] = summarize(rows)
        paired = {}
        for cap in (100, 50):
            suffix = f'cap-{cap:03d}'
            for seed in SEEDS:
                name = f'neat-seed-{seed:02d}-{suffix}'
                paired[name] = {reference: pair(raw[name], raw[f'{reference}-{suffix}'])
                                for reference in ('tuned-pid', 'old-prototype')}
        final = {'summaries': summaries, 'paired': paired,
                 'assessment_episode_records': complete['assessment_episode_records'],
                 'controller_freeze': manifest, 'test_freeze': freeze,
                 'seed_spread': {str(cap): {
                     field: {'mean': mean(summaries[f'neat-seed-{seed:02d}-cap-{cap:03d}'][field] for seed in SEEDS),
                             'minimum': min(summaries[f'neat-seed-{seed:02d}-cap-{cap:03d}'][field] for seed in SEEDS),
                             'maximum': max(summaries[f'neat-seed-{seed:02d}-cap-{cap:03d}'][field] for seed in SEEDS)}
                     for field in ('completed', 'mean_geometric_rmse_m', 'mean_command_change_rms', 'minimum_straight_speed_m_s')}
                     for cap in (100, 50)}}
    result = {'protocol_sha256': protocol_hash, 'smoothness_weight': plan['smoothness_weight'],
              'training': training, 'final_assessment': final, 'input_hashes': hashes,
              'report_source_sha256': digest(Path(__file__)),
              'scope': 'A frozen five-seed from-scratch protocol; committed data and completion state are shown separately. No seed is selected using final-test results.',
              'limitations': ['Observed all-episode errors include partial failures; paired successful-track metrics carry explicit denominators.',
                              'Seed ranges describe these five runs and are not confidence intervals.',
                              'Command-change RMS is an actuation metric per fixed control step, not measured chassis vibration.',
                              'Simulation geometry generalization does not establish hardware robustness or superiority to all PID designs.',
                              'The combined protocol does not isolate individual curriculum/reward changes or establish an optimal generation/population budget.']}
    write_json(output / 'report.json', result)
    _figures(result, output / 'figures')
    lines = ['# Compact NEAT methodology — five-seed results', '',
             '**' + ('Final shared-test assessment complete.' if final else 'Interim training report: final assessment is not complete.') + '**', '',
             result['scope'], '', f'Frozen smoothness coefficient: {plan["smoothness_weight"]:g}.', '']
    if final:
        lines += ['| Cap | Seed | Completion | Shared finishes with PID | Paired time NEAT / PID | Min straight m/s |',
                  '|---:|---:|---:|---:|---|---:|']
        for cap in (100, 50):
            for seed in SEEDS:
                name = f'neat-seed-{seed:02d}-cap-{cap:03d}'
                summary, paired = final['summaries'][name], final['paired'][name]['tuned-pid']
                timing = paired.get('completion_time_s')
                text = f'{timing["neat"]:.3f} / {timing["reference"]:.3f} s' if timing else 'n/a'
                lines.append(f'| {cap/100:g} | {seed} | {summary["completed"]}/180 | {paired["shared_successes"]}/180 | '
                             f'{text} | {summary["minimum_straight_speed_m_s"]:.4f} |')
        _methods_reference(plan, output)
    lines += ['', 'All failures, per-seed spread, paired error/jitter, physical speeds and raw input hashes are in `report.json`.', '',
              'Figures are SVG and PNG under `figures/`. Curves use fixed validation geometries at the current training cap; speed-cap transitions are marked. Fresh training batches differ across generations.', '',
              *['- ' + note for note in result['limitations']], '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(output), 'final_assessment_complete': final is not None,
                      'committed_generations': {seed: len(rows) for seed, rows in training.items()}}))
    return result


def _figures(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams['svg.fonttype'] = 'none'
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout='constrained')
    fields = [('completed', 'Validation completion / 60', 1),
              ('minimum_straight_speed_m_s', 'Min straight speed / current cap', 1),
              ('mean_geometric_rmse_m', 'Geometric RMSE (mm)', 1000),
              ('mean_command_change_rms', 'RMS command change / step', 1),
              ('maximum_straight_yaw_rms_rad_s', 'Max straight yaw RMS (rad/s)', 1)]
    for axis, (field, label, scale) in zip(axes.flat, fields):
        for seed, rows in result['training'].items():
            if not rows:
                continue
            values = [row['primary'][field]*scale / (row['cap_m_s'] if field == 'minimum_straight_speed_m_s' else 1.)
                      if row['primary'][field] is not None else math.nan for row in rows]
            axis.plot([row['generation'] for row in rows], values, label='Seed ' + seed, linewidth=1)
        shared = min(map(len, result['training'].values()), default=0)
        if shared:
            values = [[rows[generation]['primary'][field]*scale /
                       (rows[generation]['cap_m_s'] if field == 'minimum_straight_speed_m_s' else 1.)
                       if rows[generation]['primary'][field] is not None else math.nan
                       for rows in result['training'].values()] for generation in range(shared)]
            axis.plot(range(shared), [mean(row) for row in values], color='#222', linewidth=1.8, label='Five-seed mean')
            axis.fill_between(range(shared), [min(row) if all(math.isfinite(x) for x in row) else math.nan for row in values],
                              [max(row) if all(math.isfinite(x) for x in row) else math.nan for row in values],
                              color='#777', alpha=.15, label='Five-seed min–max')
        axis.set(xlabel='Generation', ylabel=label)
    for seed, rows in result['training'].items():
        if rows:
            axes.flat[-1].plot([row['generation'] for row in rows], [row['primary_fitness'] for row in rows], label='Seed ' + seed, linewidth=1)
    axes.flat[-1].set(xlabel='Generation', ylabel='Primary compact validation fitness')
    for axis in axes.flat:
        for start, cap in SPEED_SCHEDULE[1:]:
            axis.axvline(start, color='#999', linewidth=.6, linestyle=':')
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    axes.flat[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes.flat[1].axhline(.95, color='#a34b34', linestyle='--', linewidth=.8)
    figure.suptitle('Independent lineages on fixed validation geometries; cap changes at marked generations')
    for extension in ('png', 'svg'):
        figure.savefig(output / f'validation-learning.{extension}', dpi=300)
    plt.close(figure)
    final = result['final_assessment']
    if final is None:
        return
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for row_axes, cap in zip(axes, (100, 50)):
        summaries = [final['summaries'][f'neat-seed-{seed:02d}-cap-{cap:03d}'] for seed in SEEDS]
        pid = final['summaries'][f'tuned-pid-cap-{cap:03d}']
        row_axes[0].scatter(SEEDS, [row['completed']/180 for row in summaries], label='NEAT', s=45)
        row_axes[0].axhline(pid['completed']/180, color='#b55c37', linestyle='--', label='Tuned PID')
        row_axes[0].set(ylabel=f'{cap/100:g} m/s: completion fraction', ylim=(0, 1.02))
        for index, field in enumerate(('completion_time_s', 'command_jitter'), start=1):
            ratios, seeds = [], []
            for seed in SEEDS:
                pair = final['paired'][f'neat-seed-{seed:02d}-cap-{cap:03d}']['tuned-pid']
                if field in pair and pair[field]['reference'] > 0:
                    seeds.append(seed)
                    ratios.append(pair[field]['neat']/pair[field]['reference'])
            row_axes[index].scatter(seeds, ratios, s=45)
            row_axes[index].axhline(1., color='#b55c37', linestyle='--')
            row_axes[index].set(ylabel='Paired ' + ('finish-time' if index == 1 else 'command-jitter') + ' ratio to PID')
        for axis in row_axes:
            axis.set(xlabel='Independent seed', xticks=SEEDS)
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('Fresh shared tracks — all five selected controllers; paired denominators in report')
    for extension in ('png', 'svg'):
        figure.savefig(output / f'final-seed-variation.{extension}', dpi=300)
    plt.close(figure)


def _methods_reference(plan, output):
    weight = plan['smoothness_weight']
    text = r'''% Reference material only; not automatically inserted into the manuscript.
% Requires amsmath. Review prose section by section with the user.
\paragraph{Fixed task objective.}
For a batch of $m=6$ courses, let $C$ be its number of successful completions,
$p_i$ ordered-route progress, $L_i$ route length, $t_i$ successful completion
time, $v_{\max}$ the current wheel-speed cap, and $J_i$ normalized mean squared
wheel-command change at the fixed control interval. We use
\[
U=\begin{cases}
\frac{1}{m}\sum_i p_i, & C<m,\\
\frac{1}{m}\sum_i \frac{L_i/(t_i v_{\max})}{1+\lambda\sqrt{J_i}}, & C=m,
\end{cases}
\qquad F=C+\frac{U}{1+U}.
\]
Since $U\ge0$ is finite, $0\le U/(1+U)<1$; an additional completion
therefore outranks any secondary-utility improvement. The transform also
changes numerical fitness spacing used by stock NEAT species allocation.
The coefficient $\lambda=WEIGHT$ was frozen after a bounded three-coefficient
warm-start development comparison; it is not claimed optimal.

\paragraph{Independent training and selection.}
Five independently initialized populations target 100 genomes each.
One objective and fixed NEAT parameters are used for all 400 generations,
with wheel-speed caps 0.50, 0.65, 0.80 and 1.00 m/s over generation ranges
0--149, 150--199, 200--249 and 250--399. Each generation shares three mixed,
two speed/recovery and one corner course across its population. Actual
population sizes and episode budgets are logged. No intermediate champion
restart or final-stage parameter change is used. One training champion per
generation is evaluated on 60 shared primary validation courses at the current
cap and 30 separate retention courses at 0.5 m/s. Each final controller is
selected by primary validation fitness among generations 250--399, with
earliest-candidate tie breaking. Retention is reported separately.

\paragraph{Final comparison and scope.}
All five selected controllers and cap-specific full-base filtered PID
configurations are frozen before generating 180 new shared geometries:
90 mixed, 60 speed/recovery and 30 corner courses. Primary evaluation is at
1.0 m/s, with transfer at 0.5 m/s. The previously accepted NEAT prototype is
a disclosed development control. Completion and failures are reported per
seed. Time and quality comparisons on shared successful tracks include
their denominators. Command variation is not a direct vibration measure.
This simulation study tests the resulting controllers and combined protocol;
it does not establish hardware robustness, a minimal training budget, or
causal attribution to an individual curriculum/reward change.
'''.replace('WEIGHT', f'{weight:g}')
    (output / 'methods-reference.tex').write_text(text, encoding='utf-8')
