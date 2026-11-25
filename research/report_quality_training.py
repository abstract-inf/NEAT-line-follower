"""Regenerate development figures from saved episodes; never runs evolution or physics."""
import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {'mixed': '#007f86', 'speed': '#c45a28', 'corner': '#7756a0'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, output, name):
    for extension in ('png', 'svg'):
        fig.savefig(output / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def archive_completed(run, output):
    """Bundle a finished stage; never copy a population while it is evolving."""
    directory = run / 'development/seed-01'
    completed = read(directory / 'complete.json')
    plan = read(run / 'prepared/protocol.json')
    if completed['generations'] != plan['settings']['generations']:
        raise ValueError('The declared development budget is not complete.')
    if completed['signature']['protocol_sha256'] != digest(run / 'prepared/protocol.json'):
        raise ValueError('Completed stage has a different protocol.')
    target = output.parent / 'complete-development-bundle.zip'
    if target.exists():
        raise FileExistsError('Preserve the existing completed bundle; use a different --out directory for another archive.')
    paths = [p for p in run.rglob('*') if p.is_file() and p.name != 'session.lock' and not p.name.endswith('.tmp')]
    paths.extend(p for p in output.rglob('*') if p.is_file())
    paths.extend(p for p in (output.parent / 'objective-diagnostic').rglob('*') if p.is_file())
    paths.extend(p for p in (output.parent / 'candidate-pid-development').rglob('*') if p.is_file())
    paths.extend(p for p in (output.parent / 'candidate-g40-pid-development').rglob('*') if p.is_file())
    paths.extend(ROOT / relative for relative in (
        'docs/experiment-ledger.md', 'docs/neat-quality-speed-study.md',
        'run_neat_quality_study.py', 'requirements-lock.txt', 'tests/test_neat_quality_study.py',
        'research/report_quality_training.py', 'research/analyze_quality_networks.py',
        'research/diagnose_quality_objective.py',
        'research/report_quality_candidate_pid.py',
        'research/Quality_Constrained_NEAT_Design.md', 'research/Constraint_Relaxation_NEAT_Followup.md',
        'research/Quality_Speed_V1_Audit.md', 'research/Quality_Speed_Confirmation_Design.md',
        'results/neat-quality-speed-v1/fitness-calibration.json',
        'results/neat-quality-speed-v1/fitness-calibration.md',
        'results/neat-quality-speed-v1/runtime-environment.json',
        'results/neat-quality-speed-v1/neat-runtime-source-integrity.json',
        'results/neat-quality-speed-v1/prepared-bundle-reproduction-notes.md',
        'results/neat-quality-speed-v1/prepared-bundle-restore-check.json',
        'runs/neat-quality-speed-v1/implementation-checks.json'))
    calibration = read(ROOT / 'results/neat-quality-speed-v1/fitness-calibration.json')
    for relative, expected in calibration['input_sha256'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError('Preserved calibration input changed.')
        paths.append(path)
    entries = {path.relative_to(ROOT).as_posix(): path for path in set(paths)}
    # A clean extraction needs importable root packages as well as the preserved snapshot.
    # Materialize the frozen version, even if live development source later changes.
    for relative, expected in plan['source_files'].items():
        path = run / 'prepared/source' / relative
        if digest(path) != expected:
            raise ValueError('Frozen source changed before archival.')
        entries[relative] = path
    for relative, expected in plan['prepared_files'].items():
        if digest(run / 'prepared' / relative) != expected:
            raise ValueError('Prepared input changed before archival.')
    manifest = {relative: digest(path) for relative, path in sorted(entries.items())}
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for relative in manifest:
            bundle.write(entries[relative], relative)
        bundle.writestr('quality-development-file-hashes.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Archive CRC verification failed.')
    temporary.replace(target)
    record = {'file': target.name, 'sha256': digest(target), 'bytes': target.stat().st_size,
              'files': len(manifest), 'scope': 'Completed development stage, including frozen source/inputs, logs, models, population/RNG checkpoint and report. No fresh final test.',
              'restore': 'Extract into a separate clean project root, preserve paths, use the recorded Python/dependencies and run run_neat_quality_study.py verify. Do not extract over an active run.'}
    (target.parent / 'complete-development-bundle-manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f'Verified completed-run bundle: {target}')


def measurements(rows):
    windows = [w for r in rows for w in r['straight_windows']]
    observed = [w for w in windows if w['observed']]
    return {
        'minimum_straight_speed_m_s': min((w['speed_m_s'] if w['observed'] else 0. for w in windows), default=0.),
        'maximum_straight_yaw_rms_rad_s': max((w['yaw_rms_rad_s'] for w in observed), default=0.),
        'maximum_straight_steering_rms': max((w['steering_rms'] for w in observed), default=0.),
        'missing_windows': len(windows) - len(observed),
        'retention_completed': None,
    }


def candidates(directory, generations, protocol_hash):
    records, sources, cache_counts = [], {}, {}
    for generation in generations:
        path = directory / 'validation' / f'generation-{generation:04d}.json'
        sources[str(path)] = digest(path)
        for gate in read(path)['candidates']:
            cached_path = directory / 'validation-cache' / (gate['genotype_sha256'] + '.json')
            cached = read(cached_path)
            if cached['protocol_sha256'] != protocol_hash or cached['genotype_sha256'] != gate['genotype_sha256']:
                raise ValueError('Validation cache fingerprint mismatch.')
            if any(r['cap_m_s'] != 1. for r in cached['primary']) or any(r['cap_m_s'] != .5 for r in cached['retention']):
                raise ValueError('Recorded validation caps differ from the protocol.')
            sources[str(cached_path)] = digest(cached_path)
            cache_counts[gate['genotype_sha256']] = len(cached['primary']) + len(cached['retention'])
            metrics = measurements(cached['primary'])
            metrics['retention_completed'] = sum(r['completed'] for r in cached['retention'])
            records.append({**gate, **metrics})
    return records, sources, cache_counts


def figures(output, generations, fitness, records, references):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    gen = [r['generation'] for r in records]
    complete = np.array([r['completed'] == 60 and r['retention_completed'] == 20 for r in records])
    for ax, field, label, limit in zip(axes.flat, (
            'minimum_straight_speed_m_s', 'maximum_straight_yaw_rms_rad_s',
            'geometric_rmse_m', 'paired_time_ratio'), (
            'Minimum required straight speed (m/s)', 'Maximum straight yaw RMS (rad/s)',
            'Shared-finish geometric RMSE (mm)', 'Paired mean finish time / fast parent'),
            (.95, .15, references['fast']['geometric_rmse_m'] * 1000, 1.05)):
        scale = 1000 if field == 'geometric_rmse_m' else 1
        values = [np.nan if r[field] is None else r[field] * scale for r in records]
        ax.scatter(gen, values, s=17, color='#aab1b8', label='Incomplete primary or retention')
        ax.scatter(np.array(gen)[complete], np.array(values)[complete], s=22, color='#007f86', label='60/60 primary + 20/20 retention')
        ax.axhline(limit, color='#bd3f32', ls='--', label='Acceptance limit')
        ax.set(xlabel='Generation', ylabel=label)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, loc='best')
    fig.suptitle('Fixed development validation: three training candidates per generation\nRepeated genotypes are cached; points are not independent trials', fontsize=12)
    save_figure(fig, output, 'validation-speed-quality')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained')
    for ax, field, label, limit in zip(axes, (
            'maximum_straight_yaw_rms_rad_s', 'maximum_straight_steering_rms', 'command_jitter'), (
            'Maximum straight yaw RMS (rad/s)', 'Maximum straight steering RMS', 'Mean command jitter'),
            (.15, .02, references['fast']['command_jitter'])):
        points = ax.scatter([r['minimum_straight_speed_m_s'] for r in records], [r[field] for r in records],
                            c=gen, cmap='viridis', alpha=.75, s=25)
        for name, marker, label_text in (('stable', 's', 'Stable g83 parent (60/60)'),
                                         ('fast', '*', 'Fast g134 reference (60/60)'),
                                         ('pid', 'D', 'Full-speed filtered PID (59/60)')):
            ref = references[name]
            ax.scatter(ref['minimum_straight_speed_m_s'], ref[field], marker=marker, color='#c45a28',
                       edgecolors='black', s=90, label=label_text, zorder=4)
        ax.axvline(.95, color='#555', ls='--'); ax.axhline(limit, color='#555', ls='--')
        ax.set_yscale('log')
        ax.set(xlabel='Minimum required straight speed (m/s)', ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.colorbar(points, ax=axes, label='Generation', shrink=.8)
    fig.suptitle('Development speed–quality trade-offs (logarithmic quality axes; completion reported separately)')
    save_figure(fig, output, 'validation-tradeoffs')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    tiers = ('incomplete', 'completed_quality_violation', 'completed_quality_feasible')
    fractions = []
    for tier in tiers:
        fractions.append([mean(r['tier'] == tier for r in fitness if int(r['generation']) == g) for g in generations])
    axes[0].stackplot(generations, fractions, labels=('Incomplete', 'Completed, quality violation', 'Completed, quality feasible'),
                      colors=('#c45a28', '#b5bac1', '#007f86'))
    axes[0].set(ylabel='Fraction of genomes', ylim=(0, 1)); axes[0].legend(fontsize=7, loc='upper left')
    for field, label, color in (('straight_speed_utility', 'Straight speed utility', '#007f86'),
                                ('mean_route_speed_utility', 'Route speed utility', '#7756a0')):
        axes[1].plot(generations, [mean(float(r[field]) for r in fitness if int(r['generation']) == g) for g in generations],
                     label=label, color=color)
    axes[1].set(ylabel='Population mean utility', ylim=(0, 1)); axes[1].legend(fontsize=7)
    axes[2].plot(generations, [mean(float(r['quality_violation']) for r in fitness if int(r['generation']) == g) for g in generations], color='#c45a28')
    axes[2].set(ylabel='Population mean relative quality violation')
    for ax in axes:
        ax.set_xlabel('Generation'); ax.grid(alpha=.2)
    fig.suptitle('Training objective components on fresh six-course batches\nChanging batch difficulty prevents interpreting these as a fixed-suite learning curve', fontsize=11)
    save_figure(fig, output, 'training-components')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/neat-quality-speed-v1')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/neat-quality-speed-v1/training')
    parser.add_argument('--archive', action='store_true', help='Also archive the completed run; rejects active stages and existing archives.')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.out.resolve()
    directory = run / 'development/seed-01'
    if args.archive and not (directory / 'complete.json').exists():
        raise ValueError('Do not archive an active development stage; wait for complete.json.')
    generation_rows = table(directory / 'generations.csv')
    generations = [int(r['generation']) for r in generation_rows]
    if not generations:
        raise ValueError('No logged generations to report.')
    protocol_hash = digest(run / 'prepared/protocol.json')
    records, sources, cache_counts = candidates(directory, generations, protocol_hash)
    fitness = [r for r in table(directory / 'genome-fitness.csv') if int(r['generation']) in generations]
    training_caps = {float(r['cap_m_s']) for r in table(directory / 'training-episodes.csv') if int(r['generation']) in generations}
    if training_caps != {1.}:
        raise ValueError('Recorded training caps differ from the protocol.')
    references = {}
    for name, filename in (('stable', 'stable-neat-g83'), ('fast', 'fast-neat-g134'), ('pid', 'pid-fullspeed-1m')):
        path = run / 'baseline-development' / (filename + '.json')
        rows = read(path)['primary']
        sources[str(path)] = digest(path)
        references[name] = {**measurements(rows), 'geometric_rmse_m': mean(r['geometry_lateral_rmse_m'] for r in rows),
                            'command_jitter': mean(r['command_jitter'] for r in rows)}
    eligible = [r for r in records if r['eligible']]
    speed_only = [r for r in records if r['reasons'] == ['straight_speed']]
    near = max(speed_only, key=lambda r: r['minimum_straight_speed_m_s']) if speed_only else None
    selected = read(directory / 'selection.json') if (directory / 'selection.json').exists() else None
    complete = (directory / 'complete.json').exists()
    for path in (run / 'prepared/protocol.json', directory / 'training-episodes.csv',
                 directory / 'selection.json', directory / 'complete.json'):
        if path.exists():
            sources[str(path)] = digest(path)
    summary = {
        'scope': 'Development selection on repeatedly consulted validation. No independent confirmation or fresh test.',
        'complete': complete, 'logged_generations': len(generations), 'protocol_sha256': protocol_hash,
        'verified_recorded_caps_m_s': {'training': 1., 'primary_validation': 1., 'retention': .5},
        'candidate_records': len(records), 'unique_validated_genotypes': len({r['genotype_sha256'] for r in records}),
        'training_episodes': sum(int(r['training_episodes']) for r in generation_rows),
        'training_reference_episodes': sum(int(r['reference_episodes']) for r in generation_rows),
        'new_validation_episodes': sum(int(r['newly_evaluated_validation_episodes']) for r in generation_rows),
        'known_completed_cache_episodes_for_reported_candidates': sum(cache_counts.values()),
        'validation_counter_scope': 'new_validation_episodes sums committed generation counters. Known completed cache episodes include reusable results saved before an interruption. Neither counts unknown partial evaluations or currently unlogged generation candidates.',
        'qualified_candidate_records': len(eligible), 'selected': selected,
        'fastest_speed_only_nonqualifier': near,
        'rejection_counts': dict(Counter(reason for r in records for reason in r['reasons'])),
        'references': references, 'candidate_metrics': records,
        'input_sha256': {**sources, str(directory / 'generations.csv'): digest(directory / 'generations.csv'),
                         str(directory / 'genome-fitness.csv'): digest(directory / 'genome-fitness.csv')},
        'report_source_sha256': digest(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    with (output / 'candidate-metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows({**r, 'reasons': ';'.join(r['reasons'])} for r in records)
    figures(output, generations, fitness, records, references)
    text = ['# Quality-speed development results', '',
            f"Status: {'completed' if complete else 'interim; run is still in progress'}. {len(generations)} logged generations, {summary['training_episodes']:,} candidate training episodes, {summary['training_reference_episodes']:,} reference episodes and {summary['known_completed_cache_episodes_for_reported_candidates']:,} completed cache-backed validation episodes for the reported candidates.", '',
            f"Generation counters record {summary['new_validation_episodes']:,} new validation episodes. After interruption they can exclude previously cached complete evaluations; the cache-backed count retains that known work. Unknown partial evaluations and currently unlogged generation candidates are excluded from both reported budgets.", '',
            f"{len(records)} candidate records represent {summary['unique_validated_genotypes']} distinct validated genotypes. Exact repeats reuse the frozen deterministic evaluation; generations/candidates are not independent repetitions.", '',
            f"{len(eligible)} candidate records passed every unchanged acceptance rule. An official best model exists only if a candidate qualifies.", '']
    if selected:
        text += [f"The current selected model is generation {selected['generation']}, candidate rank {selected['candidate_rank']}, `{selected['model_file']}`. Its exact selection and protocol/model hashes remain in the run directory.", '']
    if near:
        text += [f"The fastest candidate rejected only for straight speed is generation {near['generation']}, rank {near['candidate_rank']}: minimum required straight speed {near['minimum_straight_speed_m_s']:.4f} m/s, maximum straight yaw RMS {near['maximum_straight_yaw_rms_rad_s']:.4f} rad/s, 60/60 primary and {near['retention_completed']}/20 retention completions. Paired time ratio to fast g134 is {near['paired_time_ratio']:.4f}; shared-finish geometric RMSE is {near['geometric_rmse_m']*1000:.3f} mm and mean command jitter {near['command_jitter']:.7f}. This is a documented trade-off, not a qualifying winner.", '']
    text += ['![Fixed validation curves](validation-speed-quality.png)', '', '![Speed and quality trade-offs](validation-tradeoffs.png)', '',
             '![Training components](training-components.png)', '',
             'All three training-ranked candidates are shown, including failures. The geometric-error reference line is the full fast-parent mean; the actual gate uses each candidate’s shared successful tracks. Required-window speed includes zero for missing windows. The yaw/steering panels describe observed windows and must be read with missing-window and completion counts in the CSV.', '',
             'The pilot starts from a previously selected g83 controller. Improvements here cannot establish independent training repeatability or generalization. Selection thresholds, training source, inputs and episode data remain frozen. Fresh shared testing follows controller freezing after protocol development.', '',
             'Regenerate from the project root:', '', '```powershell',
             '& .\\.venv\\Scripts\\python.exe research/report_quality_training.py', '```', '',
             'This command reads saved artifacts only. JSON records all numeric metrics, input hashes and the reporting-source hash. PNG and SVG versions of each figure are provided.']
    (output / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('candidate_metrics', 'input_sha256', 'references')}, indent=2))
    if args.archive:
        archive_completed(run, output)


if __name__ == '__main__':
    main()
