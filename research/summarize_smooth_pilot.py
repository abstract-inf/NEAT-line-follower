"""Report the completed seed-1 old-fitness pilot from saved logs; never evolve."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neat_smooth_study.metrics import quality


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def motion(episodes, kinds):
    output = {}
    for group in ('isolated', 'post_turn'):
        windows = [w for row in episodes if kinds[row['track_hash']] == group
                   for w in row['straight_windows']]
        observed = [w for w in windows if w['observed']]
        output[group] = {'intended_windows': len(windows), 'observed_windows': len(observed)}
        for key in ('speed_m_s', 'yaw_rms_rad_s', 'steering_rms'):
            values = [w[key] for w in observed]
            output[group][key] = {
                'mean': mean(values) if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/neat-smooth-study-v1')
    parser.add_argument('--branch', choices=('old',), default='old')
    args = parser.parse_args()
    run = args.run.resolve()
    directory = run / 'pilot/seed-01' / args.branch
    done = read(directory / 'complete.json')
    if done['signature']['branch'] != 'old' or done['signature']['seed'] != 1:
        raise ValueError('This report describes the seed-1 old-fitness control only.')
    protocol_sha = sha(run / 'prepared/protocol.json')
    if done['generations'] != 100 or done['signature']['protocol_sha256'] != protocol_sha:
        raise ValueError('A completed 100-generation pilot under the frozen protocol is required.')
    reference_path = run / 'pilot/references/seed-01.json'
    if sha(reference_path) != done['signature']['reference_sha256']:
        raise ValueError('Parent validation reference changed.')
    reference = read(reference_path)
    parent = {r['track_hash']: r for r in reference['primary']}
    suite = read(run / 'prepared/pilot/suite/suite.json')['splits']['validation']
    kinds = {entry['geometry_hash']: entry['kind'] for entry in suite}
    parent_motion = motion(reference['primary'], kinds)
    logs = csv_rows(directory / 'generations.csv')
    if [int(r['generation']) for r in logs] != list(range(100)):
        raise ValueError('Generation log is incomplete or contains duplicate generations.')
    inputs = {str(p.relative_to(run)): sha(p) for p in
              (directory / 'complete.json', directory / 'generations.csv', reference_path)}
    rows, counts = [], Counter()
    for generation in range(100):
        path = directory / 'validation' / f'generation-{generation:04d}.json'
        saved = read(path)
        gate = quality(saved['primary'], saved['retention'], reference['primary'], reference['retention'])
        if any(saved['quality'][key] != value for key, value in gate.items()):
            raise ValueError(f'Saved quality decision differs at generation {generation}.')
        shared = [r for r in saved['primary'] if r['completed'] and parent[r['track_hash']]['completed']]
        measured = motion(saved['primary'], kinds)
        row = {
            'generation': generation, **gate,
            'retention_completed': sum(r['completed'] for r in saved['retention']),
            'shared_jitter': mean(r['command_jitter'] for r in shared),
            'paired_parent_jitter': mean(parent[r['track_hash']]['command_jitter'] for r in shared),
            'paired_parent_rmse_m': mean(parent[r['track_hash']]['geometry_lateral_rmse_m'] for r in shared),
            'motion': measured,
            'drive_evaluated': saved['quality']['drive_evaluated'],
            'steering_evaluated': saved['quality']['steering_evaluated'],
        }
        row['shared_jitter_change_pct'] = 100 * (row['shared_jitter'] / row['paired_parent_jitter'] - 1)
        row['shared_rmse_change_pct'] = 100 * (row['geometric_rmse_m'] / row['paired_parent_rmse_m'] - 1)
        counts.update(gate['reasons'])
        rows.append(row)
        inputs[str(path.relative_to(run))] = sha(path)
    out = ROOT / 'results/neat-smooth-study-v1' / f'pilot-{args.branch}-100-generations'
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        'review_date': datetime.now(ZoneInfo('Asia/Amman')).date().isoformat(),
        'branch': args.branch, 'generations': 100,
        'training_episodes': sum(int(r['training_episodes']) for r in logs),
        'validation_episodes': sum(int(r['validation_episodes']) for r in logs),
        'logged_generation_wall_hours': sum(float(r['wall_seconds']) for r in logs) / 3600,
        'qualified_champions': sum(r['eligible'] for r in rows),
        'full_primary_completion_generations': sum(r['completed'] == 60 for r in rows),
        'retention_completion_range': [min(r['retention_completed'] for r in rows), max(r['retention_completed'] for r in rows)],
        'rejection_counts': dict(counts),
        'parent': {'motion': parent_motion,
                   'geometric_rmse_m': mean(r['geometry_lateral_rmse_m'] for r in reference['primary']),
                   'command_jitter': mean(r['command_jitter'] for r in reference['primary'])},
        'final_champion': rows[-1],
        'all_champions_both_outputs_evaluated': all(r['drive_evaluated'] and r['steering_evaluated'] for r in rows),
        'generations_data': rows, 'protocol_sha256': protocol_sha, 'input_sha256': inputs,
        'scope': 'One seed, fixed repeated validation, extra-training continuation. No new-fitness comparison or final test claim yet.',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    flat = [{**{k: v for k, v in row.items() if k != 'motion'},
             **{f'{group}_{key}_{stat}': value
                for group, data in row['motion'].items() for key, values in data.items()
                if isinstance(values, dict) for stat, value in values.items()}}
            for row in rows]
    with (out / 'champions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows({**r, 'reasons': ';'.join(r['reasons'])} for r in flat)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [r['generation'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    axes[0, 0].plot(x, [r['completed'] for r in rows], color='#007f86')
    axes[0, 0].axhline(60, color='#777', linestyle='--')
    axes[0, 0].set(ylabel='Primary completions / 60', ylim=(0, 61))
    axes[0, 1].plot(x, [100 * (r['paired_time_ratio'] - 1) for r in rows], color='#007f86')
    axes[0, 1].axhline(5, color='#c45a28', linestyle='--', label='Allowed +5%')
    axes[0, 1].axhline(0, color='#777', linestyle=':')
    axes[0, 1].set(ylabel='Paired lap-time change vs parent (%)')
    axes[0, 1].legend()
    for ax, key, label in (
            (axes[1, 0], 'shared_rmse_change_pct', 'Shared-finish geometric RMSE change (%)'),
            (axes[1, 1], 'shared_jitter_change_pct', 'Shared-finish command-jitter change (%)')):
        ax.plot(x, [r[key] for r in rows], color='#007f86')
        ax.axhline(0, color='#c45a28', linestyle='--', label='No degradation allowed')
        ax.set(ylabel=label)
        ax.legend()
    for ax in axes.flat:
        ax.set(xlabel='Additional generation')
        ax.grid(alpha=.2)
    fig.suptitle(f'{args.branch.capitalize()}-fitness continuation: repeated validation, not final test data')
    for extension in ('png', 'svg'):
        fig.savefig(out / f'01-validation-quality.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout='constrained')
    for group, label, color in (('isolated', 'Isolated straight', '#007f86'),
                                ('post_turn', 'Straight after turn/chicane', '#c45a28')):
        for ax, key, stat in ((axes[0], 'speed_m_s', 'min'),
                              (axes[1], 'yaw_rms_rad_s', 'max'),
                              (axes[2], 'steering_rms', 'max')):
            ax.plot(x, [r['motion'][group][key][stat] for r in rows], label=label, color=color)
    for ax, threshold, label in zip(axes, (.95, .15, .02),
                                   ('Minimum sustained speed (m/s)', 'Maximum raw yaw RMS (rad/s)', 'Maximum normalized steering RMS')):
        ax.axhline(threshold, color='#444', linestyle='--', label=f'Required threshold: {threshold}')
        ax.set(ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set(xlabel='Additional generation')
    fig.suptitle('Per-champion extremes expose failures hidden by averaging centered and recovery straights')
    for extension in ('png', 'svg'):
        fig.savefig(out / f'02-straight-gates.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    final = rows[-1]
    text = [f'# Seed-1 {args.branch}-fitness pilot: 100-generation review', '',
            '**Completed cleanly; no champion passed all predeclared gates.**' if not summary['qualified_champions'] else
            f"**Completed; {summary['qualified_champions']} champions qualified.**", '',
            f"Budget: {summary['training_episodes']:,} training episodes and {summary['validation_episodes']:,} validation episodes. Logged generation wall time: {summary['logged_generation_wall_hours']:.2f} hours; this excludes unlogged startup/overhead. Latest committed population generation: 100, corresponding to completed generation 99.", '',
            'This is the old-reward control using the new six-course profile, two workers, a target population of 100, and a fixed 1 m/s cap. These repeated validation courses select candidates; they are not untouched test evidence.', '',
            '## Parent versus final-generation champion', '',
            '| Metric | Original seed-1 g134 parent | Additional generation 99 |',
            '|---|---:|---:|',
            f"| Primary completion | 60/60 | {final['completed']}/60 |",
            f"| Isolated-straight mean speed | {parent_motion['isolated']['speed_m_s']['mean']:.3f} m/s | {final['motion']['isolated']['speed_m_s']['mean']:.3f} m/s |",
            f"| Post-turn straight mean speed | {parent_motion['post_turn']['speed_m_s']['mean']:.3f} m/s | {final['motion']['post_turn']['speed_m_s']['mean']:.3f} m/s |",
            f"| Post-turn straight mean yaw RMS | {parent_motion['post_turn']['yaw_rms_rad_s']['mean']:.3f} rad/s | {final['motion']['post_turn']['yaw_rms_rad_s']['mean']:.3f} rad/s |",
            f"| Geometric RMSE, shared finishes | {summary['parent']['geometric_rmse_m']*1000:.2f} mm | {final['geometric_rmse_m']*1000:.2f} mm |",
            f"| Command jitter, all primary episodes | {summary['parent']['command_jitter']:.6f} | {final['command_jitter']:.6f} |", '',
            f"The final champion took {100*(final['paired_time_ratio']-1):.2f}% longer on the 60 shared successful tracks, within the 5% allowance. Its improved average recovery yaw and geometric error do not establish acceptance: sustained speed, maximum per-window yaw/steering, and command jitter still fail. The final generation is an illustrative endpoint, not an officially selected best model.", '',
            '## Across the whole run', '',
            f"- {summary['full_primary_completion_generations']}/100 champions completed all 60 primary tracks. Every champion retained 20/20 low-speed completions.",
            '- Every champion failed at least one speed, yaw and steering window. One champion also had an unobserved required straight.',
            '- Generation 5 had the lowest mean yaw across all 20 windows, 0.0976 rad/s. Half the windows are perfectly centered isolated straights with effectively zero yaw. Its post-turn mean was 0.1953 and maximum 0.2959 rad/s, so a combined mean below 0.15 did not pass the per-window yaw rule. It also completed 59/60 and degraded error/jitter.',
            '- Generation 72 had the smallest shared-finish geometric RMSE (10.08 mm), but post-turn yaw averaged 1.698 rad/s, isolated speed was 0.802 m/s, and paired laps were 5.94% slower. Small geometric error alone did not ensure fast, smooth motion.',
            '- Both neural outputs were evaluated in every champion. This confirms evaluator execution status, not the cause of any remaining control behavior.', '',
            'Rejection counts overlap: ' + ', '.join(f'{key}: {value}' for key, value in counts.items()) + '.', '',
            '![Validation quality](01-validation-quality.png)', '',
            'Paired time/error/jitter use the candidate-specific shared-success tracks and matched parent values. Completion keeps all 60 trials. Curves show individual champions with no smoothing or confidence intervals.', '',
            '![Straight acceptance extremes](02-straight-gates.png)', '',
            'Each series shows the minimum speed or maximum yaw/steering among observed windows in that course group. Missing windows independently fail acceptance and remain recorded in raw results; they are not successful zero-motion measurements.', '',
            '## Interpretation and next step', '',
            'For this one seed and 100-generation budget, continued old-reward training on the new courses did not meet the combined target. Evolution found different speed/error/smoothness trade-offs. This does not establish that more generations can never help, that NEAT cannot solve the task, or that the proposed new reward will succeed.', '',
            'Next run the separately prepared smooth-fitness pilot, then review both completed branches under the same criteria. Do not extend this control or weaken its thresholds after observing its results.', '',
            '```powershell',
            '& .\\.venv\\Scripts\\python.exe run_neat_smooth_study.py pilot --branch smooth --workers 2',
            '& .\\.venv\\Scripts\\python.exe run_neat_smooth_study.py review-pilot',
            '```', '',
            'The second command requires both branches to finish; no confirmation or assessment is started by generating this report. Reproduce with:', '',
            '```powershell',
            f'& .\\.venv\\Scripts\\python.exe research/summarize_smooth_pilot.py --branch {args.branch}',
            '```', '',
            'All 100 champions remain under the original run `pilot/seed-01/old/champions/`. No qualifying `best.pkl` was manufactured. `summary.json` records all champion metrics and source hashes; `champions.csv` provides a flat table. Figures are supplied as PNG and SVG.']
    (out / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(f'Saved {out}: report, JSON, CSV and two PNG/SVG figure pairs. No simulations or evolution started.')


if __name__ == '__main__':
    main()
