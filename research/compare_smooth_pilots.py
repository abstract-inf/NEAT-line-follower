"""Compare completed seed-1 reward pilots without simulating or evolving."""
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from statistics import mean
from zoneinfo import ZoneInfo

from summarize_smooth_pilot import ROOT, read, sha, motion
from neat_smooth_study.metrics import quality, added_costs


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def load_branch(run, branch, reference, kinds):
    directory = run / 'pilot/seed-01' / branch
    done = read(directory / 'complete.json')
    logs = csv_rows(directory / 'generations.csv')
    signature = done['signature']
    if (signature['branch'] != branch or signature['seed'] != 1 or signature['phase'] != 'pilot'
            or signature['protocol_sha256'] != sha(run / 'prepared/protocol.json')
            or signature['parent_sha256'] != sha(run / 'prepared/inputs/seed-01.pkl')
            or signature['reference_sha256'] != sha(run / 'pilot/references/seed-01.json')):
        raise ValueError(f'{branch}: frozen ancestry or validation reference changed.')
    if done['generations'] != 100 or [int(r['generation']) for r in logs] != list(range(100)):
        raise ValueError(f'{branch} has not completed the 100-generation protocol.')
    if sum(int(r['training_episodes']) for r in logs) != done['training_episodes']:
        raise ValueError(f'{branch}: logged training budget disagrees with completion record.')
    parent = {r['track_hash']: r for r in reference['primary']}
    records, hashes = [], {}
    for generation in range(100):
        path = directory / 'validation' / f'generation-{generation:04d}.json'
        data = read(path)
        gate = quality(data['primary'], data['retention'], reference['primary'], reference['retention'])
        if any(data['quality'][key] != value for key, value in gate.items()):
            raise ValueError(f'{branch} generation {generation}: saved acceptance decision changed.')
        shared = [r for r in data['primary'] if r['completed'] and parent[r['track_hash']]['completed']]
        windows = [w for r in data['primary'] for w in r['straight_windows']]
        observed = [w for w in windows if w['observed']]
        records.append({
            **gate, 'branch': branch, 'generation': generation,
            'retention_completed': sum(r['completed'] for r in data['retention']),
            'motion': motion(data['primary'], kinds),
            'minimum_straight_speed_m_s': min(w['speed_m_s'] for w in observed),
            'maximum_straight_yaw_rms_rad_s': max(w['yaw_rms_rad_s'] for w in observed),
            'maximum_straight_steering_rms': max(w['steering_rms'] for w in observed),
            'missing_windows': len(windows) - len(observed),
            'straight_motion_pass': not bool(set(gate['reasons']) &
                {'straight_yaw', 'straight_steering', 'unfinished_straight', 'no_required_straights'}),
            'shared_rmse_ratio': gate['geometric_rmse_m'] / mean(parent[r['track_hash']]['geometry_lateral_rmse_m'] for r in shared),
            'shared_jitter_ratio': mean(r['command_jitter'] for r in shared) / mean(parent[r['track_hash']]['command_jitter'] for r in shared),
            'mean_old_score': mean(r['base_score'] for r in data['primary']),
            'mean_reward_score': mean(r['score'] for r in data['primary']),
            'components': {key: mean(r['components'][key] for r in data['primary'])
                           for key in data['primary'][0]['components']},
        })
        hashes[str(path.relative_to(run))] = sha(path)
    if bool(sum(r['eligible'] for r in records)) != done['qualified']:
        raise ValueError(f'{branch}: completion record disagrees with recomputed quality gates.')
    return {
        'completion': done, 'generations': records,
        'training_episodes': sum(int(r['training_episodes']) for r in logs),
        'validation_episodes': sum(int(r['validation_episodes']) for r in logs),
        'logged_generation_wall_hours': sum(float(r['wall_seconds']) for r in logs) / 3600,
        'runtime_sessions': csv_rows(directory / 'runtime.csv'),
        'full_completion_champions': sum(r['completed'] == 60 for r in records),
        'straight_motion_pass_champions': sum(r['straight_motion_pass'] for r in records),
        'qualified_champions': sum(r['eligible'] for r in records),
        'rejection_counts': dict(Counter(reason for r in records for reason in r['reasons'])),
        'input_hashes': {**hashes, str((directory / 'complete.json').relative_to(run)): sha(directory / 'complete.json'),
                         str((directory / 'generations.csv').relative_to(run)): sha(directory / 'generations.csv')},
    }


def main():
    run = ROOT / 'runs/neat-smooth-study-v1'
    out = ROOT / 'results/neat-smooth-study-v1/pilot-comparison'
    review = read(run / 'pilot/review.json')
    if not review['matched_tracks_verified'] or review['protocol_sha256'] != sha(run / 'prepared/protocol.json'):
        raise ValueError('Run the frozen review-pilot command before creating this report.')
    reference = read(run / 'pilot/references/seed-01.json')
    kinds = {e['geometry_hash']: e['kind']
             for e in read(run / 'prepared/pilot/suite/suite.json')['splits']['validation']}
    branches = {name: load_branch(run, name, reference, kinds) for name in ('old', 'smooth')}
    for name, branch in branches.items():
        if review['branches'][name]['complete_sha256'] != sha(run / 'pilot/seed-01' / name / 'complete.json'):
            raise ValueError('Pilot completion changed after review.')
    parent_motion = motion(reference['primary'], kinds)
    parent_old_score = mean(r['score'] for r in reference['primary'])
    parent_smooth_score = mean(r['score'] + sum(added_costs(r['straight_windows'], r['lateral_rmse_m']).values())
                               for r in reference['primary'])
    out.mkdir(parents=True, exist_ok=True)
    selected = {'old-final-g0099': ('old', 99), 'smooth-tradeoff-g0083': ('smooth', 83),
                'smooth-final-g0099': ('smooth', 99)}
    frozen = {}
    for label, (branch, generation) in selected.items():
        source = run / 'pilot/seed-01' / branch / 'champions' / f'generation-{generation:04d}.pkl'
        target = out / 'frozen' / (label + '.pkl')
        target.parent.mkdir(exist_ok=True)
        if target.exists() and sha(target) != sha(source):
            raise ValueError('An existing illustrative snapshot changed.')
        shutil.copy2(source, target)
        frozen[label] = {'file': target.relative_to(out).as_posix(), 'sha256': sha(target),
                         'source': str(source.relative_to(ROOT)), 'generation': generation, 'branch': branch,
                         'purpose': 'Illustrative validation trade-off; not an officially qualified winner.'}
    parent_source = run / 'prepared/inputs/seed-01.pkl'
    parent_target = out / 'frozen/parent-native-1m-g0134.pkl'
    if parent_target.exists() and sha(parent_target) != sha(parent_source):
        raise ValueError('Existing frozen parent changed.')
    shutil.copy2(parent_source, parent_target)
    frozen['parent'] = {'file': 'frozen/parent-native-1m-g0134.pkl', 'sha256': sha(parent_source),
                        'purpose': 'Original seed-1 reference.'}
    (out / 'frozen/manifest.json').write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
    summary = {
        'review_date': datetime.now(ZoneInfo('Asia/Amman')).date().isoformat(),
        'review': review, 'parent_motion': parent_motion,
        'parent_geometric_rmse_m': mean(r['geometry_lateral_rmse_m'] for r in reference['primary']),
        'parent_command_jitter': mean(r['command_jitter'] for r in reference['primary']),
        'parent_mean_validation_old_score': parent_old_score,
        'parent_mean_validation_rescored_smooth_score': parent_smooth_score,
        'branches': branches, 'illustrative_frozen_models': frozen,
        'scope': 'One paired seed-1 development pilot. Repeated validation, no final test tracks; no algorithm-wide effect estimate.',
        'runtime_caveat': 'Worker counts differ: old used 2; smooth used 4, then 6, then 8. Wall-time difference is not an isolated reward-efficiency result.',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    flat = [{k: v for k, v in row.items() if k not in ('motion', 'components')}
            for branch in branches.values() for row in branch['generations']]
    with (out / 'champions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows({**r, 'reasons': ';'.join(r['reasons'])} for r in flat)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'old': '#007f86', 'smooth': '#c45a28'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
    for name, branch in branches.items():
        rows = branch['generations']; x = [r['generation'] for r in rows]
        for ax, key in zip(axes.flat, ('completed', 'minimum_straight_speed_m_s', 'shared_rmse_ratio', 'shared_jitter_ratio')):
            ax.plot(x, [r[key] for r in rows], color=colors[name], label=name)
    labels = ('Primary completion / 60', 'Minimum observed straight speed (m/s)',
              'Shared-finish geometric RMSE / parent', 'Shared-finish command jitter / parent')
    for ax, label, threshold in zip(axes.flat, labels, (60, .95, 1, 1)):
        ax.set(xlabel='Additional generation', ylabel=label)
        ax.axhline(threshold, color='#555', linestyle='--')
        ax.grid(alpha=.2); ax.legend()
    axes[0, 0].set_ylim(0, 61)
    fig.suptitle('Paired development pilot: smaller tracking error, but the speed target remains unmet')
    for extension in ('png', 'svg'):
        fig.savefig(out / f'01-matched-validation.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    for name, branch in branches.items():
        for completed, marker, label in ((True, 'o', f'{name}: 60/60 completed'),
                                          (False, 'x', f'{name}: incomplete course(s)')):
            rows = [r for r in branch['generations'] if (r['completed'] == 60) == completed]
            ax.scatter([r['minimum_straight_speed_m_s'] for r in rows],
                       [r['maximum_straight_yaw_rms_rad_s'] for r in rows],
                       color=colors[name], marker=marker, alpha=.65, s=28, label=label)
    ax.scatter(min(parent_motion[k]['speed_m_s']['min'] for k in parent_motion),
               max(parent_motion[k]['yaw_rms_rad_s']['max'] for k in parent_motion),
               marker='*', color='#333', s=180, label='Original parent')
    ax.axvline(.95, color='#555', linestyle='--'); ax.axhline(.15, color='#555', linestyle='--')
    ax.set(xlabel='Minimum observed straight speed (m/s)', ylabel='Maximum straight yaw RMS (rad/s)',
           yscale='log', title='The smooth pilot reaches low yaw, but no champion reaches the required speed')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, .5)); ax.grid(alpha=.2)
    for extension in ('png', 'svg'):
        fig.savefig(out / f'02-speed-smoothness.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    for key, label in (('straight_yaw_term', 'Straight yaw'), ('straight_steering_term', 'Straight steering'),
                       ('tracking_term', 'Route-relative training error'), ('straight_speed_deficit_term', 'Straight speed deficit')):
        rows = branches['smooth']['generations']
        ax.plot([r['generation'] for r in rows], [r['components'][key] for r in rows], label=label)
    ax.set(xlabel='Additional generation', ylabel='Mean contribution to primary validation fitness',
           title='Smooth reward: physical-motion costs shrink; speed-deficit cost remains small')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    for extension in ('png', 'svg'):
        fig.savefig(out / f'03-fitness-components.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    old = branches['old']; smooth = branches['smooth']; tradeoff = smooth['generations'][83]
    final = smooth['generations'][99]
    text = [
        '# Completed seed-1 old versus smooth reward pilot', '',
        '**Both 100-generation branches completed. Neither produced a champion meeting all predeclared criteria. Five-lineage confirmation is blocked.**', '',
        'The frozen review verified identical parents, initial evolution seed, references and all 100 six-track generation batches. Target population and generation budgets matched. Actual training episodes differed slightly because stock NEAT population sizes can fluctuate.', '',
        '| Outcome | Old fitness | Smooth fitness |', '|---|---:|---:|',
        f"| Training episodes | {old['training_episodes']:,} | {smooth['training_episodes']:,} |",
        f"| Validation episodes | {old['validation_episodes']:,} | {smooth['validation_episodes']:,} |",
        f"| Champions completing 60/60 primary tracks | {old['full_completion_champions']}/100 | {smooth['full_completion_champions']}/100 |",
        f"| Champions passing straight yaw + steering gates | {old['straight_motion_pass_champions']}/100 | {smooth['straight_motion_pass_champions']}/100 |",
        f"| Fully qualifying champions | {old['qualified_champions']} | {smooth['qualified_champions']} |",
        f"| Logged generation wall time | {old['logged_generation_wall_hours']:.2f} h | {smooth['logged_generation_wall_hours']:.2f} h |", '',
        'All 200 champions completed all 20 low-speed retention courses. Wall times exclude unlogged startup/repeated interrupted work. The old run used two workers; smooth used four, then six, then eight, so these wall times do not isolate fitness-related compute efficiency.', '',
        '## A useful trade-off example: smooth generation 83', '',
        '| Metric | Original parent | Smooth g83 |', '|---|---:|---:|',
        f"| Primary completion | 60/60 | {tradeoff['completed']}/60 |",
        f"| Isolated straight speed | {parent_motion['isolated']['speed_m_s']['mean']:.3f} m/s | {tradeoff['motion']['isolated']['speed_m_s']['mean']:.3f} m/s |",
        f"| Post-turn straight speed | {parent_motion['post_turn']['speed_m_s']['mean']:.3f} m/s | {tradeoff['motion']['post_turn']['speed_m_s']['mean']:.3f} m/s |",
        f"| Post-turn mean yaw RMS | {parent_motion['post_turn']['yaw_rms_rad_s']['mean']:.3f} rad/s | {tradeoff['motion']['post_turn']['yaw_rms_rad_s']['mean']:.3f} rad/s |",
        f"| Shared-finish geometric RMSE | {summary['parent_geometric_rmse_m']*1000:.2f} mm | {tradeoff['geometric_rmse_m']*1000:.2f} mm |",
        f"| Whole-course command jitter | {summary['parent_command_jitter']:.6f} | {tradeoff['command_jitter']:.6f} |", '',
        f"Generation 83 also retained 20/20 low-speed completions and passed every primary yaw and steering window. Its only rejection reasons were straight speed and paired time: laps were {100*(tradeoff['paired_time_ratio']-1):.2f}% slower, exceeding the 5% allowance. It is an exploratory example chosen after reviewing the rejection list, not an official best model or a held-out test result.", '',
        'The final smooth champion (g99) also completed 60/60 and 20/20, with 7.35 mm geometric RMSE and about 0.715 m/s sustained speed. Post-turn yaw averaged 0.016 rad/s. Whole-course jitter was 0.008996 (about 38% above the parent), and paired laps were 12.45% slower. The final generation is not the best observed trade-off.', '',
        '## What the reward is doing', '',
        'The smooth reward produced many controllers passing the physical straight-motion requirements in this single development lineage. Speed remains the limiting requirement: every champion in both branches failed at least one required speed window.', '',
        'At 0.715 m/s, the additional speed-deficit penalty is approximately -0.0061 on an eligible straight course. The bounded yaw and steering terms can together approach -0.30. The original completion-speed bonus remains in the fitness. These magnitudes suggest that the new weighting can accept substantial speed loss in exchange for physical smoothness; they do not prove a unique cause or global optimum.', '',
        f"Rescoring the same stored parent validation episodes under the smooth formula gives mean score {parent_smooth_score:.6f}, versus the final smooth champion's {final['mean_reward_score']:.6f}. Under the old formula, the corresponding scores are {parent_old_score:.6f} and {final['mean_old_score']:.6f}. Thus the new formula rewards this slower final controller more highly on these validation episodes even though it fails the combined acceptance target. This is a diagnostic rescore of saved episodes, not additional physics or a breeding-fitness comparison: reproduction uses the six-course mean/CVaR aggregation.", '',
        '## Figures', '',
        '![Matched validation outcomes](01-matched-validation.png)', '',
        'Curves show all 100 champions per branch. Shared-finish error/jitter ratios use each candidate-specific matched parent subset. Raw repeated validation is selection evidence, not independent test repetitions.', '',
        '![Speed versus physical smoothness](02-speed-smoothness.png)', '',
        'Points summarize minimum observed speed and maximum raw yaw across required windows. A missing window separately fails acceptance and is not treated as success. The old run has one such missing-window champion. Steering gates and time/error/jitter remain independent checks; occupying a favorable speed/yaw region alone would not establish qualification.', '',
        '![Smooth fitness components](03-fitness-components.png)', '',
        'Components are mean primary-validation contributions. Tracking cost uses route-relative RMSE, not the independent geometric RMSE used in acceptance. The old branch has no added smoothness costs by definition; zero omitted costs would not mean zero oscillation.', '',
        '## Frozen illustrative models and next step', '',
        'The `frozen/` directory contains exact copies of the original parent, old g99, smooth g83 and smooth g99, with a SHA-256 manifest. None is relabeled as a qualifying study winner. Original checkpoint, logs and all 100 champions per branch remain in the live run archive.', '',
        'Do not launch the five confirmation lineages under this unsuccessful pilot, relax gates after seeing the outcome, or present g83 as meeting the 1 m/s goal. A separately versioned follow-up should address the speed/smoothness balance and be compared against these preserved controls. Additional generations alone have not been shown to solve it.', '',
        'This comparison is one development lineage with shared validation courses. It supports descriptive conclusions about these runs, not a general superiority claim for NEAT, an algorithm-wide causal effect of reward shaping, or hardware performance. No final test tracks were opened and no new evolution was started during review.', '',
        'Reproduce after both branches finish:', '', '```powershell',
        '& .\\.venv\\Scripts\\python.exe run_neat_smooth_study.py review-pilot',
        '& .\\.venv\\Scripts\\python.exe research/compare_smooth_pilots.py', '```', '',
        '`summary.json` contains all champion metrics, reward components and source hashes. `champions.csv` is the flat comparison table. All figures are supplied as PNG and SVG. These new reports/models are local until explicitly committed and pushed.',
    ]
    (out / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(f'Saved {out}: paired report, JSON/CSV, frozen illustrative models and three PNG/SVG figure pairs.')


if __name__ == '__main__':
    main()
