"""Reproducible plots, fixed-controller assessment tables and guarded showcases."""
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from .common import read, write_json, csv_rows, digest, verify, lock
from .metrics import select
from .workflow import freeze_final, confirmation_branch


def report(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);verify(run)
    out=run/'paper-figures';out.mkdir(exist_ok=True)
    figures=[]
    def save(fig,name,caption):
        for ext in ('png','svg'):fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig);figures.append({'file':name,'caption':caption})
    logs=sorted(p for phase in ('pilot','confirmation') for p in (run/phase).rglob('generations.csv'))
    if logs:
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for path in logs:
            rows=csv_rows(path);label='/'.join(path.parent.relative_to(run).parts)
            axes[0].plot([int(r['generation']) for r in rows],[float(r['training_best']) for r in rows],label=label)
            axes[1].plot([int(r['generation']) for r in rows],[float(r['validation_completion_rate']) for r in rows],label=label)
        axes[0].set(xlabel='Additional evolution generation',ylabel='Training champion fitness')
        axes[1].set(xlabel='Additional evolution generation',ylabel='Repeated validation completion')
        axes[0].legend(fontsize=6)
        save(fig,'01-learning','Repeated validation is selection data, not held-out test evidence. Old/new rewards have different numerical scales; compare physical quality and completion, not raw scores between rewards.')
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        for path in logs:
            rows=[r for r in csv_rows(path.parent/'fitness-components.csv') if r['role']=='training']
            for component in ('straight_yaw_term','straight_steering_term','tracking_term','straight_speed_deficit_term'):
                ax.plot([int(r['generation']) for r in rows],[float(r[component]) for r in rows],label=f"{path.parent.relative_to(run)} / {component}")
        ax.set(xlabel='Additional evolution generation',ylabel='Mean contribution to episode fitness')
        ax.legend(fontsize=5,ncol=2)
        save(fig,'02-fitness-components','Bounded added costs are shown separately. Zero costs in the old-fitness branch do not imply zero physical motion; consult validation JSON for raw metrics.')
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for path in logs:
            quality=csv_rows(path.parent/'quality.csv');x=[int(r['generation']) for r in quality]
            for ax,key in zip(axes,('straight_yaw_rms','geometric_rmse_m','paired_time_ratio')):
                ax.plot(x,[float(r[key]) if r[key] else np.nan for r in quality],label=str(path.parent.relative_to(run)))
        for ax,label in zip(axes,('Straight physical yaw RMS (rad/s)','Shared-finish geometric RMSE (m)','Paired lap-time ratio versus parent')):
            ax.set(xlabel='Additional evolution generation',ylabel=label)
        axes[0].axhline(.15,color='#777',ls='--');axes[2].axhline(1.05,color='#777',ls='--')
        axes[0].legend(fontsize=6)
        save(fig,'02b-validation-physical-quality','Raw physical metrics expose speed/smoothness trade-offs across every champion, including rejected champions. Passing a mean curve alone does not establish that each required window passed its gate.')
    rows=[read(p) for p in sorted((run/'assessment/episodes').glob('*.json'))]
    if rows:
        # Keep all failures in the exported evidence, never substitute failure time for lap time.
        with (run/'assessment/results.csv').open('w',newline='',encoding='utf-8') as stream:
            fields=list(rows[0]);writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
        summaries=[]
        for cap in (1.,.5,1.5):
            subset=[r for r in rows if r['cap_m_s']==cap]
            for policy in sorted({r['policy'] for r in subset}):
                episodes=[r for r in subset if r['policy']==policy]
                windows=[w for r in episodes for w in r['straight_windows'] if w['observed']]
                summaries.append({'cap_m_s':cap,'policy':policy,'tracks':len(episodes),'completed':sum(r['completed'] for r in episodes),
                    'mean_observed_speed_m_s':mean(r['average_forward_speed_m_s'] for r in episodes),
                    'peak_speed_m_s':max(r['peak_forward_speed_m_s'] for r in episodes),
                    'observed_geometric_rmse_m':mean(r['geometry_lateral_rmse_m'] for r in episodes),
                    'observed_command_jitter':mean(r['command_jitter'] for r in episodes),
                    'mean_straight_speed_m_s':mean(w['speed_m_s'] for w in windows) if windows else None,
                    'mean_straight_yaw_rms_rad_s':mean(w['yaw_rms_rad_s'] for w in windows) if windows else None,
                    'observed_straight_windows':len(windows),
                    'failures':{reason:sum(r['termination']==reason for r in episodes) for reason in sorted({r['termination'] for r in episodes if not r['completed']})}})
        paired=[]
        for cap in (1.,.5,1.5):
            pid={r['track_hash']:r for r in rows if r['cap_m_s']==cap and r['policy']==f'pid-{round(cap*100):03d}'}
            for seed in range(1,6):
                new=[r for r in rows if r['cap_m_s']==cap and r['policy']==f'new-s{seed}']
                reference={r['track_hash']:r for r in rows if r['cap_m_s']==cap and r['policy']==f'reference-s{seed}'}
                for label,baseline in (('pid',pid),('reference',reference)):
                    shared=[r for r in new if r['completed'] and baseline[r['track_hash']]['completed']]
                    paired.append({'seed':seed,'cap_m_s':cap,'baseline':label,'shared_finishes':len(shared),
                        'new_mean_time_s':mean(r['completion_time_s'] for r in shared) if shared else None,
                        'baseline_mean_time_s':mean(baseline[r['track_hash']]['completion_time_s'] for r in shared) if shared else None,
                        'new_rmse_m':mean(r['geometry_lateral_rmse_m'] for r in shared) if shared else None,
                        'baseline_rmse_m':mean(baseline[r['track_hash']]['geometry_lateral_rmse_m'] for r in shared) if shared else None})
        write_json(run/'assessment/summary.json',{'policies':summaries,'paired':paired,'complete':(run/'assessment/complete.json').exists(),
            'scope':'1 m/s primary; .5 and 1.5 transfer. Partial-episode quality means have unequal exposure. Shared tracks/seeds are not independent pooled trials.'})
        primary=[r for r in summaries if r['cap_m_s']==1.]
        fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
        ax.bar([r['policy'] for r in primary],[r['completed']/r['tracks'] for r in primary],color=['#007f86' if r['policy'].startswith('new') else '#a5b1ba' if r['policy'].startswith('reference') else '#c45a28' for r in primary])
        ax.set(ylabel='Completion fraction',title='Fresh shared test · primary 1 m/s cap',ylim=(0,1.05))
        ax.tick_params(axis='x',rotation=40)
        save(fig,'03-completion','Each frozen policy sees the same fresh test tracks. Across-seed spread describes the five resulting warm-started controllers, not independent track repetitions.')
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        selected=[r for r in paired if r['cap_m_s']==1. and r['baseline']=='pid']
        for r in selected:
            if r['shared_finishes']:
                reduction=100*(1-r['new_mean_time_s']/r['baseline_mean_time_s'])
                ax.bar(r['seed'],reduction,color='#007f86');ax.text(r['seed'],reduction,f"n={r['shared_finishes']}",ha='center',va='bottom')
        ax.axhline(0,color='#555');ax.set(xlabel='Seed',ylabel='Paired lap-time reduction versus PID (%)')
        save(fig,'04-paired-time','Only tracks both fixed policies complete enter lap-time comparison; each denominator is printed. Failures remain in completion and raw-result tables.')
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        for r in primary:
            if r['mean_straight_speed_m_s'] is not None:
                ax.scatter(r['mean_straight_speed_m_s'],r['mean_straight_yaw_rms_rad_s'],label=r['policy'])
        ax.axvline(.95,color='#777',ls='--');ax.axhline(.15,color='#777',ls='--')
        ax.set(xlabel='Observed mean straight chassis speed (m/s)',ylabel='Mean straight physical yaw RMS (rad/s)');ax.legend(fontsize=7)
        save(fig,'05-speed-smoothness','Speed and physical yaw are measured separately. Aggregate observed windows do not establish that every required validation or test window passes; raw rows retain all window values and missing flags.')
        text=['# Smooth NEAT study assessment','','All figures are descriptive results of frozen controllers. Primary cap: 1 m/s; other caps are transfer tests.','','| Policy | Cap | Completed | Observed RMSE (mm) | Command jitter | Mean / peak observed speed |','|---|---:|---:|---:|---:|---:|']
        text += [f"| {r['policy']} | {r['cap_m_s']:.2f} | {r['completed']}/{r['tracks']} | {r['observed_geometric_rmse_m']*1000:.2f} | {r['observed_command_jitter']:.6f} | {r['mean_observed_speed_m_s']:.3f} / {r['peak_speed_m_s']:.3f} |" for r in summaries]
        text+=['','See assessment/summary.json for paired times and their denominators, failures and straight-window metrics. Results.csv retains every episode. Do not treat partial-episode error means as equal-exposure whole-lap comparisons. No hardware performance claim is supported.']
        (run/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    if not logs and not rows:raise ValueError('Nothing has been trained or assessed; plots do not launch either operation')
    write_json(out/'captions.json',{'figures':figures})
    print(f'Generated {len(figures)} PNG/SVG figure pairs without running evolution or assessment.')


def record(run,cap=None):
    """Select with validation only. Never silently render an unqualified transfer."""
    from .media import record_pair
    run=Path(run);verify(run)
    cap=1. if cap is None else cap
    if cap==1.5:raise ValueError('No full-speed 1.5 m/s NEAT study has been qualified. Review the 1 m/s milestone before designing it; historical transfer clips are not final showcases.')
    manifest=freeze_final(run)
    branch=confirmation_branch(run)
    candidates=[]
    for seed in range(1,6):
        directory=run/'confirmation'/f'seed-{seed:02d}'/branch
        selection=read(directory/'selection.json')
        validation=read(directory/'validation'/f"generation-{selection['generation']:04d}.json")
        if cap==.5:
            windows=[w for r in validation['retention'] for w in r['straight_windows']]
            if not windows or any(not w['observed'] or w['speed_m_s']<.475 or w['yaw_rms_rad_s']>.15 or w['steering_rms']>.02 for w in windows):continue
        candidates.append({**selection,'seed':seed})
    winner=select(candidates)
    if winner is None:raise ValueError('No validation-qualified controller at this showcase cap')
    names=[f"new-s{winner['seed']}",f'pid-{round(cap*100):03d}']
    directory=run/'showcase'/f'cap-{round(cap*100):03d}'
    with lock(directory):
        record_pair(run,manifest,names,cap,directory)
