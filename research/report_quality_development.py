"""Read-only analysis of the frozen quality-speed study and shared PID baseline."""
from pathlib import Path
import json
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/neat-quality-speed-v1'
OUT = ROOT / 'results/neat-quality-speed-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    directory = RUN / 'baseline-development'
    summary = read(directory / 'summary.json')
    rows = {name: read(directory / (name+'.json'))['primary'] for name in summary['policies']}
    pid = {r['track_hash']: r for r in rows['pid-fullspeed-1m']}
    paired = {}
    for name in ('stable-neat-g83', 'fast-neat-g134'):
        shared = [r for r in rows[name] if r['completed'] and pid[r['track_hash']]['completed']]
        paired[name] = {'shared_finishes': len(shared)}
        for field in ('completion_time_s', 'geometry_lateral_rmse_m', 'command_jitter'):
            paired[name][field] = {'neat':mean(r[field] for r in shared),
                                  'pid':mean(pid[r['track_hash']][field] for r in shared)}
    report = {'scope':summary['scope'], 'raw_policy_summaries':summary['policies'],
              'shared_success_comparisons':paired,
              'failure_rows':{name:[r for r in data if not r['completed']] for name,data in rows.items()},
              'note':'This baseline is development evidence only. PID errors/jitter are compared on identical shared-success tracks separately from all-episode observed means.'}
    OUT.mkdir(exist_ok=True)
    (OUT/'shared-development-baseline.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    colors = {'mixed':'#007f86','speed':'#c45a28','corner':'#7756a0'}
    for ax, name in zip(axes, ('stable-neat-g83','fast-neat-g134')):
        shared = [r for r in rows[name] if r['completed'] and pid[r['track_hash']]['completed']]
        for group,color in colors.items():
            subset=[r for r in shared if r['group']==group]
            ax.scatter([pid[r['track_hash']]['completion_time_s'] for r in subset],
                       [r['completion_time_s'] for r in subset],color=color,label=group,alpha=.8)
        upper=max(max(r['completion_time_s'],pid[r['track_hash']]['completion_time_s']) for r in shared)*1.05
        ax.plot([0,upper],[0,upper],ls='--',color='#555',label='Equal time')
        ax.set(xlabel='Full-speed filtered PID finish time (s)',ylabel='NEAT finish time (s)',
               title=f'{name}: {len(shared)} shared successful tracks',xlim=(0,upper),ylim=(0,upper))
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    for extension in ('png','svg'):
        fig.savefig(OUT/f'baseline-paired-times.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    text=['# Shared development baseline before quality-speed evolution','',
          'Controllers and inputs were frozen before evaluation. All three controllers use the same 1 m/s cap, physical sensors, simulation and 60 development courses. This is not fresh final-test evidence.','',
          '| Controller | Completion | Observed mean chassis speed | Observed geometric RMSE | Command jitter |',
          '|---|---:|---:|---:|---:|']
    for name,item in summary['policies'].items():
        r=item['primary']
        text.append(f"| {name} | {r['completed']}/{r['courses']} | {r['mean_observed_chassis_speed_m_s']:.3f} m/s | {r['geometric_rmse_m']*1000:.2f} mm | {r['command_jitter']:.6f} |")
    text+=['','All-episode means include incomplete exposure and do not replace matched comparisons. PID completed 60/60 on its distinct tuning-validation suite, but failed one course off-route on this shared development suite. Both saved NEAT references completed 60/60 here.','',
           '## Shared-success comparison','',
           '| NEAT reference | Shared tracks | NEAT / PID mean time | NEAT / PID geometric RMSE | NEAT / PID jitter |',
           '|---|---:|---:|---:|---:|']
    for name,p in paired.items():
        t,e,j=(p[k] for k in ('completion_time_s','geometry_lateral_rmse_m','command_jitter'))
        text.append(f"| {name} | {p['shared_finishes']} | {t['neat']:.3f} / {t['pid']:.3f} s | {e['neat']*1000:.2f} / {e['pid']*1000:.2f} mm | {j['neat']:.6f} / {j['pid']:.6f} |")
    text+=['','![Paired completion times](baseline-paired-times.png)','',
           'Points below the diagonal favor NEAT time on that course; points above favor PID. Failed PID courses are excluded from time comparison and retained in the completion count/raw JSON. Course groups expose why high straight speed alone does not determine whole-route finish time.','',
           'The stable g83 controller is slower than PID on long straights, yet its mean paired completion time is smaller on the whole development mix. PID has substantially lower overall command jitter. These results do not justify blanket superiority claims and must be checked with fresh final tracks after protocol development.','',
           'The 0.5 m/s auxiliary evaluation transfers the 1 m/s PID gains; it is not labeled a separately tuned low-speed baseline. Frozen model hashes and all raw episodes are under `runs/neat-quality-speed-v1/baseline-development/`. `shared-development-baseline.json` includes the failure row and matched error/jitter denominators.','',
           'Reproduce with `& .\\.venv\\Scripts\\python.exe research/report_quality_development.py`. This reporting command starts no training or simulation.']
    (OUT/'shared-development-baseline.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Saved matched baseline report, numerical evidence and PNG/SVG paired-time figure.')


if __name__=='__main__':
    main()
