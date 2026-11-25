"""Freeze first, assess once on fresh shared courses, and report without training."""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/neat-pid-finalization-20261125'
OUT = ROOT / 'results/neat-pid-finalization-20261125'
ENGINE = ROOT / 'runs/neat-joint-study-v1/prepared/source'
sys.path.insert(0, str(ENGINE))
from neat_smooth_study.common import read, write_json, digest, dependencies
from neat_smooth_study.courses import make_suite, protected_geometry
from neat_smooth_study.engine import initialize, collect


def prepare():
    sources = {
        'neat-prototype.pkl': ROOT / 'runs/neat-joint-study-v1/development/joint/champions/generation-0049-rank-1.pkl',
        'pid-100.json': ROOT / 'runs/pid-quality-tuning-v1/cap-100/pid-fullspeed.json',
        'pid-050.json': ROOT / 'runs/pid-quality-tuning-v1/cap-050/pid-fullspeed.json',
    }
    frozen = RUN / 'frozen'; frozen.mkdir(parents=True, exist_ok=True)
    manifest = frozen / 'manifest.json'
    if manifest.exists():
        saved = read(manifest)
        for name, record in saved['controllers'].items():
            if digest(frozen / name) != record['sha256']:
                raise ValueError('Frozen controller changed.')
        if saved['dependencies'] != dependencies():
            raise ValueError('Dependencies changed.')
        return
    if any((RUN / 'episodes').glob('*.json')):
        raise ValueError('Cannot freeze after evaluation.')
    for name, source in sources.items():
        shutil.copy2(source, frozen / name)
    for cap in ('050', '100'):
        selection = read(ROOT / f'runs/pid-quality-tuning-v1/cap-{cap}/selection-freeze.json')
        if not selection['eligible'] or selection['sha256'] != digest(frozen / f'pid-{cap}.json'):
            raise ValueError('PID is not the validated full-speed selection.')
        if read(frozen / f'pid-{cap}.json')['base'] != 1.:
            raise ValueError('PID base must permit full cruise.')
    source_files = read(ROOT / 'runs/neat-joint-study-v1/prepared/protocol.json')['source_files']
    if any(digest(ENGINE / name) != expected for name, expected in source_files.items()):
        raise ValueError('Frozen evaluation source changed.')
    # Commit controllers and the test rule BEFORE generating any new geometry.
    write_json(manifest, {'scope': 'Exploratory fresh-test comparison of a validation-ineligible NEAT prototype, one ancestry; no official winner or five-seed confirmation.',
        'controllers': {name: {'source': str(path), 'sha256': digest(frozen / name)} for name, path in sources.items()},
        'dependencies': dependencies(), 'engine_source_files': source_files,
        'report_source_sha256': digest(Path(__file__)),
        'neat_selection': 'Fixed joint g49 rank1: all60+20 complete, speed/yaw/steering/time pass; geometric RMSE and jitter fail. No test-based selection.',
        'test_rule': {'namespace': 'finalization-heldout-20261125', 'mixed':90, 'speed':30, 'corner':30, 'caps':[.5,1.]},
        'pid_150': 'No eligible policy; excluded from this validated-baseline comparison and retained as failed tuning evidence.'})


def evaluate(workers):
    prepare()
    suite_path = RUN / 'suite/suite.json'
    if not suite_path.exists():
        used = protected_geometry(ROOT)
        suite = make_suite(RUN / 'suite', 'finalization-heldout-20261125',
                           {'test': {'mixed':90, 'speed':30, 'corner':30}}, used)
        write_json(RUN / 'suite-freeze.json', {'suite_sha256':digest(suite_path),
            'tracks':{e['path']:digest(RUN/'suite'/e['path']) for e in suite['splits']['test']},
            'controllers_frozen_before_generation':digest(RUN/'frozen/manifest.json')})
    freeze = read(RUN / 'suite-freeze.json')
    if digest(suite_path) != freeze['suite_sha256'] or any(digest(RUN/'suite'/p)!=h for p,h in freeze['tracks'].items()):
        raise ValueError('Fresh test inputs changed.')
    if digest(RUN/'frozen/manifest.json') != freeze['controllers_frozen_before_generation']:
        raise ValueError('Controller manifest changed after test generation.')
    entries = read(suite_path)['splits']['test']
    (RUN / 'episodes').mkdir(exist_ok=True)
    context_run = ROOT / 'runs/neat-joint-study-v1'
    with mp.get_context('spawn').Pool(workers, initialize, (str(context_run), None), maxtasksperchild=128) as pool:
        for cap in (1., .5):
            for controller in ('neat', 'pid'):
                name = f'{controller}-{round(cap*100):03d}'
                target = RUN / 'episodes' / (name+'.json')
                if target.exists():
                    saved=read(target)
                    if saved['controllers_sha256']!=freeze['controllers_frozen_before_generation'] or saved['suite_sha256']!=freeze['suite_sha256']:
                        raise ValueError('Stale assessment result.')
                    continue
                policy = {'model':str(RUN/'frozen/neat-prototype.pkl')} if controller=='neat' else {'parameters':read(RUN/f'frozen/pid-{round(cap*100):03d}.json')}
                tasks=[]
                for start in range(0,len(entries),4):
                    chunk=entries[start:start+4]
                    tasks.append({**policy,'cap':cap,'detailed':True,'branch':'old',
                        'paths':[str(RUN/'suite'/e['path']) for e in chunk],
                        'groups':{e['geometry_hash']:e['group'] for e in chunk}})
                rows=[r for batch in collect(pool,tasks,name) for r in batch]
                if len(rows)!=150 or [r['track_hash'] for r in rows]!=[e['geometry_hash'] for e in entries] or any(r['cap_m_s']!=cap for r in rows):
                    raise ValueError('Assessment geometry/cap/order mismatch.')
                write_json(target, {'controllers_sha256':freeze['controllers_frozen_before_generation'],
                    'suite_sha256':freeze['suite_sha256'],'rows':rows})
    write_json(RUN/'complete.json',{'episode_records':600,'caps':[.5,1.],
        'results':{p.name:digest(p) for p in (RUN/'episodes').glob('*.json')}})


def summarize(rows):
    windows=[w for r in rows for w in r['straight_windows']]
    seen=[w for w in windows if w['observed']]
    return {'completed':sum(r['completed'] for r in rows),'courses':len(rows),
        'failures':dict(Counter(r['termination'] for r in rows if not r['completed'])),
        'mean_progress':mean(r['progress'] for r in rows),
        'mean_episode_average_speed_m_s':mean(r['average_forward_speed_m_s'] for r in rows),
        'peak_chassis_speed_m_s':max(r['peak_forward_speed_m_s'] for r in rows),
        'mean_episode_geometric_rmse_m':mean(r['geometry_lateral_rmse_m'] for r in rows),
        'mean_command_jitter':mean(r['command_jitter'] for r in rows),
        'minimum_required_straight_speed_m_s':min(w['speed_m_s'] if w['observed'] else 0 for w in windows),
        'missing_windows':len(windows)-len(seen),
        'maximum_observed_straight_yaw_rms':max(w['yaw_rms_rad_s'] for w in seen)}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    raw={p.stem:read(p)['rows'] for p in (RUN/'episodes').glob('*.json')}
    summaries={k:summarize(v) for k,v in raw.items()}
    pairs={}
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,cap in zip(axes,(1.,.5)):
        suffix=f'{round(cap*100):03d}';a={r['track_hash']:r for r in raw['neat-'+suffix]};b={r['track_hash']:r for r in raw['pid-'+suffix]}
        shared=[h for h in a if a[h]['completed'] and b[h]['completed']]
        pairs[suffix]={'shared_successes':len(shared),'total_courses':150,
            **{f:{'neat':mean(a[h][f] for h in shared),'pid':mean(b[h][f] for h in shared)} for f in ('completion_time_s','geometry_lateral_rmse_m','command_jitter')}}
        for group,color in (('mixed','#007f86'),('speed','#c45a28'),('corner','#7756a0')):
            hs=[h for h in shared if a[h]['group']==group]
            ax.scatter([b[h]['completion_time_s'] for h in hs],[a[h]['completion_time_s'] for h in hs],s=20,color=color,label=group,alpha=.7)
        upper=max(max(a[h]['completion_time_s'],b[h]['completion_time_s']) for h in shared)*1.05
        ax.plot([0,upper],[0,upper],'--',color='#555');ax.set(xlabel='Tuned PID finish time (s)',ylabel='NEAT prototype finish time (s)',title=f'{cap:.1f} m/s cap: {len(shared)}/150 shared finishes',xlim=(0,upper),ylim=(0,upper));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Fresh shared tracks — exploratory prototype, one ancestry')
    for ext in ('png','svg'):fig.savefig(OUT/f'paired-finish-times.{ext}',dpi=180)
    plt.close(fig)
    result={'scope':read(RUN/'frozen/manifest.json')['scope'],'summaries':summaries,'paired':pairs,
        'freeze':read(RUN/'frozen/manifest.json'),'test_freeze':read(RUN/'suite-freeze.json'),
        'raw_result_hashes':read(RUN/'complete.json')['results']}
    write_json(OUT/'report.json',result)
    lines=['# Finalized NEAT and PID comparison','',result['scope'],'',
        'Controllers were frozen before generating 150 fresh geometries (90 mixed, 30 speed, 30 corner). Each pair ran at the same cap and physics. There was no further training or test-based selection.','',
        '| Cap | Controller | Completion | Mean episode speed | Peak speed | Geometric RMSE | Jitter |','|---:|---|---:|---:|---:|---:|---:|']
    for cap in ('100','050'):
        for controller in ('neat','pid'):
            r=summaries[controller+'-'+cap];lines.append(f"| {int(cap)/100:.1f} | {controller} | {r['completed']}/150 | {r['mean_episode_average_speed_m_s']:.3f} m/s | {r['peak_chassis_speed_m_s']:.3f} m/s | {r['mean_episode_geometric_rmse_m']*1000:.2f} mm | {r['mean_command_jitter']:.6f} |")
    lines+=['','All-episode means include failed partial exposure. Paired successful-lap means below use only tracks both controllers finish.','', '| Cap | Shared finishes | NEAT / PID mean time |','|---:|---:|---:|']
    for cap,p in pairs.items():lines.append(f"| {int(cap)/100:.1f} | {p['shared_successes']}/150 | {p['completion_time_s']['neat']:.3f} / {p['completion_time_s']['pid']:.3f} s |")
    lines+=['','![Paired finish times](paired-finish-times.png)','',
        'The NEAT prototype was joint-study g49 rank1. It completed 60/60 primary and 20/20 retention validation episodes, sustained at least 0.950181 m/s and passed straight yaw/steering/time checks, but failed the original geometric-RMSE and command-jitter protection gates. No official winner is claimed. Fresh test results do not retroactively remove those failures.','',
        'PID tuning evaluated 16,080 episodes. The 1.0 and 0.5 m/s tunes passed 60/60 selection tracks plus the previously consulted 45.77 m Rhythm Changes course. No eligible 1.5 m/s tune emerged; its best observed gains completed that course but only 29/60 selection tracks. That failed search is retained and excluded from this validated-baseline assessment.','',
        'This compares resulting controllers from one NEAT ancestry. Five-lineage confirmation and algorithm-wide superiority are unproven. No hardware robustness claim is made. All failures, speed/window measurements, paired error/jitter denominators and frozen model hashes are retained in report.json and the raw run.','',
        'Frozen controllers: `runs/neat-pid-finalization-20261125/frozen/`. Raw tracks/episodes: the same run folder. Earlier protocols and unsuccessful attempts remain in `docs/experiment-ledger.md`.','',
        'Reproduce the saved assessment/report (cached complete results are verified and reused):','', '```powershell','& .\\.venv\\Scripts\\python.exe research/finalize_neat_pid.py --workers 6','```','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'summaries':summaries,'paired':pairs},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();evaluate(args.workers);report()
