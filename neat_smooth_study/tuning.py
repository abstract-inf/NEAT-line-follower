"""Resumable full-base PID tuning with the existing filtered recovery controller."""
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

import numpy as np

from line_follower.pid import ResearchPIDConfig
from .common import read, write_json, digest, verify, lock, entry_path, epoch_seed
from .engine import evaluate_tasks

FIELDS=('kp','ki','kd','derivative_filter_s','slowdown_gain')
BOUNDS=((.05,3.),(0.,.25),(.005,.20),(.02,.20),(.3,10.))


def summarize(records,cap):
    dedicated=[r for r in records if r['kind']=='isolated']
    windows=[w for r in dedicated for w in r['straight_windows']]
    feasible=bool(dedicated) and all(r['completed'] for r in dedicated) and bool(windows) and all(
        w['observed'] and w['speed_m_s']>=.95*cap for w in windows)
    observed=[w for r in records for w in r['straight_windows'] if w['observed']]
    return {'completed':sum(r['completed'] for r in records),'tracks':len(records),'speed_feasible':feasible,
            'eligible':feasible and sum(r['completed'] for r in records)>=.95*len(records),
            'failure_penalized_mean_time_s':mean(r['completion_time_s'] if r['completed'] else r['time_limit_s'] for r in records),
            'steady_yaw_rms':mean(w['yaw_rms_rad_s'] for w in observed) if observed else 1e9,
            'command_jitter':mean(r['command_jitter'] for r in records),
            'geometric_rmse_m':mean(r['geometry_lateral_rmse_m'] for r in records),
            'minimum_dedicated_speed_m_s':min((w['speed_m_s'] for w in windows if w['observed']),default=None)}


def rank(summary):
    return (summary['speed_feasible'],summary['completed'],-summary['failure_penalized_mean_time_s'],
            -summary['steady_yaw_rms'],-summary['command_jitter'])


def promote(summaries,limit=12):
    eligible=[name for name,s in summaries.items() if s['speed_feasible']]
    if not eligible:raise ValueError('No full-speed-feasible PID candidate; retain failed search evidence')
    fast=sorted(eligible,key=lambda n:rank(summaries[n]),reverse=True)
    maximum=max(summaries[n]['completed'] for n in eligible)
    smooth=sorted((n for n in eligible if summaries[n]['completed']==maximum),
                  key=lambda n:(summaries[n]['steady_yaw_rms'],summaries[n]['command_jitter']))
    return list(dict.fromkeys(fast[:8]+smooth[:4]+fast))[:limit]


def tune(run,cap,workers):
    run=Path(run);verify(run)
    directory=run/'pid'/f'cap-{round(cap*100):03d}'
    signature={'protocol_sha256':digest(run/'prepared/protocol.json'),'cap':cap,'bounds':BOUNDS,'base':1.}
    signature=json.loads(json.dumps(signature))
    with lock(directory):
        if (directory/'protocol.json').exists() and read(directory/'protocol.json')!=signature:
            raise ValueError('PID tuning inputs changed')
        write_json(directory/'protocol.json',signature)
        if (directory/'selection-freeze.json').exists():
            frozen=read(directory/'selection-freeze.json')
            if digest(directory/frozen['file'])!=frozen['sha256']:raise ValueError('Selected PID changed')
            print('Existing full-speed PID selection verified');return
        suite=read(run/'prepared/pid/suite/suite.json')['splits']
        def batch(policies,entries,label):
            results={};pending=[];names=[]
            for name,parameters in policies.items():
                path=directory/'episodes'/label/f'{name}.json'
                if path.exists():
                    saved=read(path)
                    if saved['parameters']!=parameters or saved['signature']!=signature:raise ValueError('Stale PID candidate result')
                    results[name]=saved['rows']
                else:
                    pending.append({'parameters':parameters,'paths':[str(entry_path(run,'pid',e)) for e in entries],
                        'cap':cap,'detailed':True,'groups':{e['geometry_hash']:e['group'] for e in entries}})
                    names.append(name)
            # Bounded batches commit each candidate's complete track set before proceeding.
            for begin in range(0,len(pending),max(1,workers)):
                evaluated=evaluate_tasks(run,pending[begin:begin+workers],workers)
                for name,rows in zip(names[begin:begin+workers],evaluated):
                    rows=[{**r,'kind':e['kind']} for r,e in zip(rows,entries)]
                    write_json(directory/'episodes'/label/f'{name}.json',{'signature':signature,'parameters':policies[name],'rows':rows})
                    results[name]=rows
                print(f'PID {cap:.2f} {label}: {min(begin+workers,len(pending))}/{len(pending)} new candidates committed',flush=True)
            return results
        full={**read(run/'prepared/inputs/pid-fullspeed-100.json'),'base':1.}
        # Validate and retain the already-established full-speed 1 m/s tune first.
        if cap==1.:
            records=batch({'existing-fullspeed-100':full},suite['validation'],'existing-validation')
            summary=summarize(records['existing-fullspeed-100'],cap)
            if summary['eligible']:
                freeze(directory,full,'existing-fullspeed-100',summary);return
        legacy=read(run/'prepared/inputs'/f'pid-legacy-{round(cap*100):03d}.json')
        policies={};seen=set()
        def add(values,prefix):
            point=tuple(float(np.clip(v,*bounds)) for v,bounds in zip(values,BOUNDS))
            if point in seen:return None
            seen.add(point);name=f'{prefix}-{len(policies):03d}'
            policy={**full,**dict(zip(FIELDS,point))}
            ResearchPIDConfig(**policy)
            policies[name]=policy;return name
        add([full[k] for k in FIELDS],'full-reference')
        add([legacy[k] for k in FIELDS],'cap-legacy-base-one')
        for p in (.85,1.25,2.,3.):
            for d in (.6,1.,1.4,2.):
                for filtering in (.06,.10):
                    for slowdown in (1.5,4.):add((full['kp']*p,full['ki'],full['kd']*d,filtering,slowdown),'grid')
        rng=np.random.default_rng(epoch_seed('pid-search',round(cap*100)))
        for index in range(64):
            add((np.exp(rng.uniform(np.log(.05),np.log(3))),0. if index%3==0 else rng.uniform(0,.25),
                 np.exp(rng.uniform(np.log(.005),np.log(.2))),rng.uniform(.02,.2),
                 np.exp(rng.uniform(np.log(.3),np.log(10)))),'random')
        development=suite['development']
        coarse=([e for e in development if e['kind']=='isolated'][:4]+
                [e for e in development if e['kind']=='post_turn'][:4]+
                [e for e in development if e['group']=='mixed'][:2]+
                [e for e in development if e['group']=='corner'][:2])
        results=batch(policies,coarse,'coarse')
        names=promote({n:summarize(r,cap) for n,r in results.items()})
        results=batch({n:policies[n] for n in names},development,'expanded')
        for generation in (1,2):
            parents=promote({n:summarize(r,cap) for n,r in results.items()},6)
            children=[]
            for index in range(24):
                parent=policies[parents[index%len(parents)]]
                center=np.array([parent[k] for k in FIELDS])
                scales=np.array([max(.04,center[0]*.15),.025,max(.005,center[2]*.15),.012,max(.3,center[4]*.2)])
                for _ in range(100):
                    name=add(rng.normal(center,scales),f'refine{generation}')
                    if name:children.append(name);break
                else:raise ValueError('PID refinement exhausted unique candidates')
            results.update(batch({n:policies[n] for n in children},development,f'refine-{generation}'))
        finalists=promote({n:summarize(r,cap) for n,r in results.items()},10)
        records=batch({n:policies[n] for n in finalists},suite['validation'],'validation')
        summaries={n:summarize(r,cap) for n,r in records.items()}
        write_json(directory/'candidates.json',policies);write_json(directory/'validation-summary.json',summaries)
        eligible=[n for n,s in summaries.items() if s['eligible']]
        if not eligible:
            write_json(directory/'failed.json',{'reason':'No candidate passed full-speed feasibility and 95% validation completion'})
            raise ValueError('No satisfactory full-speed PID; search evidence preserved, no winner frozen')
        maximum=max(summaries[n]['completed'] for n in eligible)
        eligible=[n for n in eligible if summaries[n]['completed']==maximum]
        fastest=min(summaries[n]['failure_penalized_mean_time_s'] for n in eligible)
        eligible=[n for n in eligible if summaries[n]['failure_penalized_mean_time_s']<=1.02*fastest]
        name=min(eligible,key=lambda n:(summaries[n]['steady_yaw_rms'],summaries[n]['command_jitter'],
                                      summaries[n]['geometric_rmse_m'],summaries[n]['failure_penalized_mean_time_s']))
        freeze(directory,policies[name],name,summaries[name])


def freeze(directory,parameters,candidate,summary):
    if parameters['base']!=1. or not summary['eligible']:raise ValueError('Cannot freeze an ineligible full-speed PID')
    target=directory/'pid-fullspeed.json';write_json(target,parameters)
    write_json(directory/'selection-freeze.json',{'file':target.name,'sha256':digest(target),'candidate':candidate,
               'eligible':True,'summary':summary,'protocol_sha256':digest(directory/'protocol.json')})
    print(f'Selected full-speed PID {candidate}',flush=True)
