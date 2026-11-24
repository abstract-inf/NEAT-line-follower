"""Prepare, audit, review and assess the independent smooth-driving study."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from statistics import mean
from zipfile import ZipFile, ZIP_DEFLATED

from line_follower.config import RobotConfig, EpisodeConfig, verify_geometry
from line_follower.controllers import read_model_artifact
from .common import ROOT, SETTINGS, PARENTS, REWARDS, STUDY_SEED, read, csv_rows, digest, dependencies, sources, verify, write_json, entry_path, lock, epoch_seed
from .courses import protected_geometry, make_suite
from .metrics import quality, select
from .engine import evaluate_tasks, validation_task, references


COUNTS={'training':{'speed':20},'development':{'mixed':20,'speed':20,'corner':10},
        'validation':{'mixed':30,'speed':20,'corner':10},'retention':{'mixed':10,'speed':6,'corner':4}}


def preflight(project=ROOT):
    import run_neat_fast_study as old
    prior=old.verify(project/'runs/neat-fast-study/prepared')
    if dependencies()['neat-python']!='0.92' or dependencies()['mujoco']!='4.0.0':
        raise ValueError('Use the frozen MuJoCo 4.0.0 / NEAT-Python 0.92 environment')
    verify_geometry(project/'runs/neat-fast-study/prepared/inputs/geometry-freeze.json',RobotConfig(**prior['robot']))
    print('Legacy study fingerprints, selected robot, MuJoCo and NEAT versions verified. No simulations started.',flush=True)
    return prior


def audit_choices(project,seed):
    directory=project/f'runs/neat-fast-study/seed-{seed:02d}/training'
    log=[r for r in csv_rows(directory/'generations.csv') if 100<=int(r['generation'])<150]
    episodes=csv_rows(directory/'champion-episodes.csv')
    summaries={}
    for generation in range(100,150):
        records=[r for r in episodes if r['role']=='validation_champion' and int(r['generation'])==generation]
        if len(records)!=60:raise ValueError('Incomplete historical 1 m/s validation log')
        summaries[generation]={'completed':sum(r['completed']=='True' for r in records),
            'jitter':mean(float(r['command_jitter']) for r in records),
            'rmse':mean(float(r['geometry_lateral_rmse_m']) for r in records)}
    best=max(log,key=lambda r:(float(r['validation_completion_rate']),float(r['validation_score']),-int(r['generation'])))
    a=int(best['generation'])
    if a!=PARENTS[seed-1]:raise ValueError('Historical validated parent selection changed')
    b=min(summaries,key=lambda g:(-summaries[g]['completed'],summaries[g]['jitter'],g))
    c=min(summaries,key=lambda g:(-summaries[g]['completed'],summaries[g]['rmse'],summaries[g]['jitter'],g))
    return [{'seed':seed,'generation':g,'source':directory/f'champions/generation-{g:04d}.pkl',
             'historical_metrics':summaries[g]} for g in dict.fromkeys((a,b,c))]


def prepare(run):
    run=Path(run)
    if (run/'prepared/protocol.json').exists():
        print('Prepared protocol exists; verifying without rewriting it.');return verify(run)
    prior=preflight()
    with lock(run):
        prepared=run/'prepared'
        if prepared.exists():raise FileExistsError('Incomplete preparation exists; preserve it and choose another --run')
        (prepared/'inputs').mkdir(parents=True)
        full=ROOT/'runs/neat-fast-study/final-review-20261121/comparison-1m/frozen'
        old_manifest=read(full/'manifest.json')
        inputs={}
        for seed in range(1,6):
            source=full/f'seed-{seed:02d}.pkl'
            if digest(source)!=old_manifest['policies'][f'seed-{seed:02d}']['sha256']:
                raise ValueError('The validated parent was modified')
            inputs[f'seed-{seed:02d}.pkl']=source
        inputs['pid-fullspeed-100.json']=full/'pid-fullspeed.json'
        if digest(inputs['pid-fullspeed-100.json'])!=old_manifest['policies']['pid']['sha256']:
            raise ValueError('The full-speed PID was modified')
        full_pid=read(inputs['pid-fullspeed-100.json'])
        if full_pid['base']!=1. or full_pid['corner_supervision']:
            raise ValueError('Expected the validated full-base filtered PID reference')
        for label in ('050','100','150'):
            inputs[f'pid-legacy-{label}.json']=ROOT/f'runs/neat-fast-study/prepared/inputs/pid-{label}.json'
        inputs['geometry-freeze.json']=ROOT/'runs/neat-fast-study/prepared/inputs/geometry-freeze.json'
        inputs['showcase-track.json']=ROOT/'runs/long-course-candidates-20261121/04-rhythm-changes.json'
        audit=[]
        for seed in range(1,6):
            for choice in audit_choices(ROOT,seed):
                name=f"audit-s{seed}-g{choice['generation']:04d}.pkl"
                inputs[name]=choice.pop('source')
                audit.append({**choice,'file':f'inputs/{name}'})
        for name,source in inputs.items():
            shutil.copy2(source,prepared/'inputs'/name)
            if source.suffix=='.pkl':
                metadata=read_model_artifact(source)['metadata']
                if metadata['controller_semantics']!='mirrored-sensor-history-v1' or metadata['robot']['max_speed']!=1.:
                    raise ValueError('Expected a native 1 m/s, unchanged sensor-history model')
                if any(metadata['robot'][k]!=v for k,v in prior['robot'].items() if k!='max_speed'):
                    raise ValueError('Parent robot geometry/physics mismatch')
        protected=protected_geometry(ROOT)
        used=set(protected)
        for phase in ('pilot','confirmation'):
            counts=COUNTS if phase=='pilot' else {k:v for k,v in COUNTS.items() if k!='development'}
            make_suite(prepared/phase/'suite',phase,counts,used)
        make_suite(prepared/'pid/suite','pid',{'development':COUNTS['development'],'validation':COUNTS['validation']},used)
        source_files=sources()
        for relative in source_files:
            target=prepared/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,target)
        with ZipFile(prepared/'source.zip','w',ZIP_DEFLATED) as archive:
            for relative in source_files:archive.write(ROOT/relative,relative)
        write_json(prepared/'audit-candidates.json',{'selection':'Three distinct historical 1 m/s validation leaders: completion/fitness, completion/jitter, completion/geometric RMSE. No Rhythm Changes outcomes used.','candidates':audit})
        plan={'study':'neat-smooth-study-v1','suite_seed':STUDY_SEED,'settings':SETTINGS,
            'dependencies':dependencies(),'source_files':source_files,'robot':prior['robot'],'episode':prior['episode'],
            'parent_generations':PARENTS,'input_sources':{n:str(p) for n,p in inputs.items()},
            'phase_seeds':{'pilot_both_branches':epoch_seed('pilot',1),
                'confirmation':{str(seed):epoch_seed('confirmation',seed) for seed in range(1,6)},
                'pid_search':{str(cap):epoch_seed('pid-search',round(cap*100)) for cap in (.5,1.,1.5)}},
            'protected_geometry':sorted(protected),'reserved_geometry':sorted(used-protected),
            'reward_versions':REWARDS,'reward_weights':{'yaw':.15,'steering':.15,'tracking':.10,'speed_deficit':.10},
            'gates':{'paired_time_ratio_max':1.05,'straight_speed_min':.95,'yaw_rms_max':.15,'steering_rms_max':.02},
            'test_seed_commitment':digest_text([STUDY_SEED,'final-test-v1']),
            'test_counts':{'mixed':90,'speed':30,'corner':30},
            'execution':'Preparation does not run simulations, tune PID or evolve NEAT. Pilot branches and confirmation seeds are separate manual commands.',
            'source_isolation':'Commands execute the frozen source bundle, not edited legacy modules.',
            'claim_scope':'Five warm-started lineages; nominal matched generation/population pilot, actual episode counts reported; pilot reward comparison is one lineage only.'}
        plan['prepared_files']={p.relative_to(prepared).as_posix():digest(p) for p in prepared.rglob('*')
                                if p.is_file() and 'source' not in p.relative_to(prepared).parts}
        write_json(prepared/'protocol.json',plan)
        (prepared/'protocol.sha256').write_text(digest(prepared/'protocol.json')+'\n',encoding='utf-8')
        verify(run)
        write_json(run/'preparation-summary.json',{'prepared':True,'protocol_sha256':digest(prepared/'protocol.json'),
            'distinct_prepared_tracks':len(used-protected),'audit_candidates':len(audit),'parents':PARENTS,
            'legacy_fingerprints_verified':True,'simulation_or_evolution_started':False,
            'source_bundle_verified':True,'final_test_tracks_generated':False})
        print(f'Prepared {len(used-protected)} disjoint course geometries and {len(audit)} audit models. No evolution or tuning started.',flush=True)
        return plan


def digest_text(value):
    import hashlib
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def audit(run,workers):
    run=Path(run);verify(run)
    directory=run/'audit'
    with lock(directory):
        candidates=read(run/'prepared/audit-candidates.json')['candidates']
        results=[]
        for choice in candidates:
            target=directory/f"s{choice['seed']}-g{choice['generation']:04d}.json"
            if target.exists():record=read(target)
            else:
                reference=references(run,'pilot',choice['seed'],workers)
                model=run/'prepared'/choice['file']
                primary,low=evaluate_tasks(run,[validation_task(run,'pilot',split,model=str(model))
                                                for split in ('validation','retention')],workers)
                gate=quality(primary,low,reference['primary'],reference['retention'])
                record={**choice,'quality':{**gate,'generation':choice['generation']},'primary':primary,'retention':low,
                        'protocol_sha256':digest(run/'prepared/protocol.json'),'model_sha256':digest(model)}
                write_json(target,record)
            if record['protocol_sha256']!=digest(run/'prepared/protocol.json'):raise ValueError('Stale audit evidence')
            results.append(record)
        write_json(directory/'complete.json',{'candidates':len(results),'qualified_existing':[r for r in results if r['quality']['eligible']],
                    'protocol_sha256':digest(run/'prepared/protocol.json')})
        print(f"Audit complete: {sum(r['quality']['eligible'] for r in results)}/{len(results)} qualified existing models",flush=True)


def review_pilot(run):
    run=Path(run);verify(run)
    candidates=[];signatures=[];branches={}
    for branch in ('old','smooth'):
        directory=run/'pilot/seed-01'/branch
        done=read(directory/'complete.json')
        if done['generations']!=100 or done['signature']['protocol_sha256']!=digest(run/'prepared/protocol.json'):
            raise ValueError('Both pilot branches must finish the frozen protocol')
        if done['qualified']:
            if digest(directory/'best.pkl')!=done['best_sha256']:raise ValueError('Pilot winner changed')
            candidates.append({**read(directory/'selection.json'),'branch':branch})
        signatures.append({k:v for k,v in done['signature'].items() if k!='branch'})
        branches[branch]={'complete_sha256':digest(directory/'complete.json'),
                          'best_sha256':done['best_sha256'],
                          'selection_sha256':digest(directory/'selection.json') if done['qualified'] else None}
    if signatures[0]!=signatures[1]:raise ValueError('Pilot branches have different parents, settings or references')
    for generation in range(100):
        paths=[run/'pilot/seed-01'/branch/'tracks'/f'generation-{generation:04d}'/'batch.json' for branch in ('old','smooth')]
        if read(paths[0])['hashes']!=read(paths[1])['hashes']:raise ValueError('Pilot branches did not use identical training batches')
    winner=select(candidates)
    record={'protocol_sha256':digest(run/'prepared/protocol.json'),'successful':winner is not None,
            'branches':branches,'matched_tracks_verified':True,
            'chosen_branch':winner['branch'] if winner else None,'selection':winner,
            'reason':'Same predeclared gates/ranking; exact branch ties prefer old fitness. No test tracks opened.',
            'confirmation_requires_explicit_launch':True}
    write_json(run/'pilot/review.json',record)
    print(json.dumps(record,indent=2))
    return record


def confirmation_branch(run):
    record=read(Path(run)/'pilot/review.json')
    if not record['successful'] or record['protocol_sha256']!=digest(Path(run)/'prepared/protocol.json'):
        raise ValueError('A successful frozen pilot review is required before confirmation')
    for branch,evidence in record.get('branches',{}).items():
        directory=Path(run)/'pilot/seed-01'/branch
        if digest(directory/'complete.json')!=evidence['complete_sha256']:
            raise ValueError('Pilot completion changed after review')
        if evidence['best_sha256'] is not None and (digest(directory/'best.pkl')!=evidence['best_sha256'] or
                digest(directory/'selection.json')!=evidence['selection_sha256']):
            raise ValueError('Pilot selection changed after review')
    return record['chosen_branch']


def freeze_final(run):
    run=Path(run);verify(run)
    branch=confirmation_branch(run)
    policies={}
    for seed in range(1,6):
        directory=run/'confirmation'/f'seed-{seed:02d}'/branch
        if not (directory/'complete.json').exists():raise ValueError(f'Confirmation seed {seed} is incomplete; assessment cannot start')
        done=read(directory/'complete.json')
        if done['generations']!=100 or not done['qualified']:
            raise ValueError(f'Seed {seed} has no completed qualifying winner; final assessment is blocked')
        source=directory/'best.pkl'
        signature=done['signature']
        expected_parent=digest(run/'prepared/inputs'/f'seed-{seed:02d}.pkl')
        metadata=read_model_artifact(source)['metadata']
        if signature['seed']!=seed or signature['phase']!='confirmation' or signature['branch']!=branch or signature['parent_sha256']!=expected_parent or metadata['parent_sha256']!=expected_parent:
            raise ValueError('Seed-specific confirmation ancestry changed')
        if digest(source)!=done['best_sha256'] or done['signature']['protocol_sha256']!=digest(run/'prepared/protocol.json'):
            raise ValueError('Confirmation inputs/winner changed')
        policies[f'new-s{seed}']={'source':source,'cap_specific':1.}
        policies[f'reference-s{seed}']={'source':run/'prepared/inputs'/f'seed-{seed:02d}.pkl','cap_specific':1.}
    for cap in (.5,1.,1.5):
        frozen=run/'pid'/f'cap-{round(cap*100):03d}'
        if not (frozen/'selection-freeze.json').exists():raise ValueError(f'Full-speed PID {cap:.2f} is not frozen; assessment cannot start')
        record=read(frozen/'selection-freeze.json')
        source=frozen/record['file']
        pid_protocol=read(frozen/'protocol.json')
        if pid_protocol['protocol_sha256']!=digest(run/'prepared/protocol.json') or record['protocol_sha256']!=digest(frozen/'protocol.json'):
            raise ValueError('PID selection belongs to different tuning inputs')
        parameters=read(source)
        if parameters['base']!=1. or parameters['corner_supervision']:
            raise ValueError('Final PID must be full-base filtered remembered-side recovery')
        if digest(source)!=record['sha256'] or not record['eligible']:
            raise ValueError(f'No qualified full-speed PID at cap {cap}')
        policies[f'pid-{round(cap*100):03d}']={'source':source,'cap_specific':cap}
    directory=run/'assessment/frozen'
    if (directory/'manifest.json').exists():
        manifest=read(directory/'manifest.json')
        for key,record in manifest['policies'].items():
            if digest(directory/record['file'])!=record['sha256'] or digest(policies[key]['source'])!=record['sha256']:
                raise ValueError('Final frozen controller changed')
        return manifest
    directory.mkdir(parents=True,exist_ok=True)
    frozen={}
    for key,record in policies.items():
        target=directory/(key+record['source'].suffix);shutil.copy2(record['source'],target)
        frozen[key]={'file':target.name,'sha256':digest(target),'source':str(record['source']),
                     'cap_specific':record['cap_specific']}
    manifest={'protocol_sha256':digest(run/'prepared/protocol.json'),'policies':frozen,
              'note':'All policies frozen before final-test course generation/loading'}
    write_json(directory/'manifest.json',manifest)
    return manifest


def assess(run,workers):
    run=Path(run);verify(run)
    with lock(run/'assessment'):
        manifest=freeze_final(run)
        suite=run/'assessment/suite/suite.json'
        if not suite.exists():
            plan=read(run/'prepared/protocol.json')
            if plan['test_seed_commitment']!=digest_text([STUDY_SEED,'final-test-v1']):raise ValueError('Test seed changed')
            used=set(plan['protected_geometry'])|set(plan['reserved_geometry'])
            for path in (run/'confirmation').rglob('batch.json'):used.update(read(path)['hashes'])
            for path in (run/'pilot').rglob('batch.json'):used.update(read(path)['hashes'])
            make_suite(suite.parent,'final-test-v1',{'test':plan['test_counts']},used)
            write_json(run/'assessment/test-freeze.json',{'suite_sha256':digest(suite),'controllers_sha256':digest(run/'assessment/frozen/manifest.json'),
                 'files':{p.relative_to(suite.parent).as_posix():digest(p) for p in suite.parent.rglob('*.json')}})
        freeze=read(run/'assessment/test-freeze.json')
        if freeze['controllers_sha256']!=digest(run/'assessment/frozen/manifest.json'):raise ValueError('Assessment controllers changed')
        for rel,sha in freeze['files'].items():
            if digest(suite.parent/rel)!=sha:raise ValueError('Final test input changed')
        entries=read(suite)['splits']['test']
        count=0
        pending=[]
        for cap in (1.,.5,1.5):
            for name,policy in manifest['policies'].items():
                if name.startswith('pid') and policy['cap_specific']!=cap:continue
                model=run/'assessment/frozen'/policy['file']
                parameters=read(model) if model.suffix=='.json' else None
                for slot,entry in enumerate(entries):
                    path=run/'assessment/episodes'/f'{name}-cap-{round(cap*100):03d}-{slot:03d}.json'
                    if path.exists():
                        saved=read(path)
                        if saved['policy_sha256']!=policy['sha256'] or saved['test_freeze_sha256']!=digest(run/'assessment/test-freeze.json'):
                            raise ValueError('Stale assessment result')
                        count+=1;continue
                    task={'paths':[str(suite.parent/entry['path'])],'cap':cap,'detailed':True,
                          'groups':{entry['geometry_hash']:entry['group']},
                          **({'parameters':parameters} if parameters else {'model':str(model)})}
                    pending.append((path,name,slot,policy['sha256'],task))
        # Persistent workers reuse masks/simulations; parent commits each completed
        # episode in bounded chunks, rather than rebuilding a process per track.
        import multiprocessing as mp
        from .engine import initialize,worker
        if pending:
            with mp.get_context('spawn').Pool(workers,initialize,(str(run),None),
                     maxtasksperchild=SETTINGS['worker_recycle_tasks']) as pool:
                iterator=pool.imap(worker,[t[-1] for t in pending],chunksize=1)
                for (path,name,slot,sha,_),records in zip(pending,iterator):
                    write_json(path,{'policy':name,'slot':slot,'policy_sha256':sha,
                        'test_freeze_sha256':digest(run/'assessment/test-freeze.json'),**records[0]})
                    count+=1
                    if count%20==0:print(f'Assessment: {count}/4950 committed episodes',flush=True)
        write_json(run/'assessment/complete.json',{'episodes':count,'test_freeze_sha256':digest(run/'assessment/test-freeze.json')})
        print('Assessment complete; run plot to produce summaries and figures.',flush=True)
