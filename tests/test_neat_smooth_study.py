"""Preparation/dispatch and metric regressions. Evolution is never started."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import pickle
import random
import shutil

import numpy as np
import pytest

from line_follower.config import RobotConfig
from line_follower.controllers import read_model_artifact
from line_follower.experiments import warm_start_population,_save_checkpoint
from line_follower.fitness import progress_fitness
from line_follower.pid import ResearchPIDConfig,ResearchPIDController
from line_follower.tracks import TrackSpec,Section,validate_track
from neat_smooth_study import common,metrics,courses,engine,workflow,tuning
import run_neat_smooth_study as launcher


def episode(group,identity,**changes):
    row={'group':group,'track_hash':identity,'completed':True,'completion_time_s':10.,
         'geometry_lateral_rmse_m':.01,'command_jitter':.001,
         'straight_windows':[{'observed':True,'samples':120,'speed_m_s':.99,'yaw_rms_rad_s':.1,'steering_rms':.01}] if group=='speed' else []}
    return {**row,**changes}


def gate_fixture():
    rows=[episode(group,str(i)) for i,group in enumerate(('mixed','speed','corner'))]
    return rows,deepcopy(rows),deepcopy(rows),deepcopy(rows)


def test_normalized_penalty_orders_motion_and_speed():
    smooth=[{'observed':True,'speed_m_s':.98,'yaw_rms_rad_s':.05,'steering_rms':.005}]
    jitter=[{**smooth[0],'yaw_rms_rad_s':.68,'steering_rms':.09}]
    assert sum(metrics.added_costs(smooth,.01).values())>sum(metrics.added_costs(jitter,.01).values())
    slow=[{**smooth[0],'speed_m_s':.5}]
    assert sum(metrics.added_costs(smooth,.01).values())>sum(metrics.added_costs(slow,.01).values())
    assert -.5<=sum(metrics.added_costs([{'observed':False}],10).values())<=0
    assert metrics.added_costs([],0)['straight_yaw_term']==0


def test_completed_reward_stays_above_early_failure_even_at_worst_cost():
    completed=progress_fitness(1,True,120,120,0,7200,.5,command_jitter=1,command_jitter_penalty=.2,route_length=1,fitness_version='route-speed-1.0')-.5
    failed=progress_fitness(.999,False,.01,120,1e9,1,.5,command_jitter=0,command_jitter_penalty=.2,route_length=1,fitness_version='route-speed-1.0')
    assert completed>failed


def test_windows_exclude_curves_braking_and_short_straights():
    track=TrackSpec('window',(Section('straight',length=6),Section('arc',radius=1,angle=np.pi/2),Section('straight',length=3)))
    assert metrics.straight_windows(track)==[(2.,4.)]
    assert not metrics.window_metrics([(1,0,0)]*20,True)['observed']
    assert not metrics.window_metrics([(1,0,0)]*120,False)['observed']
    assert metrics.window_metrics([(1,0,0)]*120,True)['yaw_rms_rad_s']==0


@pytest.mark.parametrize('fault',['completion','retention','time','rmse','jitter','speed','yaw','steering','missing'])
def test_quality_rejects_each_regression(fault):
    primary,retention,reference,low=gate_fixture()
    if fault=='completion':primary[0]['completed']=False
    if fault=='retention':retention[0]['completed']=False
    if fault=='time':
        for r in primary:r['completion_time_s']=10.501
    if fault in ('rmse','jitter'):
        primary[0]['geometry_lateral_rmse_m' if fault=='rmse' else 'command_jitter']*=2
    if fault in ('speed','yaw','steering','missing'):
        field,value={'speed':('speed_m_s',.94),'yaw':('yaw_rms_rad_s',.16),'steering':('steering_rms',.021),'missing':('observed',False)}[fault]
        primary[1]['straight_windows'][0][field]=value
    assert not metrics.quality(primary,retention,reference,low)['eligible']


def test_quality_boundaries_and_identity_checks():
    a,b,c,d=gate_fixture()
    for r in a:r['completion_time_s']=10.5
    a[1]['straight_windows'][0].update(speed_m_s=.95,yaw_rms_rad_s=.15,steering_rms=.02)
    assert metrics.quality(a,b,c,d)['eligible']
    a[0]['track_hash']='other'
    with pytest.raises(ValueError):metrics.quality(a,b,c,d)


def test_ranking_uses_two_percent_band_and_old_branch_ties():
    record={'eligible':True,'completed':60,'paired_time_ratio':1.,'straight_yaw_rms':.1,
            'command_jitter':.001,'geometric_rmse_m':.01,'branch':'old','generation':50}
    good={**record,'paired_time_ratio':1.01,'straight_yaw_rms':.01,'branch':'smooth','generation':1}
    outside={**record,'paired_time_ratio':1.021,'straight_yaw_rms':0.,'generation':0}
    assert metrics.select([record,good,outside]) is good
    assert metrics.select([{**record,'branch':'smooth','generation':1},record]) is record


def test_course_bank_and_pilot_batches_are_shared_and_disjoint(tmp_path):
    bank=[courses.speed_course(4200+i,i,'training') for i in range(20)]
    assert all(not validate_track(t) for t in bank)
    assert len({t.geometry_hash for t in bank})==20
    for t in bank:
        assert t.resolution==.002 and all(s.width==.02 for s in t.sections)
        assert metrics.straight_windows(t)
    old=courses.generation_batch(91,0,bank,set())
    smooth=courses.generation_batch(91,0,bank,set())
    assert [t.geometry_hash for t in old]==[t.geometry_hash for t in smooth]
    assert len(old)==6 and len({t.geometry_hash for t in old})==6
    assert old[0] in bank[:10] and old[1] in bank[10:]
    used=set()
    manifest=courses.make_suite(tmp_path,'test-separation',{'validation':{'mixed':3,'speed':4,'corner':2}},used)
    assert len(used)==9
    assert all(TrackSpec.load(tmp_path/e['path']).geometry_hash==e['geometry_hash'] for e in manifest['splits']['validation'])


def test_actual_pid_class_permits_full_target_only_at_base_one():
    sensors=np.zeros(15);sensors[7]=1
    for cap in (.5,1.,1.5):
        for base in (.75,1.):
            controller=ResearchPIDController(ResearchPIDConfig(base=base),RobotConfig(max_speed=cap))
            action=controller.act(sensors,1/60)
            assert np.allclose(action,[base,base])
            assert np.mean(action)*cap==base*cap


def test_seed_specific_warm_start_is_deterministic_and_preserves_exact_parent():
    for seed,generation in enumerate(common.PARENTS,1):
        path=common.ROOT/f'runs/neat-fast-study/final-review-20261121/comparison-1m/frozen/seed-{seed:02d}.pkl'
        artifact=read_model_artifact(path)
        assert artifact['metadata']['generation']==generation
        random.seed(common.epoch_seed('pilot',seed));a=warm_start_population(path,100)
        random.seed(common.epoch_seed('pilot',seed));b=warm_start_population(path,100)
        assert len(a.population)==100
        assert pickle.dumps(a.population)==pickle.dumps(b.population)
        exact=deepcopy(artifact['genome']);exact.key=1;exact.fitness=None
        assert pickle.dumps(a.population[1])==pickle.dumps(exact)


def frozen_fixture(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    inventory=common.sources()
    for relative in inventory:
        target=prepared/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(common.ROOT/relative,target)
    (prepared/'input.json').write_text('{}')
    common.write_json(prepared/'protocol.json',{'settings':common.SETTINGS,'dependencies':common.dependencies(),
        'source_files':inventory,'prepared_files':{'input.json':common.digest(prepared/'input.json')}})
    (prepared/'protocol.sha256').write_text(common.digest(prepared/'protocol.json'),encoding='utf-8')
    return prepared


@pytest.mark.parametrize('change',['input','frozen-source','settings'])
def test_changed_resume_inputs_are_rejected(tmp_path,change):
    prepared=frozen_fixture(tmp_path)
    common.verify(tmp_path)
    if change=='input':(prepared/'input.json').write_text('{"changed":true}')
    if change=='frozen-source':
        with (prepared/'source/neat_smooth_study/metrics.py').open('a') as f:f.write('\n# changed\n')
    if change=='settings':
        p=common.read(prepared/'protocol.json');p['settings']['generations']=101;common.write_json(prepared/'protocol.json',p)
    with pytest.raises(ValueError):common.verify(tmp_path)


def test_dispatch_stubs_evolution_and_rejects_incomplete_confirmation(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'verify',lambda r:{})
    (tmp_path/'prepared').mkdir();common.write_json(tmp_path/'prepared/protocol.json',{})
    common.write_json(tmp_path/'audit/complete.json',{'protocol_sha256':common.digest(tmp_path/'prepared/protocol.json'),'qualified_existing':[]})
    calls=[]
    monkeypatch.setattr(engine,'train',lambda *a,**k:calls.append((a,k)))
    args=launcher.parser().parse_args(['pilot','--branch','smooth','--run',str(tmp_path)])
    launcher.dispatch(args)
    assert calls[0][0][1:4]==('pilot',1,'smooth')
    monkeypatch.setattr(workflow,'verify',lambda r:{})
    common.write_json(tmp_path/'pilot/review.json',{'successful':True,'chosen_branch':'smooth',
        'protocol_sha256':common.digest(tmp_path/'prepared/protocol.json')})
    with pytest.raises(ValueError,match='incomplete'):workflow.freeze_final(tmp_path)
    assert not (tmp_path/'assessment/suite').exists()


def test_checkpoint_commits_and_roundtrips_random_state(tmp_path):
    path=tmp_path/'latest.pkl.gz'
    state={'checkpoint_version':2,'neat_version':'0.92','random_state':random.getstate(),
           'signature':{'parent':'frozen','source':'frozen'},'population':{'generation':10}}
    _save_checkpoint(path,state)
    with gzip.open(path,'rb') as f:assert pickle.load(f)==state
    assert not list(tmp_path.glob('*.tmp'))


def test_physics_smoke_component_accounting_and_fullspeed_window(tmp_path):
    from line_follower.fast_study import centered_course
    prior=common.read(common.ROOT/'runs/neat-fast-study/prepared/protocol.json')
    common.write_json(tmp_path/'prepared/protocol.json',{'robot':prior['robot'],'episode':prior['episode']})
    track=centered_course([Section('straight',length=6)],'smoke',23,'development')
    path=tmp_path/'straight.json';track.save(path)
    params=common.read(common.ROOT/'runs/neat-fast-study/final-review-20261121/comparison-1m/frozen/pid-fullspeed.json')
    context=engine.Context(tmp_path)
    old=context.evaluate({'parameters':params,'paths':[str(path)],'detailed':True,'branch':'old'})[0]
    smooth=context.evaluate({'parameters':params,'paths':[str(path)],'detailed':True,'branch':'smooth'})[0]
    assert old['completed'] and smooth['completed']
    assert old['straight_windows'][0]['observed']
    assert old['straight_windows'][0]['speed_m_s']>=.95
    for result in (old,smooth):assert sum(result['components'].values())==pytest.approx(result['score'],abs=1e-12)
    assert old['elapsed_s']==smooth['elapsed_s']
    assert old['command_jitter']==smooth['command_jitter']
    assert old['geometry_lateral_rmse_m']==smooth['geometry_lateral_rmse_m']
    # Exercise the real Windows-spawn boundary without evolving or tuning.
    parent=common.ROOT/'runs/neat-fast-study/final-review-20261121/comparison-1m/frozen/seed-01.pkl'
    results=engine.evaluate_tasks(tmp_path,[{'parameters':params,'paths':[str(path)],'branch':'old'},
        {'genome':read_model_artifact(parent)['genome'],'paths':[str(path)],'branch':'smooth'}],workers=2,parent=parent)
    assert results[0][0]['elapsed_s']==old['elapsed_s'] and results[0][0]['score']==old['score']
    assert results[1][0]['completed'] and results[1][0]['straight_windows'][0]['observed']


def test_cache_evicts_before_allocation_and_distinguishes_pose(tmp_path,monkeypatch):
    from types import SimpleNamespace
    common.write_json(tmp_path/'prepared/protocol.json',{})
    context=engine.Context(tmp_path)
    monkeypatch.setattr(engine,'SETTINGS',{**common.SETTINGS,'mask_cache_bytes':100000})
    allocations=[]
    def fake_mask(track):
        allocations.append(len(context.masks))
        return SimpleNamespace(field=np.zeros((200,200),dtype=np.uint8),pixels=200,track=track)
    monkeypatch.setattr(engine,'TrackMask',fake_mask)
    a=TrackSpec('a',(Section('straight',length=1),),arena_size=4,resolution=.02)
    b=TrackSpec('b',a.sections,start=(0,0,0),arena_size=4,resolution=.02)
    first=context.mask(a)
    assert context.mask(a) is first
    assert context.mask(b) is not first
    assert allocations==[0,0]
    assert sum(m.field.nbytes+m.pixels*m.pixels for m in context.masks.values())<=100000


def test_pid_final_eligibility_requires_speed_and_95_percent_completion():
    records=[]
    for i in range(60):
        r=episode('speed' if i<20 else 'mixed',str(i))
        r.update(kind='isolated' if i<10 else 'post_turn' if i<20 else 'mixed',time_limit_s=45.)
        records.append(r)
    for r in records[-3:]:r['completed']=False
    assert tuning.summarize(records,1.)['eligible']
    records[-4]['completed']=False
    assert not tuning.summarize(records,1.)['eligible']
    records[-4]['completed']=True
    records[0]['straight_windows'][0]['speed_m_s']=.75
    assert not tuning.summarize(records,1.)['eligible']


class StubPopulation:
    """Exercises the commit boundary without calling NEAT evolution."""
    def __init__(self,artifact):
        self.config=artifact['config'];self.generation=0;self.draws=[]
        self.population={i:deepcopy(artifact['genome']) for i in (1,2)}
        for i,g in self.population.items():g.key=i
        from types import SimpleNamespace
        self.species=SimpleNamespace(species={})

    def run(self,callback,n):
        assert n==1
        callback(list(self.population.items()),self.config)
        self.draws.append(random.random())
        self.generation+=1


class StubIterator:
    def __init__(self,records):self.records=iter(records)
    def next(self,timeout=None):return next(self.records)


class StubPool:
    def __init__(self,*args,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def imap(self,function,tasks,chunksize=1):return StubIterator([function(t) for t in tasks])


def test_training_commit_resume_boundary_with_evolution_stubbed(tmp_path,monkeypatch):
    from types import SimpleNamespace
    parent=common.ROOT/'runs/neat-fast-study/final-review-20261121/comparison-1m/frozen/seed-01.pkl'
    artifact=read_model_artifact(parent)
    prepared=tmp_path/'prepared';(prepared/'inputs').mkdir(parents=True)
    shutil.copy2(parent,prepared/'inputs/seed-01.pkl')
    prior=common.read(common.ROOT/'runs/neat-fast-study/prepared/protocol.json')
    plan={**prior,'protected_geometry':[]}
    common.write_json(prepared/'protocol.json',plan)
    entries=[];tracks=[]
    for i,group in enumerate(('mixed','speed','corner')):
        track=TrackSpec(f'stub-{i}',(Section('straight',length=1+i*.1),))
        rel=f'{i}.json';entries.append({'path':rel,'geometry_hash':track.geometry_hash,'group':group,'kind':'isolated'})
        tracks.append(track)
    for phase in ('pilot','confirmation','pid'):
        for entry,track in zip(entries,tracks):track.save(prepared/phase/'suite'/entry['path'])
        common.write_json(prepared/phase/'suite/suite.json',{'splits':{'training':entries,'validation':entries,'retention':entries}})
    reference={'primary':[episode(e['group'],e['geometry_hash']) for e in entries],
               'retention':[episode(e['group'],e['geometry_hash']) for e in entries]}
    common.write_json(tmp_path/'pilot/references/seed-01.json',reference)
    monkeypatch.setattr(engine,'verify',lambda r:plan)
    monkeypatch.setattr(engine,'references',lambda *a:reference)
    settings={**common.SETTINGS,'generations':2,'population':2}
    monkeypatch.setattr(engine,'SETTINGS',settings)
    monkeypatch.setattr(engine,'warm_start_population',lambda *a:StubPopulation(artifact))
    monkeypatch.setattr(engine,'generation_batch',lambda *a:tracks+tracks)
    monkeypatch.setattr(engine.mp,'get_context',lambda *a:SimpleNamespace(Pool=StubPool))
    def fake_worker(task):
        output=[]
        for path in task['paths']:
            t=TrackSpec.load(path);group=task.get('groups',{}).get(t.geometry_hash,'training')
            output.append({**episode(group,t.geometry_hash),'score':2.4,'components':{'stub_term':2.4}})
        return output
    monkeypatch.setattr(engine,'worker',fake_worker)
    directory=engine.train(tmp_path,'pilot',1,'old',stop_after=1)
    assert len(common.csv_rows(directory/'generations.csv'))==1
    # Simulate logs from generation N that reached disk without its checkpoint.
    common.append_csv(directory/'generations.csv',[{**common.csv_rows(directory/'generations.csv')[0],'generation':1}],
                      list(common.csv_rows(directory/'generations.csv')[0]))
    common.write_json(directory/'validation/generation-0001.json',{'uncommitted':True})
    engine.train(tmp_path,'pilot',1,'old',resume=True)
    assert [r['generation'] for r in common.csv_rows(directory/'generations.csv')]==['0','1']
    assert common.read(directory/'complete.json')['generations']==2
    with gzip.open(directory/'checkpoints/latest.pkl.gz','rb') as stream:checkpoint=pickle.load(stream)
    expected=random.Random(common.epoch_seed('pilot',1))
    assert checkpoint['population'].draws==[expected.random(),expected.random()]
    assert checkpoint['population'].generation==2
