"""Scientific-ordering checks; these tests do not run evolution."""
from copy import deepcopy

import pytest

from neat_quality_study.fitness import score_batch


def fixture(speed=.8):
    rows = []
    for index in range(6):
        rows.append({'track_hash':str(index), 'completed':True, 'progress':1.,
            'geometry_lateral_rmse_m':.01, 'command_jitter':.003,
            'completion_time_s':6./speed, 'route_length_m':6.,
            'straight_windows':[{'observed':True, 'speed_m_s':speed,
                'yaw_rms_rad_s':.08, 'steering_rms':.01}] if index < 2 else []})
    return rows, deepcopy(rows)


def test_quality_plateau_does_not_pay_for_unnecessary_smoothness():
    rows, baseline = fixture()
    more_smooth = deepcopy(rows)
    for row in more_smooth:
        row['geometry_lateral_rmse_m'] = .001
        row['command_jitter'] = .00001
        for window in row['straight_windows']:
            window.update(yaw_rms_rad_s=.001, steering_rms=.001)
    assert score_batch(rows, baseline)['fitness'] == score_batch(more_smooth, baseline)['fitness']


def test_acceptable_faster_policy_beats_slower_policy():
    slow, baseline = fixture(.7)
    fast, _ = fixture(.98)
    assert score_batch(fast, baseline)['fitness'] > score_batch(slow, baseline)['fitness']


def test_quality_violation_cannot_be_bought_with_speed():
    slow, baseline = fixture(.6)
    unsafe, _ = fixture(.99)
    unsafe[0]['straight_windows'][0]['yaw_rms_rad_s'] = .151
    assert score_batch(slow, baseline)['fitness'] > score_batch(unsafe, baseline)['fitness']


def test_failure_cannot_game_short_episode_or_exposure():
    complete, baseline = fixture(.1)
    for row in complete:
        row['geometry_lateral_rmse_m'] = 10.
    failed, _ = fixture(.99)
    failed[0].update(completed=False, progress=.999999, completion_time_s=None)
    assert score_batch(complete, baseline)['fitness'] > score_batch(failed, baseline)['fitness']


def test_infeasible_candidates_have_a_graded_direction():
    bad, baseline = fixture()
    worse = deepcopy(bad)
    bad[0]['straight_windows'][0]['yaw_rms_rad_s'] = .3
    worse[0]['straight_windows'][0]['yaw_rms_rad_s'] = .6
    assert score_batch(bad, baseline)['fitness'] > score_batch(worse, baseline)['fitness']


def test_missing_windows_and_identity_are_protected():
    rows, baseline = fixture()
    rows[0]['straight_windows'][0] = {'observed':False}
    result = score_batch(rows, baseline)
    assert not result['quality_feasible'] and result['minimum_straight_speed_m_s'] == 0
    rows[0]['track_hash'] = 'other'
    with pytest.raises(ValueError):score_batch(rows, baseline)


def test_reference_zero_error_uses_explicit_training_floor():
    rows, baseline = fixture()
    for row in baseline:
        row['geometry_lateral_rmse_m'] = 0.
        row['command_jitter'] = 0.
    for row in rows:
        row['geometry_lateral_rmse_m'] = .004
        row['command_jitter'] = .0004
    result = score_batch(rows, baseline)
    assert result['quality_feasible']
    assert result['geometric_rmse_limit_m'] == .005
    assert result['command_jitter_limit'] == .0005


def test_genotype_identity_ignores_key_and_fitness_but_not_weights():
    from line_follower.controllers import read_model_artifact
    from neat_quality_study.protocol import ROOT
    from neat_quality_study.training import genotype_hash
    a = read_model_artifact(ROOT/'runs/neat-smooth-study-v1/prepared/inputs/seed-01.pkl')['genome']
    b = deepcopy(a); b.key += 100; b.fitness = -999
    assert genotype_hash(a) == genotype_hash(b)
    next(iter(b.nodes.values())).bias += .001
    assert genotype_hash(a) != genotype_hash(b)


@pytest.mark.parametrize('change', ['input', 'source', 'protocol'])
def test_protocol_rejects_changed_frozen_material(tmp_path, change):
    import shutil
    from neat_smooth_study.common import write_json, digest, dependencies
    from neat_quality_study import protocol
    from neat_quality_study.fitness import VERSION, LIMITS
    prepared=tmp_path/'prepared';prepared.mkdir()
    inventory=protocol.source_inventory()
    for relative in inventory:
        target=prepared/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(protocol.ROOT/relative,target)
    (prepared/'input.json').write_text('{}')
    write_json(prepared/'protocol.json',{'settings':protocol.SETTINGS,'fitness_version':VERSION,
        'quality_limits':LIMITS,'dependencies':dependencies(),'source_files':inventory,
        'prepared_files':{'input.json':digest(prepared/'input.json')}})
    (prepared/'protocol.sha256').write_text(digest(prepared/'protocol.json'))
    protocol.verify(tmp_path)
    if change=='input':(prepared/'input.json').write_text('{"changed":true}')
    if change=='source':
        with (prepared/'source/neat_quality_study/fitness.py').open('a') as f:f.write('\n# changed\n')
    if change=='protocol':
        with (prepared/'protocol.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError):protocol.verify(tmp_path)


def test_quality_training_checkpoint_resume_with_reproduction_stubbed(tmp_path, monkeypatch):
    import json, pickle, gzip, csv, shutil, random
    from types import SimpleNamespace
    from line_follower.controllers import read_model_artifact
    from line_follower.tracks import TrackSpec, Section
    from neat_smooth_study.common import write_json, append_csv
    from neat_quality_study import training, protocol
    from test_neat_smooth_study import StubPopulation, StubPool

    original = protocol.ROOT/'runs/neat-smooth-study-v1/pilot/seed-01/smooth/champions/generation-0083.pkl'
    inputs = tmp_path/'prepared/inputs'; inputs.mkdir(parents=True)
    for name in ('warm-start-g0083.pkl', 'reference-g0134.pkl'):
        shutil.copy2(original, inputs/name)
    plan = json.loads((protocol.ROOT/'runs/neat-smooth-study-v1/prepared/protocol.json').read_text())
    plan['protected_geometry'] = []
    write_json(tmp_path/'prepared/protocol.json', plan)
    tracks = [TrackSpec(f'stub-{i}', (Section('straight', length=1+i*.1),)) for i in range(6)]
    entries = []
    for i, track in enumerate(tracks):
        entry = {'path':f'{i}.json', 'geometry_hash':track.geometry_hash,
                 'group':('speed','speed','mixed','mixed','corner','corner')[i], 'kind':'isolated'}
        entries.append(entry);track.save(tmp_path/'prepared/suite'/entry['path'])
    write_json(tmp_path/'prepared/suite/suite.json', {'splits':{split:entries for split in ('training','validation','retention')}})
    rows, _ = fixture(.99)
    for row, entry in zip(rows, entries):
        row.update(track_hash=entry['geometry_hash'], group=entry['group'])
    write_json(inputs/'reference-validation.json', {'primary':rows, 'retention':rows})
    monkeypatch.setattr(training,'verify',lambda _:plan)
    monkeypatch.setattr(training,'SETTINGS',{**protocol.SETTINGS,'generations':2,'population':2})
    monkeypatch.setattr(training,'warm_start_population',lambda *args:StubPopulation(read_model_artifact(original)))
    monkeypatch.setattr(training,'generation_batch',lambda *args:tracks)
    monkeypatch.setattr(training.mp,'get_context',lambda *args:SimpleNamespace(Pool=StubPool))
    by_hash={row['track_hash']:row for row in rows}
    def fake_collect(pool,tasks,label):
        return [[deepcopy(by_hash[TrackSpec.load(path).geometry_hash]) for path in task['paths']] for task in tasks]
    monkeypatch.setattr(training,'collect',fake_collect)
    directory=training.train(tmp_path,stop_after=1)
    assert (directory/'checkpoints/latest.pkl.gz').exists()
    assert (directory/'best.pkl').exists()
    # A torn, uncommitted generation must not affect resumed selection or logs.
    write_json(directory/'validation/generation-0001.json', {'uncommitted':True})
    with (directory/'genome-fitness.csv').open(newline='') as f:scored=list(csv.DictReader(f))
    append_csv(directory/'genome-fitness.csv',[{**scored[0],'generation':1}],list(scored[0]))
    training.train(tmp_path,resume=True)
    with gzip.open(directory/'checkpoints/latest.pkl.gz','rb') as f:saved=pickle.load(f)
    assert saved['population'].generation==2
    expected=random.Random(protocol.evolution_seed())
    assert saved['population'].draws==[expected.random(),expected.random()]
    with (directory/'genome-fitness.csv').open(newline='') as f:scored=list(csv.DictReader(f))
    assert len(scored)==4 and [r['generation'] for r in scored]==['0','0','1','1']
    assert json.loads((directory/'complete.json').read_text())['qualified']
