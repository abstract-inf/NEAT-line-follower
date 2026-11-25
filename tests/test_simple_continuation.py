"""Boundary preparation and inherited-cache checks with workers/evolution stubbed."""
from copy import deepcopy
from functools import partial
import shutil
from types import SimpleNamespace

import pytest

import run_neat_simple_continue as launcher
from line_follower.tracks import Section, TrackSpec
from neat_quality_followup import training
from neat_simple_study import continuation
from neat_smooth_study.common import dependencies, digest, read, write_json
from run_neat_simple_study import compatible_score
from test_quality_followup_training import checkpoint, prepared_run, table


@pytest.fixture
def completed_parent(prepared_run, monkeypatch):
    fixture = prepared_run
    parent = fixture.run
    prepared = parent / 'prepared'
    shutil.copy2(prepared / 'initial/fast.pkl.gz', prepared / 'initial/common.pkl.gz')
    suite, references = {'splits': {}}, {}
    for split, offset, cap in (('validation', 8., 1.), ('retention', 9., .5)):
        entries, rows = [], []
        for index in range(6):
            track = TrackSpec(f'{split}-{index}', (Section('straight', length=offset + index*.1),))
            relative = f'{split}/{index}.json'
            track.save(prepared / 'suite' / relative)
            entries.append({'path': relative, 'geometry_hash': track.geometry_hash, 'group': 'mixed'})
            row = deepcopy(fixture.rows[0])
            row.update(track_hash=track.geometry_hash, route_length_m=track.route.length,
                       cap_m_s=cap, completion_time_s=track.route.length/(.99*cap))
            rows.append(row)
        suite['splits'][split] = entries
        references['primary' if split == 'validation' else 'retention'] = rows
    write_json(prepared / 'suite/suite.json', suite)
    write_json(prepared / 'inputs/reference-validation.json', references)

    def collect(pool, tasks, label):
        results = []
        for task in tasks:
            rows = []
            for path in task['paths']:
                track = TrackSpec.load(path)
                row = deepcopy(fixture.rows[0])
                cap = task.get('cap', 1.)
                row.update(track_hash=track.geometry_hash, route_length_m=track.route.length,
                           cap_m_s=cap, completion_time_s=track.route.length/(.99*cap))
                row['straight_windows'][0]['speed_m_s'] = .99*cap
                rows.append(row)
            results.append(rows)
        return results

    monkeypatch.setattr(training, 'collect', collect)
    import neat_quality_study.training as validation_dispatch
    monkeypatch.setattr(validation_dispatch, 'collect', collect)
    settings = {**fixture.plan['settings'], 'generations': 1}
    fixture.plan.update(study='neat-simple-pilot-v1', fitness_version=continuation.VERSION,
                        settings=settings, dependencies=dependencies(),
                        arms={arm: {'parent': 'inputs/fast-parent.pkl',
                                    'initial_population': 'initial/common.pkl.gz',
                                    'smoothness_weight': weight} for arm, weight in continuation.ARMS.items()})
    source = prepared / 'source/parent-source.py'
    source.parent.mkdir()
    source.write_text('# immutable parent source\n', encoding='utf-8')
    fixture.plan['source_files'] = {'parent-source.py': digest(source)}
    fixture.plan['prepared_files'] = {path.relative_to(prepared).as_posix(): digest(path)
        for path in prepared.rglob('*') if path.is_file() and 'source' not in path.relative_to(prepared).parts
        and path.name not in ('protocol.json', 'protocol.sha256')}
    write_json(prepared / 'protocol.json', fixture.plan)
    (prepared / 'protocol.sha256').write_text(digest(prepared / 'protocol.json'))
    for arm, weight in continuation.ARMS.items():
        training.train(parent, arm, workers=6, verify_protocol=lambda _: fixture.plan,
            fitness_function=partial(compatible_score, smoothness_weight=weight),
            fitness_version=continuation.VERSION, condition=continuation.CONDITION)
    write_json(parent / 'preflight/physics-replay.json', {
        'all_match': True, 'protocol_sha256': digest(prepared / 'protocol.json'), 'episodes': 12})

    # Exercise the same preparation path with a one-plus-one test budget.
    monkeypatch.setattr(continuation, 'INHERITED_GENERATIONS', 1)
    monkeypatch.setattr(continuation, 'SETTINGS', {**settings, 'generations': 2})
    monkeypatch.setattr(continuation, 'PARENT_RUN', parent)
    monkeypatch.setattr(continuation, 'BATCH_ANCESTOR_RUN', parent)
    code = parent / 'test-code'
    code.mkdir()
    (code / 'run_neat_simple_continue.py').write_text('# minimal source inventory fixture\n')
    monkeypatch.setattr(continuation, 'ROOT', code)

    def inventory(root=code):
        return {'run_neat_simple_continue.py': digest(root / 'run_neat_simple_continue.py')}

    monkeypatch.setattr(continuation, 'source_inventory', inventory)
    return SimpleNamespace(parent=parent, run=parent / 'child', plan=fixture.plan)


def test_production_budget_is_fifty_total_not_fifty_additional():
    assert continuation.INHERITED_GENERATIONS == 20
    assert continuation.SETTINGS['generations'] == 50
    assert continuation.SETTINGS['generations'] - continuation.INHERITED_GENERATIONS == 30


def test_sealed_continuation_can_verify_without_the_original_location(completed_parent, monkeypatch):
    fixture = completed_parent
    continuation.prepare(fixture.run)
    original_exists = type(fixture.parent).exists
    monkeypatch.setattr(type(fixture.parent), 'exists', lambda path: False if path == fixture.parent else original_exists(path))
    assert continuation.verify(fixture.run)['parent_protocol_sha256'] == digest(fixture.parent / 'prepared/protocol.json')


def test_prepare_preserves_parent_history_population_rng_and_exact_batches(completed_parent):
    fixture = completed_parent
    before = continuation._file_hashes(fixture.parent / 'development')
    continuation.prepare(fixture.run)
    plan = continuation.verify(fixture.run)
    assert continuation._file_hashes(fixture.parent / 'development') == before
    assert plan['inherited_generations'] == 1
    assert plan['budget']['new_generations_per_arm'] == 1
    for arm in continuation.ARMS:
        parent_state = checkpoint(fixture.parent / 'development' / arm)
        child_state = checkpoint(fixture.run / 'development' / arm)
        assert child_state['population'].generation == 1
        assert child_state['random_state'] == parent_state['random_state']
        assert child_state['population'].draws == parent_state['population'].draws
        assert child_state['training_episodes'] == parent_state['training_episodes']
        assert child_state['signature']['protocol_sha256'] != parent_state['signature']['protocol_sha256']
        assert plan['inherited_validation_signatures'][arm] == parent_state['signature']
        for relative in continuation._file_hashes(fixture.parent / 'development' / arm):
            if relative.startswith('checkpoints/') or relative == 'session.lock':
                continue
            old = fixture.parent / 'development' / arm / relative
            assert digest(fixture.run / 'prepared/history' / arm / relative) == digest(old)
            if relative not in ('run.json', 'complete.json'):
                assert digest(fixture.run / 'development' / arm / relative) == digest(old)
        assert not (fixture.run / 'development' / arm / 'complete.json').exists()
    for generation in range(2):
        relative = f'batches/generation-{generation:04d}'
        assert continuation._file_hashes(fixture.run / 'prepared' / relative) == continuation._file_hashes(fixture.parent / 'prepared' / relative)


def test_real_shared_trainer_reuses_exact_ancestor_cache_and_stops_at_total(completed_parent):
    fixture = completed_parent
    continuation.prepare(fixture.run)
    arm = 'lambda-1'
    directory = training.train(fixture.run, arm, workers=8, resume=True,
        verify_protocol=continuation.verify,
        fitness_function=partial(compatible_score, smoothness_weight=1.),
        fitness_version=continuation.VERSION, condition=continuation.CONDITION)
    state = checkpoint(directory)
    assert state['population'].generation == 2
    assert state['training_episodes'] == 48
    assert read(directory / 'complete.json')['generations'] == 2
    assert table(directory / 'generations.csv')[1]['newly_evaluated_validation_episodes'] == '0'
    assert table(directory / 'generations.csv')[1]['cached_validation_episodes'] == '36'
    # Verification binds immutable history, not appended mutable runtime logs.
    continuation.verify(fixture.run)
    before = digest(directory / 'checkpoints/latest.pkl.gz')
    continuation.prepare(fixture.run)
    assert digest(directory / 'checkpoints/latest.pkl.gz') == before


@pytest.mark.parametrize('change', ['initial', 'history', 'source', 'budget'])
def test_changed_resume_inputs_and_frozen_design_are_rejected(completed_parent, change):
    fixture = completed_parent
    continuation.prepare(fixture.run)
    if change == 'budget':
        path = fixture.run / 'prepared/protocol.json'
        plan = read(path)
        plan['settings']['generations'] += 1
        write_json(path, plan)
        (path.parent / 'protocol.sha256').write_text(digest(path))
    else:
        relative = {'initial': 'initial/lambda-0.pkl.gz',
                    'history': 'history/lambda-0/generations.csv',
                    'source': 'source/run_neat_simple_continue.py'}[change]
        with (fixture.run / 'prepared' / relative).open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError):
        continuation.verify(fixture.run)


def test_launcher_always_resumes_and_skips_verified_fifty_complete(tmp_path, monkeypatch):
    plan = {'arms': {arm: {'smoothness_weight': weight} for arm, weight in continuation.ARMS.items()}}
    monkeypatch.setattr(launcher, 'verify', lambda _: plan)
    monkeypatch.setattr(training, 'training_signature', lambda *args, **kwargs: {'frozen': True})
    calls = []
    monkeypatch.setattr(training, 'train', lambda *args, **kwargs: calls.append((args, kwargs)))
    for arm in continuation.ARMS:
        path = tmp_path / 'development' / arm / 'checkpoints/latest.pkl.gz'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'boundary')
    args = SimpleNamespace(command='train', run=tmp_path, arm=None, all=True, workers=6)
    launcher.dispatch(args)
    assert len(calls) == 3 and all(kwargs['resume'] for _, kwargs in calls)
    assert [kwargs['fitness_function'].keywords['smoothness_weight'] for _, kwargs in calls] == [0., 1., 10.]
    calls.clear()
    write_json(tmp_path / 'development/lambda-0/complete.json',
               {'generations': 50, 'signature': {'frozen': True}})
    launcher.dispatch(SimpleNamespace(**{**vars(args), 'arm': 'lambda-0', 'all': False}))
    assert not calls
    write_json(tmp_path / 'development/lambda-0/complete.json',
               {'generations': 51, 'signature': {'frozen': True}})
    with pytest.raises(ValueError, match='50-total'):
        launcher.dispatch(SimpleNamespace(**{**vars(args), 'arm': 'lambda-0', 'all': False}))
