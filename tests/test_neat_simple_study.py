"""New-objective dispatch, provenance and resume checks with evolution stubbed."""
from copy import deepcopy
import csv
from functools import partial
import shutil
from types import SimpleNamespace

import pytest

import run_neat_simple_study as study
from neat_quality_followup import training
from neat_simple_study.fitness import score_batch
from test_quality_followup_training import checkpoint, prepared_run


def test_compatibility_diagnostics_never_change_breeding_fitness(prepared_run):
    rows = prepared_run.rows
    reference = deepcopy(rows)
    for row in reference:
        row['geometry_lateral_rmse_m'] *= 10
        row['command_jitter'] *= 10
    first = study.compatible_score(rows, rows, smoothness_weight=10.)
    second = study.compatible_score(rows, reference, smoothness_weight=10.)
    assert first['fitness'] == second['fitness'] == score_batch(rows, smoothness_weight=10.)['fitness']
    assert first['fitness_version'] == study.VERSION
    assert first['historical_quality_gates_are_diagnostic_only']
    noisy = deepcopy(rows)
    for row in noisy:
        row['geometry_lateral_rmse_m'] *= 5
    third = study.compatible_score(noisy, rows, smoothness_weight=10.)
    assert third['fitness'] == first['fitness']
    assert not third['quality_feasible'] and first['quality_feasible']


def test_new_scorer_runs_shared_trainer_and_resumes_exact_rng(prepared_run):
    fixture = prepared_run
    arm = 'lambda-1'
    fixture.plan['arms'][arm] = fixture.plan['arms']['fast']
    scorer = partial(study.compatible_score, smoothness_weight=1.)
    kwargs = dict(verify_protocol=lambda _: fixture.plan, fitness_function=scorer,
                  fitness_version=study.VERSION, condition='simple-reward-development')
    directory = training.train(fixture.run, arm, workers=6, stop_after=1, **kwargs)
    initial = checkpoint(directory)
    assert initial['population'].generation == 1
    assert initial['signature']['condition'] == 'simple-reward-development'
    with (directory / 'genome-fitness.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and all(float(row['smoothness_weight']) == 1. for row in rows)
    assert all(row['fitness_version'] == study.VERSION for row in rows)
    training.train(fixture.run, arm, workers=8, resume=True, **kwargs)
    saved = checkpoint(directory)
    assert saved['population'].generation == 2
    assert saved['population'].draws[:1] == initial['population'].draws
    assert saved['training_episodes'] == 48
    assert study.read(directory / 'complete.json')['generations'] == 2


def test_changed_condition_cannot_resume(prepared_run):
    fixture = prepared_run
    scorer = partial(study.compatible_score, smoothness_weight=1.)
    kwargs = dict(verify_protocol=lambda _: fixture.plan, fitness_function=scorer,
                  fitness_version=study.VERSION)
    training.train(fixture.run, 'fast', stop_after=1, condition='first', **kwargs)
    with pytest.raises(ValueError, match='Resume'):
        training.train(fixture.run, 'fast', resume=True, condition='changed', **kwargs)


def test_report_ignores_uncommitted_candidates_and_records_new_objective(prepared_run):
    from neat_simple_study.reporting import report
    fixture = prepared_run
    arm = 'lambda-1'
    fixture.plan['arms'] = {arm: {**fixture.plan['arms']['fast'], 'smoothness_weight': 1.}}
    study.write_json(fixture.run / 'prepared/protocol.json', fixture.plan)
    scorer = partial(study.compatible_score, smoothness_weight=1.)
    directory = training.train(fixture.run, arm, workers=6, stop_after=1,
        verify_protocol=lambda _: fixture.plan, fitness_function=scorer,
        fitness_version=study.VERSION, condition='simple-reward-development')
    original = directory / 'validation/generation-0000.json'
    uncommitted = study.read(original)
    for row in uncommitted['candidates']:
        row['generation'] = 1
    study.write_json(directory / 'validation/generation-0001.json', uncommitted)
    report(fixture.run)
    result = study.read(fixture.run / 'report/report.json')['arms'][arm]
    assert result['committed_generations'] == 1
    assert len(result['candidates']) == 3
    assert result['selected_development_candidate']['generation'] == 0
    assert result['actual_training_episodes'] == 24
    assert result['selected_development_candidate']['development_validation_fitness'] < 7


@pytest.mark.parametrize('change', ['none', 'input', 'source', 'budget', 'coefficient'])
def test_frozen_protocol_detects_input_and_design_changes(tmp_path, change):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    inventory = study.source_inventory()
    for relative in inventory:
        path = prepared / 'source' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(study.ROOT / relative, path)
    (prepared / 'input.json').write_text('{}')
    plan = {'study': 'neat-simple-pilot-v1', 'fitness_version': study.VERSION,
            'settings': deepcopy(study.SETTINGS), 'dependencies': study.dependencies(),
            'arms': {name: {'smoothness_weight': weight, 'parent': 'inputs/fast-parent.pkl',
                            'initial_population': 'initial/common.pkl.gz'} for name, weight in study.ARMS.items()},
            'source_files': inventory, 'prepared_files': {'input.json': study.digest(prepared / 'input.json')}}
    if change == 'budget':
        plan['settings']['generations'] += 1
    if change == 'coefficient':
        plan['arms']['lambda-1']['smoothness_weight'] = 2.
    study.write_json(prepared / 'protocol.json', plan)
    (prepared / 'protocol.sha256').write_text(study.digest(prepared / 'protocol.json'))
    if change == 'input':
        (prepared / 'input.json').write_text('{"changed":true}')
    if change == 'source':
        (prepared / 'source/neat_simple_study/fitness.py').write_text('# changed')
    if change == 'none':
        study.verify(tmp_path)
    else:
        with pytest.raises(ValueError):
            study.verify(tmp_path)


def test_dispatch_uses_six_workers_explicit_callbacks_and_no_global_changes(tmp_path, monkeypatch):
    plan = {'arms': {name: {'smoothness_weight': value} for name, value in study.ARMS.items()}}
    monkeypatch.setattr(study, 'verify', lambda _: plan)
    original = training.score_batch
    monkeypatch.setattr(study, 'digest', lambda _: 'protocol')
    replay = tmp_path / 'preflight/physics-replay.json'
    study.write_json(replay, {'all_match': True, 'protocol_sha256': 'protocol'})
    calls = []
    monkeypatch.setattr(training, 'train', lambda *args, **kwargs: calls.append((args, kwargs)))
    args = SimpleNamespace(command='pilot', run=tmp_path, arm=None, all=True, workers=6, resume=False)
    study.dispatch(args)
    assert [args[1] for args, _ in calls] == list(study.ARMS)
    assert all(args[2] == 6 for args, _ in calls)
    assert all(kwargs['condition'] == 'simple-reward-development' for _, kwargs in calls)
    assert [kwargs['fitness_function'].keywords['smoothness_weight'] for _, kwargs in calls] == [0., 1., 10.]
    assert training.score_batch is original
    with pytest.raises(ValueError, match='6, 7 or 8'):
        study.dispatch(SimpleNamespace(**{**vars(args), 'workers': 4}))
