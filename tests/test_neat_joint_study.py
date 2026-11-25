"""Focused joint scoring, provenance and callback tests; no evolution or physics."""
from copy import deepcopy
import csv
import shutil

import pytest

from neat_joint_study import fitness, protocol
from neat_quality_study.fitness import score_batch as original_score
from test_quality_followup_training import prepared_run


def batch(speed=.95, yaw=.1):
    return [{'track_hash': str(index), 'completed': True, 'progress': 1.,
             'geometry_lateral_rmse_m': .01, 'command_jitter': .003,
             'completion_time_s': 6/speed, 'route_length_m': 6.,
             'straight_windows': [{'observed': True, 'speed_m_s': speed,
                                  'yaw_rms_rad_s': yaw, 'steering_rms': .01}] if index < 2 else []}
            for index in range(6)]


def test_joint_feasibility_requires_speed_and_quality_with_plateau():
    reference = batch()
    good = fitness.score_batch(batch(), reference)
    slow = fitness.score_batch(batch(.94), reference)
    noisy = fitness.score_batch(batch(.95, .18), reference)
    assert good['joint_feasible'] and good['fitness'] >= 70
    assert slow['quality_only_feasible'] and not slow['joint_feasible']
    assert slow['speed_shortfall'] == pytest.approx(.2)
    assert not noisy['quality_only_feasible'] and not noisy['joint_feasible']
    assert good['fitness'] > max(slow['fitness'], noisy['fitness'])
    assert fitness.score_batch(batch(.95, .001), reference)['fitness'] == good['fitness']


def test_joint_directions_and_continuity_away_from_feasibility():
    reference = batch()
    assert fitness.score_batch(batch(.945), reference)['fitness'] > fitness.score_batch(batch(.94), reference)['fitness']
    assert fitness.score_batch(batch(.95, .165), reference)['fitness'] > fitness.score_batch(batch(.95, .18), reference)['fitness']
    a = fitness.score_batch(batch(.94), reference)['fitness']
    b = fitness.score_batch(batch(.94-1e-10), reference)['fitness']
    assert 0 < a-b < 1e-6


def test_original_sum_is_retained_but_joint_uses_maximum():
    reference, rows = batch(), batch()
    for row in rows:
        row['geometry_lateral_rmse_m'] *= 1.1
        row['command_jitter'] *= 1.1
    result = fitness.score_batch(rows, reference)
    assert result['original_quality_violation_sum'] == pytest.approx(.2)
    assert result['quality_max_violation'] == pytest.approx(.1)
    assert result['joint_violation'] == pytest.approx(.1)
    assert result['fitness'] == pytest.approx(60+9/1.1)


def test_completion_priority_incomplete_identity_and_bounds():
    reference, failed = batch(), batch()
    failed[0].update(completed=False, progress=.999999, completion_time_s=None)
    partial = fitness.score_batch(failed, reference)
    assert partial['fitness'] == original_score(failed, reference)['fitness']
    complete = batch(.1)
    for row in complete:
        row['geometry_lateral_rmse_m'] = 1e8
    assert 60 <= fitness.score_batch(complete, reference)['fitness'] < 69
    assert fitness.score_batch(complete, reference)['fitness'] > partial['fitness']
    assert 70 <= fitness.score_batch(batch(.99), reference)['fitness'] <= 79
    assert not partial['joint_feasible']


def test_saved_parent_selection_reproduces_declared_candidate():
    audit = protocol.select_parent(protocol.ROOT / 'runs/neat-quality-followup-v1')
    selected = audit['selected']
    assert len(audit['records']) == 150
    assert (selected['generation'], selected['candidate_rank']) == (33, 2)
    assert selected['genotype_sha256'] == protocol.EXPECTED_PARENT
    assert selected['minimum_straight_speed_m_s'] >= .95


def test_template_changes_only_powers_and_preserves_100_actions(tmp_path):
    original = protocol.ROOT / 'runs/neat-quality-followup-v1/development/fast/champions/generation-0033-rank-2.pkl'
    target = tmp_path / 'template.pkl'
    audit = protocol.initialization_template(original, target)
    assert audit['genotype_sha256'] == protocol.EXPECTED_PARENT
    assert audit['sensor_steps'] == 100 and audit['actions_exactly_equal']
    assert audit['only_declared_config_fields_changed']
    artifact = protocol.read_model_artifact(target)
    assert artifact['config'].genome_config.bias_mutate_power == .03
    assert artifact['config'].genome_config.weight_mutate_power == .035


@pytest.mark.parametrize('change', ['none', 'input', 'source', 'settings'])
def test_frozen_joint_source_and_inputs_are_checked(tmp_path, change):
    prepared = tmp_path / 'prepared'; prepared.mkdir()
    inventory = protocol.source_inventory()
    for relative in inventory:
        target = prepared / 'source' / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(protocol.ROOT / relative, target)
    (prepared / 'input.json').write_text('{}')
    plan = {'study': 'neat-joint-study-v1', 'settings': deepcopy(protocol.SETTINGS), 'arms': protocol.ARMS,
            'evolution_seed': protocol.EVOLUTION_SEED, 'fitness_version': fitness.VERSION,
            'quality_limits': fitness.LIMITS, 'mutation_powers': {key: list(value) for key, value in protocol.POWERS.items()},
            'dependencies': protocol.dependencies(), 'source_files': inventory,
            'prepared_files': {'input.json': protocol.digest(prepared / 'input.json')}}
    if change == 'settings':
        plan['settings']['generations'] += 1
    protocol.write_json(prepared / 'protocol.json', plan)
    (prepared / 'protocol.sha256').write_text(protocol.digest(prepared / 'protocol.json'))
    if change == 'input':
        (prepared / 'input.json').write_text('{"changed":true}')
    if change == 'source':
        with (prepared / 'source/neat_joint_study/fitness.py').open('a') as stream:
            stream.write('\n# changed\n')
    if change == 'none':
        protocol.verify(tmp_path)
    else:
        with pytest.raises(ValueError):
            protocol.verify(tmp_path)


def test_explicit_fitness_callbacks_use_shared_stubbed_trainer(prepared_run):
    from neat_quality_followup import training
    fixture = prepared_run
    fixture.plan['arms']['joint'] = fixture.plan['arms']['smooth']
    calls = []
    def scorer(rows, reference):
        calls.append(len(rows))
        return fitness.score_batch(rows, reference)
    directory = training.train(fixture.run, 'joint', stop_after=1,
        verify_protocol=lambda _: fixture.plan, fitness_function=scorer, fitness_version=fitness.VERSION)
    assert calls == [6]*4
    assert protocol.read(directory / 'run.json')['fitness_version'] == fitness.VERSION
    with (directory / 'genome-fitness.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and all(row['joint_feasible'] == 'True' for row in rows)


def test_launcher_passes_callbacks_without_mutating_globals(monkeypatch, tmp_path):
    import run_neat_joint_study as launcher
    from neat_quality_followup import training
    original = training.score_batch
    monkeypatch.setattr(protocol, 'verify', lambda _: {})
    captured = {}
    monkeypatch.setattr(training, 'train', lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    launcher.dispatch(launcher.parser().parse_args(['train', '--run', str(tmp_path), '--workers', '6']))
    assert captured['args'][1] == 'joint'
    assert captured['kwargs']['fitness_function'] is fitness.score_batch
    assert captured['kwargs']['fitness_version'] == fitness.VERSION
    assert training.score_batch is original
