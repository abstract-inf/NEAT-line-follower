"""Fitness and immutable protocol checks; no physics or population evolution."""
from copy import deepcopy
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import pytest

from neat_quality_followup import fitness, protocol
from neat_quality_study.fitness import score_batch as original_score


def batch(speed=.8):
    return [{'track_hash': str(i), 'completed': True, 'progress': 1.,
             'geometry_lateral_rmse_m': .01, 'command_jitter': .003,
             'completion_time_s': 6. / speed, 'route_length_m': 6.,
             'straight_windows': [{'observed': True, 'speed_m_s': speed,
                'yaw_rms_rad_s': .08, 'steering_rms': .01}] if i < 2 else []} for i in range(6)]


def test_completed_fitness_uses_continuous_discount_without_feasibility_jump():
    reference = batch()
    at_limit = batch(); at_limit[0]['straight_windows'][0]['yaw_rms_rad_s'] = .15
    over_limit = deepcopy(at_limit); over_limit[0]['straight_windows'][0]['yaw_rms_rad_s'] += 1e-9
    a, b = (fitness.score_batch(rows, reference) for rows in (at_limit, over_limit))
    assert b['fitness'] == pytest.approx(60 + 9 * b['speed_utility'] / (1 + b['quality_violation']))
    assert 0 < a['fitness'] - b['fitness'] < 1e-6
    faster = batch(.98); faster[0]['straight_windows'][0]['yaw_rms_rad_s'] = .151
    assert fitness.score_batch(faster, reference)['fitness'] > a['fitness']


def test_completion_priority_and_incomplete_fitness_are_preserved():
    reference = batch()
    failed = batch(.99); failed[0].update(completed=False, progress=.999999, completion_time_s=None)
    assert fitness.score_batch(failed, reference)['fitness'] == original_score(failed, reference)['fitness']
    complete = batch(.1)
    for row in complete:
        row['geometry_lateral_rmse_m'] = 1e8
    assert fitness.score_batch(complete, reference)['fitness'] > fitness.score_batch(failed, reference)['fitness']
    assert fitness.score_batch(batch(.98), reference)['fitness'] > fitness.score_batch(batch(.7), reference)['fitness']


def test_smooth_reconstruction_matches_three_saved_v1_genotypes():
    original = protocol.ROOT / 'runs/neat-quality-speed-v1'
    previous_state = random.getstate()
    try:
        random.seed(protocol.EVOLUTION_SEED)
        population = protocol.warm_start_population(original / 'prepared/inputs/warm-start-g0083.pkl', 100)
    finally:
        random.setstate(previous_state)
    gates = protocol.read(original / 'development/seed-01/validation/generation-0000.json')['candidates']
    assert len(population.population) == 100 and len(gates) == 3
    for gate in gates:
        artifact = protocol.read_model_artifact(original / 'development/seed-01' / gate['model_file'])
        assert protocol.genotype_hash(population.population[gate['genome_id']]) == gate['genotype_sha256']
        assert protocol.genotype_hash(artifact['genome']) == gate['genotype_sha256']


@pytest.mark.parametrize('fault', ['none', 'order', 'reference', 'cap'])
def test_copied_batch_identity_order_and_reference_caps(tmp_path, monkeypatch, fault):
    from line_follower.tracks import TrackSpec, Section
    monkeypatch.setattr(protocol, 'SETTINGS', {**protocol.SETTINGS, 'generations': 1})
    directory = tmp_path / 'batches/generation-0000'; directory.mkdir(parents=True)
    hashes = []
    for slot in range(6):
        track = TrackSpec(f'fixture-{slot}', (Section('straight', length=1 + slot / 10),))
        track.save(directory / f'{slot}.json'); hashes.append(track.geometry_hash)
    rows = [{'track_hash': identity, 'cap_m_s': 1.} for identity in hashes]
    manifest = {'hashes': hashes, 'paths': [f'C:/original/generation-0000/{slot}.json' for slot in range(6)]}
    if fault == 'order':
        manifest['hashes'] = list(reversed(hashes))
    if fault == 'reference':
        rows.reverse()
    if fault == 'cap':
        rows[0]['cap_m_s'] = .5
    protocol.write_json(directory / 'batch.json', manifest)
    protocol.write_json(directory / 'reference.json', rows)
    if fault == 'none':
        protocol.check_batches(tmp_path)
    else:
        with pytest.raises(ValueError):
            protocol.check_batches(tmp_path)


@pytest.mark.parametrize('fault', ['input', 'source', 'protocol', 'signature'])
def test_verify_rejects_tampering(tmp_path, fault):
    prepared = tmp_path / 'prepared'; prepared.mkdir()
    inventory = protocol.source_inventory()
    for relative in inventory:
        target = prepared / 'source' / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(protocol.ROOT / relative, target)
    (prepared / 'inputs').mkdir()
    (prepared / 'inputs/v1-protocol.json').write_text('{}')
    (prepared / 'input.json').write_text('{}')
    plan = {'study': 'neat-quality-followup-v1', 'settings': protocol.SETTINGS,
            'fitness_version': fitness.VERSION, 'quality_limits': fitness.LIMITS,
            'arms': protocol.ARMS, 'evolution_seed': protocol.EVOLUTION_SEED,
            'dependencies': protocol.dependencies(), 'source_files': inventory,
            'source_v1_protocol_sha256': protocol.digest(prepared / 'inputs/v1-protocol.json'),
            'prepared_files': {p: protocol.digest(prepared / p) for p in ('input.json', 'inputs/v1-protocol.json')}}
    protocol.write_json(prepared / 'protocol.json', plan)
    (prepared / 'protocol.sha256').write_text(protocol.digest(prepared / 'protocol.json'))
    protocol.verify(tmp_path)
    if fault == 'input':
        (prepared / 'input.json').write_text('{"changed":true}')
    elif fault == 'source':
        with (prepared / 'source/neat_quality_followup/fitness.py').open('a') as stream:
            stream.write('\n# changed\n')
    elif fault == 'protocol':
        with (prepared / 'protocol.json').open('a') as stream:
            stream.write(' ')
    else:
        plan['evolution_seed'] += 1
        protocol.write_json(prepared / 'protocol.json', plan)
        (prepared / 'protocol.sha256').write_text(protocol.digest(prepared / 'protocol.json'))
    with pytest.raises(ValueError):
        protocol.verify(tmp_path)


def test_prepare_requires_completed_v1_before_creating_run(tmp_path, monkeypatch):
    original_read = protocol.read
    def incomplete(path):
        result = original_read(path)
        if Path(path).name == 'complete.json':
            result['generations'] = 49
        return result
    monkeypatch.setattr(protocol, 'read', incomplete)
    with pytest.raises(ValueError, match='completed'):
        protocol.prepare(tmp_path / 'new-run')
    assert not (tmp_path / 'new-run').exists()


def test_launcher_dispatches_selected_arm_to_frozen_source(tmp_path, monkeypatch):
    import run_neat_quality_followup as launcher
    prepared = tmp_path / 'prepared'; prepared.mkdir()
    protocol.write_json(prepared / 'protocol.json', {'source_files': {'frozen.py': 'recorded'}})
    monkeypatch.setattr(launcher, 'source_inventory', lambda _: {'frozen.py': 'recorded'})
    captured = []
    monkeypatch.setattr(launcher.subprocess, 'run', lambda command, cwd: captured.append((command, cwd)) or SimpleNamespace(returncode=0))
    launcher.main(['train', '--run', str(tmp_path), '--arm', 'fast', '--workers', '2', '--resume'])
    command, cwd = captured[0]
    assert command[1] == str(prepared / 'source/run_neat_quality_followup.py')
    assert command[-1] == '--frozen-source' and '--resume' in command
    assert command[command.index('--arm') + 1] == 'fast' and cwd == prepared / 'source'
    assert launcher.parser().parse_args(['train']).workers == 3


def test_launcher_requires_explicit_arm(monkeypatch):
    import run_neat_quality_followup as launcher
    monkeypatch.setattr(protocol, 'verify', lambda _: {})
    with pytest.raises(ValueError, match='exactly one'):
        launcher.dispatch(launcher.parser().parse_args(['train']))
