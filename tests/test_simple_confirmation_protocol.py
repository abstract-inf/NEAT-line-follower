"""The independent study cannot start from an incomplete or untraceable pilot."""
from copy import deepcopy
import gzip
import pickle
from types import SimpleNamespace

import pytest

from neat_simple_study import confirmation as protocol


@pytest.fixture
def decision_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    from neat_quality_followup.training import training_signature
    pilot = tmp_path / 'runs' / protocol.PILOT_STUDY
    prepared = pilot / 'prepared'
    plan = {'study': protocol.PILOT_STUDY, 'fitness_version': protocol.VERSION,
            'settings': {'generations': 50}, 'evolution_seed': 719,
            'arms': {name: {'smoothness_weight': weight, 'parent': 'inputs/fast-parent.pkl',
                            'initial_population': f'initial/{name}.pkl.gz'} for name, weight in protocol.ARMS.items()}}
    for relative, data in {'inputs/fast-parent.pkl': b'parent',
                           'inputs/reference-validation.json': b'{}',
                           'suite/suite.json': b'{"splits": {}}',
                           **{f'initial/{arm}.pkl.gz': arm.encode() for arm in protocol.ARMS},
                           'source/neat_simple_study/fitness.py': b'# archived scorer; not current source\n'}.items():
        destination = prepared / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    plan['source_files'] = {'neat_simple_study/fitness.py': protocol.digest(prepared / 'source/neat_simple_study/fitness.py')}
    plan['prepared_files'] = {path.relative_to(prepared).as_posix(): protocol.digest(path)
        for path in prepared.rglob('*') if path.is_file() and 'source' not in path.relative_to(prepared).parts}
    protocol.write_json(pilot / 'prepared/protocol.json', plan)
    fingerprint = protocol.digest(pilot / 'prepared/protocol.json')
    (prepared / 'protocol.sha256').write_text(fingerprint)
    blobs = {'protocol': (prepared / 'protocol.json').read_bytes(), 'analysis-source': b'# archived analysis\n'}
    for arm in plan['arms']:
        signature = training_signature(pilot, plan, arm, protocol.VERSION, condition='simple-reward-development')
        directory = pilot / 'development' / arm
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        checkpoint.parent.mkdir(parents=True)
        with gzip.open(checkpoint, 'wb') as stream:
            pickle.dump({'population': SimpleNamespace(generation=50), 'signature': signature,
                         'training_episodes': 30000}, stream)
        protocol.write_json(directory / 'complete.json',
                            {'generations': 50, 'signature': signature, 'training_episodes': 30000})
        blobs[f'{arm}/complete'] = (directory / 'complete.json').read_bytes()
        blobs[f'{arm}/checkpoint'] = checkpoint.read_bytes()
    output = tmp_path / 'results' / protocol.PILOT_STUDY
    manifest = output / 'snapshots/staging/manifest.json'
    manifest.parent.mkdir(parents=True)
    entries = {}
    for index, (role, data) in enumerate(blobs.items()):
        payload = manifest.parent / f'{index:04d}.bin'
        payload.write_bytes(data)
        entries[role] = {'file': payload.name, 'sha256': protocol.digest(payload)}
    (manifest.parent / 'analysis.py').write_bytes(blobs['analysis-source'])
    protocol.write_json(manifest, {'inputs': entries})
    address = protocol.digest(manifest)
    manifest.parent.rename(output / 'snapshots' / address)
    report = {'complete': True, 'protocol_sha256': fingerprint, 'snapshot_sha256': address,
              'arms': {arm: {'complete': True, 'committed_generations': 50} for arm in protocol.ARMS}}
    protocol.write_json(output / 'report.json', report)
    choice = {'pilot_study': protocol.PILOT_STUDY, 'pilot_protocol_sha256': fingerprint,
              'smoothness_weight': 1., 'rationale': 'Fixture choice only.',
              'pilot_snapshot_sha256': address, 'pilot_report_sha256': protocol.digest(output / 'report.json')}
    path = tmp_path / 'decision.json'
    protocol.write_json(path, choice)
    return path, pilot, output, choice


def test_completed_pilot_decision_is_traceable(decision_tree):
    path, _, _, expected = decision_tree
    actual, _ = protocol.checked_decision(path)
    assert actual == expected


@pytest.mark.parametrize('change', ['missing-arm', 'short-budget', 'checkpoint-count', 'signature',
    'training-count', 'untested-coefficient', 'report', 'report-count', 'report-hash',
    'snapshot', 'payload', 'analysis-source', 'frozen-source', 'prepared-geometry', 'missing-study', 'legacy-study'])
def test_incomplete_or_changed_development_cannot_open_confirmation(decision_tree, change):
    path, pilot, output, decision = decision_tree
    arm_path = pilot / 'development/lambda-10/complete.json'
    if change == 'missing-arm':
        arm_path.unlink()
    elif change == 'short-budget':
        value = protocol.read(arm_path)
        value['generations'] = 49
        protocol.write_json(arm_path, value)
    elif change in ('checkpoint-count', 'signature', 'training-count'):
        checkpoint = pilot / 'development/lambda-10/checkpoints/latest.pkl.gz'
        with gzip.open(checkpoint, 'rb') as stream:
            saved = pickle.load(stream)
        if change == 'checkpoint-count':
            saved['population'].generation = 49
        elif change == 'signature':
            saved['signature']['fitness_version'] = 'different'
        else:
            saved['training_episodes'] -= 6
        with gzip.open(checkpoint, 'wb') as stream:
            pickle.dump(saved, stream)
    elif change == 'untested-coefficient':
        choice = deepcopy(decision)
        choice['smoothness_weight'] = 5.
        protocol.write_json(path, choice)
    elif change in ('report', 'report-count', 'report-hash'):
        value = protocol.read(output / 'report.json')
        if change == 'report':
            value['complete'] = False
        elif change == 'report-count':
            value['arms']['lambda-10']['committed_generations'] = 20
        else:
            value['changed'] = True
        protocol.write_json(output / 'report.json', value)
        if change != 'report-hash':
            decision['pilot_report_sha256'] = protocol.digest(output / 'report.json')
            protocol.write_json(path, decision)
    elif change == 'snapshot':
        (output / 'snapshots' / decision['pilot_snapshot_sha256'] / 'manifest.json').write_text('{}')
    elif change in ('payload', 'analysis-source'):
        snapshot = output / 'snapshots' / decision['pilot_snapshot_sha256']
        relative = ('analysis.py' if change == 'analysis-source' else
                    protocol.read(snapshot / 'manifest.json')['inputs']['analysis-source']['file'])
        (snapshot / relative).write_bytes(b'changed archived evidence')
    elif change == 'frozen-source':
        (pilot / 'prepared/source/neat_simple_study/fitness.py').write_bytes(b'changed scorer')
    elif change == 'prepared-geometry':
        (pilot / 'prepared/suite/suite.json').write_bytes(b'changed saved geometry manifest')
    else:
        if change == 'missing-study':
            del decision['pilot_study']
        else:
            decision['pilot_study'] = 'neat-simple-pilot-v1'
        protocol.write_json(path, decision)
    with pytest.raises((ValueError, FileNotFoundError)):
        protocol.checked_decision(path)


def test_prepare_never_launches_without_a_completed_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, 'checked_decision', lambda _: (_ for _ in ()).throw(ValueError('pilot incomplete')))
    run = tmp_path / 'confirmation'
    with pytest.raises(ValueError, match='pilot incomplete'):
        protocol.prepare(run, tmp_path / 'decision.json')
    assert not (run / 'prepared').exists()


def test_small_real_preparation_matches_trainer_schema_and_five_fresh_populations(tmp_path, monkeypatch):
    """Two generations exercise preparation contracts, not the production budget."""
    from pathlib import Path
    import gzip
    import pickle
    from line_follower.controllers import read_model_artifact
    from line_follower.tracks import Section, TrackSpec
    from neat_simple_study.confirmation_training import _prepared_batch
    from neat_smooth_study.courses import speed_course
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    monkeypatch.setattr(protocol, 'ROOT', workspace)
    monkeypatch.setattr(protocol, 'GENERATIONS', 2)
    monkeypatch.setattr(protocol, 'SETTINGS', {**protocol.SETTINGS, 'generations': 2})
    monkeypatch.setattr(protocol, 'protected_geometry', lambda _: set())
    script = workspace / 'script.py'
    script.write_text('# preparation contract fixture\n')
    monkeypatch.setattr(protocol, 'source_inventory', lambda root=None: {
        'script.py': protocol.digest((Path(root) if root is not None else workspace) / 'script.py')})
    monkeypatch.setattr(protocol, 'checked_decision', lambda _: ({'pilot_study': protocol.PILOT_STUDY,
                                                               'smoothness_weight': 1., 'pilot_snapshot_sha256': 'fixture'},
                                                              {'robot': {}, 'episode': {}}))
    decision = workspace / 'decision.json'
    protocol.write_json(decision, {'fixture': True, 'pilot_study': protocol.PILOT_STUDY,
                                   'smoothness_weight': 1.})
    protocol.write_json(workspace / 'results' / protocol.PILOT_STUDY / 'report.json', {'fixture': True})
    protocol.write_json(workspace / 'results' / protocol.PILOT_STUDY / 'snapshots/fixture/manifest.json', {'fixture': True})
    old = workspace / 'runs' / protocol.PILOT_STUDY / 'prepared/inputs'
    old.mkdir(parents=True)
    sources = {'accepted-prototype.pkl': 'neat-prototype.pkl', 'pid-100.json': 'pid-100.json', 'pid-050.json': 'pid-050.json'}
    records = {}
    for filename, manifest_name in sources.items():
        (old / filename).write_bytes(filename.encode())
        records[manifest_name] = {'sha256': protocol.digest(old / filename)}
    protocol.write_json(old / 'controller-manifest.json', {'controllers': records})
    def suite(directory, namespace, counts, used):
        result = {'splits': {}}
        serial = 0
        for split, groups in counts.items():
            entries = []
            for group, count in groups.items():
                for index in range(count):
                    serial += 1
                    if split == 'training':
                        slot = (index % 2)*10 + index//2
                        track = speed_course(81000+index, slot, split)
                        kind = 'isolated' if slot < 10 else 'post_turn'
                    else:
                        track = TrackSpec(f'fixture-{serial}', (Section('straight', length=1+serial*.01),), split=split)
                        kind = group
                    relative = f'{split}/{group}-{index:03d}.json'
                    track.save(directory / relative)
                    entries.append({'path': relative, 'group': group, 'kind': kind, 'geometry_hash': track.geometry_hash})
                    used.add(track.geometry_hash)
            result['splits'][split] = entries
        protocol.write_json(directory / 'suite.json', result)
        return result
    monkeypatch.setattr(protocol, 'make_suite', suite)
    run = workspace / 'runs/contract'
    protocol.prepare(run, decision)
    plan = protocol.verify(run)
    assert protocol.read(run / 'prepared/inputs/pilot-report.json') == {'fixture': True}
    assert protocol.read(run / 'prepared/inputs/pilot-analysis-manifest.json') == {'fixture': True}
    assert (run / 'prepared/inputs/pid-100.json').read_bytes() == b'pid-100.json'
    assert len(plan['seeds']) == 5
    identities = set()
    for seed in range(1, 6):
        item = plan['seeds'][str(seed)]
        with gzip.open(run / 'prepared' / item['initial_population'], 'rb') as stream:
            population = pickle.load(stream)['population']
        assert population.generation == 0 and len(population.population) == 100
        current = {protocol.genotype_hash(genome) for genome in population.population.values()}
        assert len(current) == 100 and not identities & current
        identities.update(current)
        template = read_model_artifact(run / 'prepared' / item['controller_template'])
        assert template['metadata']['trained'] is False
        entries, paths, _ = _prepared_batch(run, seed, 0, 6, .5)
        assert len(entries) == len(paths) == 6
    assert len(identities) == 500
    assert not (run / 'assessment/suite').exists()
