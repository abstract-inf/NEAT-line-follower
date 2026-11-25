"""Trainer dispatch/resume checks with stubbed evolution and no physics."""
from copy import deepcopy
import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import pytest

from line_follower.tracks import Section, TrackSpec
from neat_smooth_study.common import append_csv, read, write_json
from neat_quality_followup import training


@dataclass
class StubGenome:
    key: int
    trait: int
    fitness: float | None = None


class StubPopulation:
    def __init__(self):
        self.generation = 0
        self.population = {key: StubGenome(key, trait) for key, trait in ((1, 1), (2, 1), (3, 2), (4, 3))}
        self.config = SimpleNamespace(genome_config=SimpleNamespace(output_keys=(0, 1), node_indexer=0))
        self.species = SimpleNamespace(species={1: SimpleNamespace(members=self.population)})
        self.draws = []

    def run(self, evaluate, generations):
        assert generations == 1
        evaluate(list(self.population.items()), self.config)
        self.draws.append(random.random())
        # This stock-like mutation bookkeeping must not invalidate phenotype caches.
        self.config.genome_config.node_indexer += 1
        self.generation += 1


class StubPool:
    def __init__(self, workers, initializer, initargs, maxtasksperchild):
        self.workers, self.initargs = workers, initargs
        assert maxtasksperchild == 128

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def table(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def checkpoint(directory):
    with gzip.open(directory / 'checkpoints/latest.pkl.gz', 'rb') as stream:
        return pickle.load(stream)


@pytest.fixture
def prepared_run(tmp_path, monkeypatch):
    prepared = tmp_path / 'prepared'
    inputs = prepared / 'inputs'
    inputs.mkdir(parents=True)
    plan = {'settings': {'generations': 2, 'population': 4, 'courses_per_genome': 6,
                        'validation_candidates_per_generation': 3, 'worker_recycle_tasks': 128},
            'evolution_seed': 7036297164298145801, 'robot': {}, 'episode': {}, 'arms': {}}
    initial_rng = random.Random(plan['evolution_seed'])
    initial_rng.random()  # Prepared warm-start construction already consumed this.
    initial_state = initial_rng.getstate()
    for arm in ('smooth', 'fast'):
        parent = inputs / f'{arm}-parent.pkl'
        parent.write_bytes(arm.encode())
        initial = prepared / 'initial' / f'{arm}.pkl.gz'
        initial.parent.mkdir(exist_ok=True)
        with gzip.open(initial, 'wb') as stream:
            pickle.dump({'population': StubPopulation(), 'random_state': initial_state}, stream)
        plan['arms'][arm] = {'parent': str(parent.relative_to(prepared)),
                             'initial_population': str(initial.relative_to(prepared))}
    write_json(prepared / 'protocol.json', plan)
    tracks = [TrackSpec(f'stub-{index}', (Section('straight', length=6 + index*.1),)) for index in range(6)]
    rows, entries = [], []
    for index, track in enumerate(tracks):
        group = ('speed', 'speed', 'mixed', 'mixed', 'corner', 'corner')[index]
        rows.append({'track_hash': track.geometry_hash, 'completed': True, 'progress': 1.,
                     'geometry_lateral_rmse_m': .01, 'command_jitter': .003,
                     'completion_time_s': 6/.99, 'route_length_m': 6., 'cap_m_s': 1., 'group': group,
                     'straight_windows': [{'observed': True, 'speed_m_s': .99,
                                           'yaw_rms_rad_s': .08, 'steering_rms': .01}],
                     'components': {'diagnostic_only': 1.}})
        entries.append({'path': f'{index}.json', 'geometry_hash': track.geometry_hash, 'group': group})
        track.save(prepared / 'suite' / f'{index}.json')
    write_json(prepared / 'suite/suite.json', {'splits': {'validation': entries, 'retention': entries}})
    retention = deepcopy(rows)
    for row in retention:
        row['cap_m_s'] = .5
        row['completion_time_s'] *= 2
        row['straight_windows'][0]['speed_m_s'] *= .5
    write_json(inputs / 'reference-validation.json', {'primary': rows, 'retention': retention})
    for generation in range(2):
        directory = prepared / 'batches' / f'generation-{generation:04d}'
        for index, track in enumerate(tracks):
            track.save(directory / f'{index}.json')
        write_json(directory / 'batch.json', {'hashes': [track.geometry_hash for track in tracks]})
        write_json(directory / 'reference.json', rows)
    monkeypatch.setattr(training, 'verify', lambda _: plan)
    monkeypatch.setattr(training.mp, 'get_context', lambda _: SimpleNamespace(Pool=StubPool))
    monkeypatch.setattr(training, 'NEATController', lambda *args: SimpleNamespace(network=SimpleNamespace(node_evals=[(0,), (1,)])))

    def save_model(path, genome, config, metadata):
        path.write_bytes(pickle.dumps({'genome': genome, 'config': config, 'metadata': metadata}))

    monkeypatch.setattr(training, 'save_model', save_model)
    by_hash = {row['track_hash']: row for row in rows}
    calls = []
    behavior = {'candidate_cap': 1., 'fail_training': False}

    def collect(pool, tasks, label):
        calls.append((label, deepcopy(tasks), pool.initargs))
        assert not any(task.get('model') for task in tasks), 'Reference simulation is forbidden.'
        if label.endswith(' training') and behavior['fail_training']:
            raise RuntimeError('Stubbed evaluation interruption')
        output = []
        for task in tasks:
            current = [deepcopy(by_hash[TrackSpec.load(path).geometry_hash]) for path in task['paths']]
            for row in current:
                cap = task.get('cap', 1.)
                row['cap_m_s'] = behavior['candidate_cap'] if label.endswith(' training') else cap
                row['completion_time_s'] /= cap
                row['straight_windows'][0]['speed_m_s'] *= cap
            output.append(current)
        return output

    monkeypatch.setattr(training, 'collect', collect)
    # Exercise the real reused validation dispatcher, with only its evaluator stubbed.
    import neat_quality_study.training as original_training
    monkeypatch.setattr(original_training, 'collect', collect)
    return SimpleNamespace(run=tmp_path, plan=plan, rows=rows, tracks=tracks,
                           calls=calls, behavior=behavior, initial_state=initial_state)


def test_copied_batches_cached_validation_and_checkpoint_resume(prepared_run):
    fixture = prepared_run
    directory = training.train(fixture.run, 'smooth', workers=3, stop_after=1)
    first = checkpoint(directory)
    assert first['population'].generation == 1
    best = (directory / 'best.pkl').read_bytes()
    assert len(read(directory / 'validation/generation-0000.json')['candidates']) == 3
    assert len(list((directory / 'validation-cache').glob('*.json'))) == 3
    assert [row['genome_id'] for row in read(directory / 'validation/generation-0000.json')['candidates']] == [1, 3, 4]
    raw = read(directory / 'training-raw/generation-0000.json')
    assert raw['genomes'][0]['episodes'][0]['straight_windows']
    assert raw['genomes'][0]['episodes'][0]['components'] == {'diagnostic_only': 1.}
    assert first['signature']['initial_population_sha256']
    assert 'workers' not in first['signature']

    # Simulate writes made before the next reproduction/checkpoint commits.
    (directory / 'best.pkl').write_bytes(b'torn model')
    write_json(directory / 'selection.json', {'uncommitted': True})
    write_json(directory / 'validation/generation-0001.json', {'uncommitted': True})
    write_json(directory / 'training-raw/generation-0001.json', {'uncommitted': True})
    (directory / 'champions/generation-0001-rank-0.pkl').write_bytes(b'uncommitted')
    for name in ('generations.csv', 'genome-fitness.csv', 'training-episodes.csv'):
        row = table(directory / name)[0]
        append_csv(directory / name, [{**row, 'generation': 1}], list(row))
    training.train(fixture.run, 'smooth', workers=1, resume=True)
    saved = checkpoint(directory)
    expected = random.Random()
    expected.setstate(fixture.initial_state)
    assert saved['population'].draws == [expected.random(), expected.random()]
    assert saved['population'].generation == 2
    assert saved['population'].config.genome_config.node_indexer == 2
    assert saved['training_episodes'] == 48
    assert len(table(directory / 'genome-fitness.csv')) == 8
    assert len(table(directory / 'training-episodes.csv')) == 48
    assert (directory / 'best.pkl').read_bytes() == best
    assert read(directory / 'selection.json')['generation'] == 0
    assert read(directory / 'complete.json')['qualified']
    generations = table(directory / 'generations.csv')
    assert [row['generation'] for row in generations] == ['0', '1']
    assert [int(row['reference_episodes_new']) for row in generations] == [0, 0]
    assert [int(row['reference_episodes_reused']) for row in generations] == [6, 6]
    assert [int(row['newly_evaluated_validation_episodes']) for row in generations] == [36, 0]
    assert [int(row['cached_validation_episodes']) for row in generations] == [0, 36]
    assert all(row['reference_sha256'] for row in generations)
    training_calls = [tasks for label, tasks, _ in fixture.calls if label.endswith(' training')]
    assert len(training_calls) == 2
    for generation, tasks in enumerate(training_calls):
        expected_directory = fixture.run / 'prepared/batches' / f'generation-{generation:04d}'
        assert all(task['cap'] == 1. and task['detailed'] for task in tasks)
        assert all(Path(path).parent == expected_directory for task in tasks for path in task['paths'])
    assert [row['workers'] for row in table(directory / 'runtime.csv')] == ['3', '1']


def test_initial_checkpoint_survives_first_evaluation_interruption(prepared_run):
    fixture = prepared_run
    fixture.behavior['fail_training'] = True
    with pytest.raises(RuntimeError, match='interruption'):
        training.train(fixture.run, 'fast')
    directory = fixture.run / 'development/fast'
    saved = checkpoint(directory)
    assert saved['population'].generation == 0 and saved['population'].draws == []
    assert saved['random_state'] == fixture.initial_state
    assert saved['best_model_bytes'] is None and saved['selection'] is None
    fixture.behavior['fail_training'] = False
    training.train(fixture.run, 'fast', resume=True, stop_after=1)
    expected = random.Random()
    expected.setstate(fixture.initial_state)
    assert checkpoint(directory)['population'].draws == [expected.random()]


def test_arm_caches_do_not_cross_even_for_identical_genotypes(prepared_run):
    fixture = prepared_run
    directories = [training.train(fixture.run, arm, stop_after=1) for arm in ('smooth', 'fast')]
    assert sum(label == 'validation' for label, _, _ in fixture.calls) == 6
    for arm, directory in zip(('smooth', 'fast'), directories):
        caches = [read(path) for path in (directory / 'validation-cache').glob('*.json')]
        assert len(caches) == 3
        assert all(row['signature']['arm'] == arm for row in caches)
        assert table(directory / 'generations.csv')[0]['newly_evaluated_validation_episodes'] == '36'
        assert all(args[1].endswith(f'{arm}-parent.pkl') for label, _, args in fixture.calls if label.startswith(arm))


@pytest.mark.parametrize('change', ['geometry', 'reference_geometry', 'reference_cap', 'candidate_cap'])
def test_batch_geometry_and_caps_are_checked(prepared_run, change):
    fixture = prepared_run
    batch = fixture.run / 'prepared/batches/generation-0000'
    if change == 'geometry':
        record = read(batch / 'batch.json')
        record['hashes'][0] = 'changed'
        write_json(batch / 'batch.json', record)
    elif change in ('reference_geometry', 'reference_cap'):
        rows = read(batch / 'reference.json')
        rows[0]['track_hash' if change == 'reference_geometry' else 'cap_m_s'] = 'changed' if change == 'reference_geometry' else .5
        write_json(batch / 'reference.json', rows)
    else:
        fixture.behavior['candidate_cap'] = .5
    with pytest.raises(ValueError, match='geometry|cap'):
        training.train(fixture.run, 'smooth')
    assert checkpoint(fixture.run / 'development/smooth')['population'].generation == 0
    assert not (fixture.run / 'development/smooth/generations.csv').exists()


def test_resume_rejects_foreign_cache_scope_and_changed_prepared_initial(prepared_run):
    fixture = prepared_run
    directory = training.train(fixture.run, 'smooth', stop_after=1)
    path = next((directory / 'validation-cache').glob('*.json'))
    cached = read(path)
    cached['signature']['arm'] = 'fast'
    write_json(path, cached)
    with pytest.raises(ValueError, match='cache belongs'):
        training.train(fixture.run, 'smooth', resume=True)
    assert checkpoint(directory)['population'].generation == 1
    initial = fixture.run / 'prepared' / fixture.plan['arms']['smooth']['initial_population']
    with initial.open('ab') as stream:
        stream.write(b'changed input')
    with pytest.raises(ValueError, match='Resume arm'):
        training.train(fixture.run, 'smooth', resume=True)
