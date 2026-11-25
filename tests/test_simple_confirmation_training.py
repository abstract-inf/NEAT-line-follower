"""Prospective confirmation checks with stubbed evolution/workers; no physics."""
from copy import deepcopy
import csv
from dataclasses import dataclass
import gzip
import pickle
import random
from types import SimpleNamespace

import pytest

from line_follower.tracks import Section, TrackSpec
from neat_simple_study import confirmation_training as training
from neat_smooth_study.common import append_csv, read, write_json


@dataclass
class StubGenome:
    key: int
    trait: int
    fitness: float | None = None


class StubPopulation:
    def __init__(self):
        self.generation = 0
        self.population = {key: StubGenome(key, key) for key in range(1, 5)}
        self.config = SimpleNamespace(genome_config=SimpleNamespace(
            bias_mutate_power=.12, weight_mutate_power=.14, node_indexer=0, output_keys=(0, 1)))
        self.species = SimpleNamespace(species={1: SimpleNamespace(members=self.population)})
        self.draws = []

    def run(self, evaluate, generations):
        assert generations == 1
        evaluate(list(self.population.items()), self.config)
        self.draws.append(random.random())
        self.config.genome_config.node_indexer += 1
        self.generation += 1


class StubPool:
    def __init__(self, workers, initializer, initargs, maxtasksperchild):
        self.workers, self.initargs = workers, initargs
        assert initializer is training.initialize and maxtasksperchild == 128

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def table(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def checkpoint(directory):
    with gzip.open(directory / 'checkpoints/latest.pkl.gz', 'rb') as stream:
        return pickle.load(stream)


@pytest.fixture
def prepared_run(tmp_path, monkeypatch):
    prepared = tmp_path / 'prepared'
    (prepared / 'initial').mkdir(parents=True)
    plan = {'settings': {'generations': 2, 'population': 4, 'courses_per_genome': 6,
                         'worker_recycle_tasks': 128},
            'smoothness_weight': 1., 'robot': {}, 'episode': {},
            'seeds': {'1': {'initial_population': 'initial/seed-01.pkl.gz',
                            'controller_template': 'initial/seed-01-template.pkl', 'evolution_seed': 719}}}
    initial_state = random.Random(719).getstate()
    with gzip.open(prepared / 'initial/seed-01.pkl.gz', 'wb') as stream:
        pickle.dump({'population': StubPopulation(), 'random_state': initial_state}, stream)
    (prepared / 'initial/seed-01-template.pkl').write_bytes(b'fresh config template')
    write_json(prepared / 'protocol.json', plan)
    tracks = [TrackSpec(f'train-{index}', (Section('straight', length=2 + index*.1),)) for index in range(6)]
    for generation, cap in enumerate((.5, 1.)):
        entries = []
        for index, (track, group) in enumerate(zip(tracks, training.curriculum.BATCH_GROUPS)):
            relative = f'batches/seed-01/generation-{generation:04d}/{index}.json'
            track.save(prepared / relative)
            entries.append({'path': relative, 'geometry_hash': track.geometry_hash, 'group': group})
        write_json(prepared / f'batches/seed-01/generation-{generation:04d}/batch.json',
                   {'cap_m_s': cap, 'entries': entries})
    suite = {'splits': {}}
    for split, offset in (('validation', 3.), ('retention', 4.)):
        entries = []
        for index in range(2):
            track = TrackSpec(f'{split}-{index}', (Section('straight', length=offset + index*.1),))
            relative = f'{split}/{index}.json'
            track.save(prepared / 'suite' / relative)
            entries.append({'path': relative, 'geometry_hash': track.geometry_hash, 'group': 'mixed'})
        suite['splits'][split] = entries
    write_json(prepared / 'suite/suite.json', suite)
    monkeypatch.setattr(training.curriculum, 'speed_cap', lambda generation: (.5, 1.)[generation])
    monkeypatch.setattr(training.curriculum, 'SPEED_SCHEDULE', ((0, .5), (1, 1.)))
    monkeypatch.setattr(training.mp, 'get_context', lambda _: SimpleNamespace(Pool=StubPool))
    monkeypatch.setattr(training, 'genotype_hash', lambda genome: f'trait-{genome.trait}')
    monkeypatch.setattr(training, 'NEATController', lambda *args: SimpleNamespace(
        network=SimpleNamespace(node_evals=[(0,), (1,)])))

    def save_model(path, genome, config, metadata):
        path.write_bytes(pickle.dumps({'genome': genome, 'config': config, 'metadata': metadata}))

    monkeypatch.setattr(training, 'save_model', save_model)
    calls, behavior = [], {'fail_training': False, 'wrong_cap': False}

    def collect(pool, tasks, label):
        calls.append((label, deepcopy(tasks), pool.workers, pool.initargs))
        if label.endswith(' training') and behavior['fail_training']:
            raise RuntimeError('Stubbed evaluation interruption')
        result = []
        for task in tasks:
            assert task.get('genome') is not None and 'model' not in task
            rows = []
            for path in task['paths']:
                track = TrackSpec.load(path)
                cap = task['cap']
                rows.append({'track_hash': track.geometry_hash, 'completed': True,
                    'progress': 1., 'completion_time_s': track.route.length / (.8*cap),
                    'route_length_m': track.route.length, 'command_jitter': .01,
                    'cap_m_s': .25 if behavior['wrong_cap'] else cap, 'error': None,
                    'score': -1e20, 'legacy_return': -1e30,
                    'group': task.get('groups', {}).get(track.geometry_hash, 'training'),
                    'geometry_lateral_rmse_m': .02 if task['detailed'] else None,
                    'straight_windows': []})
            result.append(rows)
        return result

    monkeypatch.setattr(training, 'collect', collect)
    return SimpleNamespace(run=tmp_path, plan=plan, calls=calls, behavior=behavior,
                           initial_state=initial_state, verify=lambda _: plan)


def train(fixture, **kwargs):
    return training.train(fixture.run, 1, verify_protocol=fixture.verify, **kwargs)


def test_dispatch_cache_cap_separation_and_resume_rng(prepared_run):
    fixture = prepared_run
    directory = train(fixture, workers=6, stop_after=1)
    first = checkpoint(directory)
    assert first['population'].generation == 1
    assert not (directory / 'best.pkl').exists()
    assert not (directory / 'complete.json').exists()
    assert first['signature']['smoothness_weight'] == 1.
    assert 'workers' not in first['signature']
    assert len(list((directory / 'validation-cache').glob('*.json'))) == 2
    raw = read(directory / 'training-raw/generation-0000.json')
    assert all(row['geometry_lateral_rmse_m'] is None for genome in raw['genomes'] for row in genome['episodes'])
    assert {row['group'] for row in raw['genomes'][0]['episodes']} == {'mixed', 'speed', 'corner'}
    train(fixture, workers=8, resume=True)
    saved = checkpoint(directory)
    rng = random.Random()
    rng.setstate(fixture.initial_state)
    assert saved['population'].draws == [rng.random(), rng.random()]
    assert saved['population'].config.genome_config.node_indexer == 2
    assert saved['training_episodes'] == 48
    assert read(directory / 'selection.json')['generation'] == 1
    assert read(directory / 'complete.json')['generations'] == 2
    assert len(list((directory / 'validation-cache').glob('*.json'))) == 3
    generations = table(directory / 'generations.csv')
    assert [int(row['newly_evaluated_validation_episodes']) for row in generations] == [4, 2]
    assert [int(row['cached_validation_episodes']) for row in generations] == [0, 2]
    assert [row['workers'] for row in table(directory / 'runtime.csv')] == ['6', '8']
    for label, tasks, _, arguments in fixture.calls:
        assert arguments[1].endswith('seed-01-template.pkl')
        assert all(task['detailed'] == (not label.endswith(' training')) for task in tasks)
    assert len(table(directory / 'genome-fitness.csv')) == 8
    assert all(float(row['fitness']) > 6 for row in table(directory / 'genome-fitness.csv'))


def test_resume_removes_uncommitted_outputs_and_restores_selection(prepared_run):
    fixture = prepared_run
    # Both generations are in the final stage; identical validation ties keep generation zero.
    training.curriculum.speed_cap = lambda _: 1.
    training.curriculum.SPEED_SCHEDULE = ((0, 1.),)
    path = fixture.run / 'prepared/batches/seed-01/generation-0000/batch.json'
    batch = read(path)
    batch['cap_m_s'] = 1.
    write_json(path, batch)
    directory = train(fixture, stop_after=1)
    expected_best = (directory / 'best.pkl').read_bytes()
    (directory / 'best.pkl').write_bytes(b'uncommitted candidate')
    write_json(directory / 'selection.json', {'uncommitted': True})
    for folder, suffix in (('validation', '.json'), ('training-raw', '.json'), ('champions', '.pkl')):
        (directory / folder / f'generation-0001{suffix}').write_bytes(b'uncommitted')
    for name in ('generations.csv', 'genome-fitness.csv'):
        row = table(directory / name)[0]
        append_csv(directory / name, [{**row, 'generation': 1}], list(row))
    train(fixture, resume=True)
    assert read(directory / 'selection.json')['generation'] == 0
    assert (directory / 'best.pkl').read_bytes() == expected_best
    assert len(table(directory / 'generations.csv')) == 2
    assert len(table(directory / 'genome-fitness.csv')) == 8
    assert table(directory / 'generations.csv')[1]['newly_evaluated_validation_episodes'] == '0'
    assert table(directory / 'generations.csv')[1]['cached_validation_episodes'] == '4'


def test_initial_checkpoint_survives_interruption(prepared_run):
    fixture = prepared_run
    fixture.behavior['fail_training'] = True
    with pytest.raises(RuntimeError, match='interruption'):
        train(fixture)
    saved = checkpoint(fixture.run / 'seed-01')
    assert saved['population'].generation == 0
    assert saved['random_state'] == fixture.initial_state
    fixture.behavior['fail_training'] = False
    train(fixture, resume=True, stop_after=1)
    assert checkpoint(fixture.run / 'seed-01')['population'].generation == 1


@pytest.mark.parametrize('change', ['weight', 'initial', 'template', 'protocol'])
def test_changed_resume_signature_is_rejected(prepared_run, change):
    fixture = prepared_run
    train(fixture, stop_after=1)
    if change == 'weight':
        fixture.plan['smoothness_weight'] = 10.
    else:
        relative = {'initial': 'initial/seed-01.pkl.gz', 'template': 'initial/seed-01-template.pkl',
                    'protocol': 'protocol.json'}[change]
        with (fixture.run / 'prepared' / relative).open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError, match='Resume'):
        train(fixture, resume=True)


def test_validation_cache_scope_is_verified(prepared_run):
    fixture = prepared_run
    directory = train(fixture, stop_after=1)
    path = next(path for path in (directory / 'validation-cache').glob('*retention*'))
    cached = read(path)
    cached['cap_m_s'] = 1.
    write_json(path, cached)
    with pytest.raises(ValueError, match='cache belongs'):
        train(fixture, resume=True)
    assert checkpoint(directory)['population'].generation == 1


def test_wrong_training_cap_cannot_commit_a_generation(prepared_run):
    fixture = prepared_run
    fixture.behavior['wrong_cap'] = True
    with pytest.raises(ValueError, match='cap'):
        train(fixture)
    assert checkpoint(fixture.run / 'seed-01')['population'].generation == 0


def test_cap_one_before_final_stage_is_not_selected(prepared_run):
    fixture = prepared_run
    training.curriculum.speed_cap = lambda _: 1.
    path = fixture.run / 'prepared/batches/seed-01/generation-0000/batch.json'
    batch = read(path)
    batch['cap_m_s'] = 1.
    write_json(path, batch)
    directory = train(fixture, stop_after=1)
    assert not read(directory / 'validation/generation-0000.json')['selection_eligible']
    assert not (directory / 'best.pkl').exists()
    train(fixture, resume=True)
    assert read(directory / 'selection.json')['generation'] == 1
