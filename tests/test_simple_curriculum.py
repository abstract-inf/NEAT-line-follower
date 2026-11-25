"""Prospective curriculum/initialization checks; no physics or evolution."""
import random

import pytest

from neat_simple_study import curriculum
from neat_smooth_study.courses import speed_course
from line_follower.tracks import validate_track


@pytest.fixture(scope='module')
def bank():
    return [speed_course(503000 + slot, slot, 'training') for slot in range(20)]


def test_exact_speed_boundaries_and_400_generation_budget():
    assert curriculum.GENERATIONS == 400
    assert curriculum.SPEED_SCHEDULE == ((0, .5), (150, .65), (200, .8), (250, 1.))
    expected = {0: .5, 149: .5, 150: .65, 199: .65, 200: .8,
                249: .8, 250: 1., 399: 1.}
    assert {generation: curriculum.speed_cap(generation) for generation in expected} == expected
    assert sum(curriculum.speed_cap(generation) == 1. for generation in range(400)) == 150
    for generation in (-1, 400, 0.5, True):
        with pytest.raises(ValueError):
            curriculum.speed_cap(generation)


def test_difficulty_composition_and_corner_direction_are_explicit():
    assert curriculum.BATCH_GROUPS == ('mixed', 'mixed', 'mixed', 'speed', 'speed', 'corner')
    for generation in (0, 49):
        assert curriculum.difficulty_profile(generation) == (('beginner',) * 3, 'beginner')
    for generation in (50, 99):
        assert curriculum.difficulty_profile(generation) == (('beginner', 'intermediate', 'intermediate'), 'intermediate')
    for generation in (100, 399):
        assert curriculum.difficulty_profile(generation) == (('beginner', 'intermediate', 'advanced'), 'advanced')
    specs = [curriculum.corner_spec(generation) for generation in range(12)]
    assert [slot for _, slot, _ in specs] == [0, 1, 2] * 4
    assert [direction for _, _, direction in specs] == [1, -1] * 6


def test_each_speed_bank_half_cycles_once_per_ten_generations():
    for start in (0, 10, 390):
        slots = [curriculum.speed_bank_slots(41, generation) for generation in range(start, start + 10)]
        assert sorted(a for a, _ in slots) == list(range(10))
        assert sorted(b for _, b in slots) == list(range(10, 20))
    assert curriculum.speed_bank_slots(41, 10) == curriculum.speed_bank_slots(41, 10)
    assert [curriculum.speed_bank_slots(41, g) for g in range(10)] != [curriculum.speed_bank_slots(42, g) for g in range(10)]


@pytest.mark.parametrize('generation', [0, 50, 100, 101, 399])
def test_fresh_batches_are_reproducible_valid_and_disjoint(bank, generation):
    protected = {'protected-geometry'}
    before = set(protected)
    first = curriculum.generation_batch(71, generation, bank, protected)
    second = curriculum.generation_batch(71, generation, bank, protected)
    hashes = [track.geometry_hash for track in first]
    assert hashes == [track.geometry_hash for track in second]
    assert len(set(hashes)) == 6 and not set(hashes) & protected
    assert protected == before
    assert all(track.split == 'training' and not validate_track(track) for track in first)
    mixed, corner = curriculum.difficulty_profile(generation)
    assert [track.difficulty for track in first] == [*mixed, 'speed', 'speed', corner]
    isolated, recovery = curriculum.speed_bank_slots(71, generation)
    assert hashes[3:5] == [bank[isolated].geometry_hash, bank[recovery].geometry_hash]
    first_turn = next(section.angle for section in first[-1].sections if section.kind == 'corner')
    assert first_turn * curriculum.corner_spec(generation)[2] > 0
    # A fresh mixed geometry that becomes protected must be regenerated.
    excluded = curriculum.generation_batch(71, generation, bank, {hashes[0]})
    assert hashes[0] not in {track.geometry_hash for track in excluded}


def test_invalid_bank_duplicates_overlap_and_order_are_rejected(bank):
    for invalid in (bank[:-1], [bank[0], *bank[1:-1], bank[0]],
                    [bank[10], *bank[1:10], bank[0], *bank[11:]]):
        with pytest.raises(ValueError):
            curriculum.generation_batch(71, 0, invalid, set())
    with pytest.raises(ValueError):
        curriculum.generation_batch(71, 0, bank, {bank[0].geometry_hash})


def genotype(genome):
    return (tuple((key, node.bias, node.response, node.activation, node.aggregation)
                  for key, node in sorted(genome.nodes.items())),
            tuple((key, connection.weight, connection.enabled)
                  for key, connection in sorted(genome.connections.items())))


def test_fresh_random_population_is_independent_reproducible_and_restores_rng():
    caller_state = random.getstate()
    first, state = curriculum.fresh_population(509)
    assert random.getstate() == caller_state
    repeated, repeated_state = curriculum.fresh_population(509)
    other, _ = curriculum.fresh_population(510)
    assert random.getstate() == caller_state
    assert first.generation == 0 and len(first.population) == 100
    fingerprints = [genotype(genome) for genome in first.population.values()]
    assert len(set(fingerprints)) == 100
    assert fingerprints == [genotype(genome) for genome in repeated.population.values()]
    assert fingerprints != [genotype(genome) for genome in other.population.values()]
    assert state == repeated_state and state != caller_state
    assert all(genome.fitness is None for genome in first.population.values())
    config = first.config
    assert config.pop_size == 100
    assert config.genome_config.bias_mutate_power == .12
    assert config.genome_config.weight_mutate_power == .14
    original = curriculum.load_neat_config(curriculum.NEAT_CONFIG)
    # Every configured scalar remains unchanged, except the declared population size.
    for key, value in vars(original.genome_config).items():
        if isinstance(value, (int, float, str, bool)):
            assert getattr(config.genome_config, key) == value
    for part in ('reproduction_config', 'species_set_config', 'stagnation_config'):
        actual = {key: value for key, value in vars(getattr(config, part)).items()
                  if not key.startswith('_')}
        expected = {key: value for key, value in vars(getattr(original, part)).items()
                    if not key.startswith('_')}
        assert actual == expected
