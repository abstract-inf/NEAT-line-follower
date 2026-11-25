"""Prospective, inactive from-scratch curriculum helpers; no study is launched.

The 400-generation budget and speed stages are design choices, not a demonstrated
minimum. A future trainer must keep one population, fitness and NEAT config
throughout the schedule; these helpers never run evolution or save artifacts.
"""
from dataclasses import replace
from pathlib import Path
import random

import neat
import numpy as np

from line_follower.controllers import load_neat_config
from line_follower.suites import corner_track, derived_seed
from line_follower.tracks import GeneratorConfig, TrackGenerationError, generate_track, validate_track


GENERATIONS = 400
SPEED_SCHEDULE = ((0, .5), (150, .65), (200, .8), (250, 1.))
BATCH_GROUPS = ('mixed', 'mixed', 'mixed', 'speed', 'speed', 'corner')
NEAT_CONFIG = Path(__file__).resolve().parents[1] / 'line_follower/configs/neat-corner.txt'


def _generation(generation):
    if type(generation) is not int or not 0 <= generation < GENERATIONS:
        raise ValueError('Training generation must be an integer in [0, 400).')


def speed_cap(generation):
    """Return the declared cap for one zero-indexed training generation."""
    _generation(generation)
    return next(cap for start, cap in reversed(SPEED_SCHEDULE) if generation >= start)


def difficulty_profile(generation):
    """Return the three mixed difficulties and the one corner difficulty."""
    _generation(generation)
    if generation < 50:
        return ('beginner',) * 3, 'beginner'
    if generation < 100:
        return ('beginner', 'intermediate', 'intermediate'), 'intermediate'
    return ('beginner', 'intermediate', 'advanced'), 'advanced'


def corner_spec(generation):
    """Cycle drill slots, independently alternating the first turn direction."""
    _, difficulty = difficulty_profile(generation)
    stage = {'beginner': 0, 'intermediate': 1, 'advanced': 3}[difficulty]
    return stage, generation % 3, 1 if generation % 2 == 0 else -1


def speed_bank_slots(seed, generation):
    """Use each isolated and post-turn entry once per ten-generation cycle."""
    _generation(generation)
    cycle, offset = divmod(generation, 10)
    rng = np.random.default_rng(derived_seed(seed, 'simple-speed-bank-cycle', cycle))
    return int(rng.permutation(10)[offset]), 10 + int(rng.permutation(10)[offset])


def _check_bank(bank, forbidden):
    if len(bank) != 20:
        raise ValueError('The speed bank requires 20 entries: 10 isolated, then 10 post-turn.')
    hashes = [track.geometry_hash for track in bank]
    if len(set(hashes)) != 20 or set(hashes) & forbidden:
        raise ValueError('Speed-bank geometries must be unique and disjoint from protected tracks.')
    for index, track in enumerate(bank):
        isolated = all(section.kind == 'straight' for section in track.sections)
        if isolated != (index < 10) or track.sections[-1].kind != 'straight' or validate_track(track):
            raise ValueError('Invalid speed-bank geometry or isolated/post-turn ordering.')
    return set(hashes)


def generation_batch(seed, generation, bank, forbidden):
    """Return three fresh mixed, two bank speed and one fresh corner course.

    Geometry generation uses local deterministic streams. No caller collection
    or global random state is changed. Earlier fresh mixed/corner hashes may be
    added to ``forbidden``; speed-bank entries intentionally repeat each cycle.
    """
    mixed, corner_difficulty = difficulty_profile(generation)
    forbidden = set(forbidden)
    used = forbidden | _check_bank(bank, forbidden)
    tracks = []
    stage, corner_slot, direction = corner_spec(generation)
    for slot, difficulty in enumerate((*mixed, corner_difficulty)):
        for attempt in range(100):
            track_seed = derived_seed(seed, 'simple-from-scratch-batch', generation, slot, attempt)
            try:
                if slot < 3:
                    track = generate_track(track_seed, difficulty)
                else:
                    track = corner_track(track_seed, stage, corner_slot, difficulty, GeneratorConfig())
                    first_turn = next(section.angle for section in track.sections if section.kind == 'corner')
                    if first_turn * direction < 0:
                        track = replace(track, sections=tuple(replace(section, angle=-section.angle)
                                                             for section in track.sections),
                                        start=(track.start[0], -track.start[1], -track.start[2]))
                track = replace(track, split='training')
            except TrackGenerationError:
                continue
            if track.geometry_hash not in used and not validate_track(track):
                break
        else:
            raise ValueError(f'Fresh generation course search exhausted: generation={generation}, slot={slot}.')
        tracks.append(track)
        used.add(track.geometry_hash)
    isolated, post_turn = speed_bank_slots(seed, generation)
    return tracks[:3] + [replace(bank[isolated], split='training'),
                         replace(bank[post_turn], split='training')] + tracks[3:]


def fresh_population(seed):
    """Construct 100 stock random genomes and return their subsequent RNG state.

    No saved controller or ancestor is read. Only population size overrides the
    existing corner configuration; all mutation parameters stay constant.
    """
    if type(seed) is not int:
        raise ValueError('An explicit integer initialization seed is required.')
    caller_state = random.getstate()
    try:
        random.seed(seed)
        config = load_neat_config(NEAT_CONFIG)
        config.pop_size = 100
        population = neat.Population(config)
        return population, random.getstate()
    finally:
        random.setstate(caller_state)
