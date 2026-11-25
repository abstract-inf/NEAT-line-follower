"""Prospective from-scratch trainer; importing this module launches no study."""
from datetime import datetime, timezone
import gzip
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import shutil
from statistics import mean
import time

from line_follower.controllers import NEATController, save_model
from line_follower.experiments import _save_checkpoint
from line_follower.tracks import TrackSpec
from neat_quality_study.training import genotype_hash, trim_logs
from neat_smooth_study.common import append_csv, digest, lock, read, write_json
from neat_smooth_study.engine import collect, initialize
from . import curriculum
from .fitness import VERSION, score_batch


CONDITION = 'simple-from-scratch-v1'


def _check_rows(rows, entries, cap, *, detailed=False):
    if [row['track_hash'] for row in rows] != [entry['geometry_hash'] for entry in entries]:
        raise ValueError('Episode geometry/order differs from the prepared manifest.')
    if any(row.get('cap_m_s') != cap for row in rows):
        raise ValueError('Episode cap differs from the declared curriculum cap.')
    if detailed:
        for row in rows:
            error = row.get('geometry_lateral_rmse_m')
            if error is None or not math.isfinite(error) or error < 0:
                raise ValueError('Detailed validation requires finite geometric RMSE.')


def _prepared_batch(run, seed, generation, courses, cap):
    manifest_path = run / 'prepared/batches' / f'seed-{seed:02d}' / f'generation-{generation:04d}/batch.json'
    batch = read(manifest_path)
    entries = batch['entries']
    if (batch['cap_m_s'] != cap or len(entries) != courses
            or tuple(entry['group'] for entry in entries) != curriculum.BATCH_GROUPS
            or len({entry['geometry_hash'] for entry in entries}) != courses):
        raise ValueError('Prepared batch composition or cap differs from the curriculum.')
    paths = [run / 'prepared' / entry['path'] for entry in entries]
    if [TrackSpec.load(path).geometry_hash for path in paths] != [entry['geometry_hash'] for entry in entries]:
        raise ValueError('Prepared track geometry differs from the batch manifest.')
    return entries, [str(path) for path in paths], digest(manifest_path)


def _validation(pool, run, directory, genome, identity, signature, cap, weight):
    """Cache exact genotypes separately for every split and physical cap."""
    suite = read(run / 'prepared/suite/suite.json')['splits']
    outputs, scores = {}, {}
    new_episodes = cached_episodes = 0
    for split, current_cap in (('validation', cap), ('retention', .5)):
        entries = suite[split]
        cache_path = directory / 'validation-cache' / f'{identity}-{split}-cap-{current_cap:g}.json'
        cache_scope = {'signature': signature, 'genotype_sha256': identity,
                       'split': split, 'cap_m_s': current_cap}
        if cache_path.exists():
            cached = read(cache_path)
            if any(cached.get(key) != value for key, value in cache_scope.items()):
                raise ValueError('Validation cache belongs to a different genotype, split, cap or protocol.')
            rows = cached['episodes']
            cached_episodes += len(rows)
        else:
            tasks = [{'genome': genome,
                      'paths': [str(run / 'prepared/suite' / entry['path']) for entry in entries[start:start+4]],
                      'groups': {entry['geometry_hash']: entry['group'] for entry in entries[start:start+4]},
                      'cap': current_cap, 'detailed': True, 'branch': 'old'}
                     for start in range(0, len(entries), 4)]
            rows = [row for result in collect(pool, tasks, f'{split} cap {current_cap:g}') for row in result]
            _check_rows(rows, entries, current_cap, detailed=True)
            score_batch(rows, smoothness_weight=weight)
            write_json(cache_path, {**cache_scope, 'episodes': rows})
            new_episodes += len(rows)
        _check_rows(rows, entries, current_cap, detailed=True)
        outputs[split] = rows
        scores[split] = score_batch(rows, smoothness_weight=weight)
    return outputs, scores, new_episodes, cached_episodes


def _restore_outputs(directory, saved):
    next_generation = saved['population'].generation
    trim_logs(directory, next_generation)
    for folder in ('champions', 'validation', 'training-raw'):
        for path in (directory / folder).glob('generation-*'):
            if int(path.stem.split('-')[1]) >= next_generation:
                path.unlink()
    (directory / 'complete.json').unlink(missing_ok=True)
    if saved['selection'] is None:
        (directory / 'best.pkl').unlink(missing_ok=True)
        (directory / 'selection.json').unlink(missing_ok=True)
    else:
        (directory / 'best.pkl').write_bytes(saved['best_model_bytes'])
        write_json(directory / 'selection.json', saved['selection'])


def _species_sizes(population):
    return {key: len(species.members) for key, species in population.species.species.items()}


def train(run, seed, workers=6, resume=False, stop_after=None, verify_protocol=None):
    """Run only explicitly invoked, frozen generations; no historical gates.

    Workers are runtime-only. Checkpoints commit the post-reproduction population
    and RNG; interrupted generations are rolled back while exact caches survive.
    """
    if verify_protocol is None:
        from .confirmation import verify as verify_protocol
    run = Path(run).resolve()
    plan = verify_protocol(run)
    seed_plan = plan['seeds'].get(str(seed))
    if seed_plan is None:
        raise ValueError('Seed is absent from the frozen confirmation protocol.')
    seed = int(seed)
    if workers < 1 or (stop_after is not None and stop_after < 1):
        raise ValueError('Workers and stop_after must be positive.')
    settings = plan['settings']
    weight = plan['smoothness_weight']
    initial = run / 'prepared' / seed_plan['initial_population']
    template = run / 'prepared' / seed_plan['controller_template']
    signature = {'protocol_sha256': digest(run / 'prepared/protocol.json'),
                 'initial_population_sha256': digest(initial), 'template_sha256': digest(template),
                 'seed': seed, 'evolution_seed': seed_plan['evolution_seed'],
                 'fitness_version': VERSION, 'smoothness_weight': weight,
                 'settings': settings, 'condition': CONDITION}
    directory = run / f'seed-{seed:02d}'
    with lock(directory):
        if not resume and any(path.name != 'session.lock' for path in directory.iterdir()):
            raise FileExistsError('Use --resume for this existing seed.')
        for folder in ('checkpoints', 'champions', 'validation', 'validation-cache', 'training-raw'):
            (directory / folder).mkdir(exist_ok=True)
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        if resume:
            with gzip.open(checkpoint, 'rb') as stream:
                saved = pickle.load(stream)
            if saved['signature'] != signature:
                raise ValueError('Resume protocol, seed, initial population, template or objective changed.')
            population = saved['population']
            random.setstate(saved['random_state'])
            total, selection = saved['training_episodes'], saved['selection']
            _restore_outputs(directory, saved)
        else:
            with gzip.open(initial, 'rb') as stream:
                prepared = pickle.load(stream)
            population = prepared['population']
            if population.generation != 0 or len(population.population) != settings['population']:
                raise ValueError('Prepared initial population must have the declared fresh size and generation zero.')
            random.setstate(prepared['random_state'])
            total, selection = 0, None
            _save_checkpoint(checkpoint, {'signature': signature, 'population': population,
                'random_state': random.getstate(), 'training_episodes': total,
                'selection': None, 'best_model_bytes': None})
            write_json(directory / 'run.json', {**signature,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Prospective from-scratch training; validation selection is not final assessment.',
                'selection_rule': 'Highest compact primary-validation fitness at cap 1 in the final curriculum stage; earliest generation wins ties. Retention is reported separately.'})
        genome_config = population.config.genome_config
        if genome_config.bias_mutate_power != .12 or genome_config.weight_mutate_power != .14:
            raise ValueError('Confirmation requires the original constant mutation powers .12/.14.')
        if not 0 <= population.generation <= settings['generations']:
            raise ValueError('Checkpoint generation is outside the frozen budget.')
        runtime = {'started_utc': datetime.now(timezone.utc).isoformat(), 'workers': workers,
                   'resume': resume, 'next_generation': population.generation}
        append_csv(directory / 'runtime.csv', [runtime], list(runtime))
        generations_this_call = 0
        if population.generation < settings['generations']:
            with mp.get_context('spawn').Pool(workers, initialize, (str(run), str(template)),
                    maxtasksperchild=settings['worker_recycle_tasks']) as pool:
                while population.generation < settings['generations']:
                    generation = population.generation
                    started = time.perf_counter()
                    cap = curriculum.speed_cap(generation)
                    entries, paths, batch_sha = _prepared_batch(
                        run, seed, generation, settings['courses_per_genome'], cap)
                    species_before = _species_sizes(population)
                    stats = {}

                    def evaluate_generation(genomes, config):
                        nonlocal selection
                        tasks = [{'genome': genome, 'paths': paths, 'cap': cap,
                                  'groups': {entry['geometry_hash']: entry['group'] for entry in entries},
                                  'detailed': False, 'branch': 'old'} for _, genome in genomes]
                        evaluated = collect(pool, tasks, f'seed-{seed:02d} generation {generation} training')
                        if len(evaluated) != len(genomes):
                            raise ValueError('Training evaluator returned an incomplete population.')
                        scored, raw = [], []
                        for (key, genome), episodes in zip(genomes, evaluated):
                            _check_rows(episodes, entries, cap)
                            result = score_batch(episodes, smoothness_weight=weight)
                            genome.fitness = result['fitness']
                            scored.append({'generation': generation, 'genome_id': key, **result})
                            raw.append({'genome_id': key, 'episodes': episodes})
                        write_json(directory / 'training-raw' / f'generation-{generation:04d}.json',
                            {'generation': generation, 'seed': seed, 'cap_m_s': cap,
                             'batch_sha256': batch_sha, 'genomes': raw})
                        append_csv(directory / 'genome-fitness.csv', scored, list(scored[0]))
                        champion = min((genome for _, genome in genomes), key=lambda genome: (-genome.fitness, genome.key))
                        identity = genotype_hash(champion)
                        model_path = directory / 'champions' / f'generation-{generation:04d}.pkl'
                        save_model(model_path, champion, config, {**signature, 'generation': generation,
                            'cap_m_s': cap, 'robot': {**plan['robot'], 'max_speed': cap},
                            'episode': plan['episode'], 'genotype_sha256': identity})
                        rows, validation_scores, new, cached = _validation(
                            pool, run, directory, champion, identity, signature, cap, weight)
                        active = {node[0] for node in NEATController(champion, config).network.node_evals}
                        outputs = config.genome_config.output_keys
                        candidate = {'generation': generation, 'cap_m_s': cap,
                            'genome_id': champion.key, 'genotype_sha256': identity,
                            'model_file': model_path.relative_to(directory).as_posix(),
                            'training_fitness': champion.fitness,
                            'primary_validation_fitness': validation_scores['validation']['fitness'],
                            'primary_score': validation_scores['validation'],
                            'retention_score': validation_scores['retention'],
                            'drive_evaluated': outputs[0] in active, 'steering_evaluated': outputs[1] in active,
                            'selection_eligible': cap == 1. and generation >= curriculum.SPEED_SCHEDULE[-1][0]}
                        write_json(directory / 'validation' / f'generation-{generation:04d}.json',
                            {**candidate, 'primary': rows['validation'], 'retention': rows['retention']})
                        if candidate['selection_eligible'] and (selection is None or
                                candidate['primary_validation_fitness'] > selection['primary_validation_fitness']):
                            selection = candidate
                            shutil.copy2(model_path, directory / 'best.pkl')
                            write_json(directory / 'selection.json', selection)
                        stats.update(training_episodes=sum(map(len, evaluated)),
                            training_mean=mean(genome.fitness for _, genome in genomes),
                            training_best=champion.fitness,
                            primary_validation_fitness=candidate['primary_validation_fitness'],
                            primary_completed=validation_scores['validation']['completed'],
                            retention_completed=validation_scores['retention']['completed'],
                            newly_evaluated_validation_episodes=new, cached_validation_episodes=cached)

                    population.run(evaluate_generation, 1)
                    total += stats['training_episodes']
                    row = {'generation': generation, 'cap_m_s': cap, **stats,
                           'total_training_episodes': total, 'batch_sha256': batch_sha,
                           'wall_seconds': time.perf_counter() - started,
                           'species_before': json.dumps(species_before),
                           'species_after': json.dumps(_species_sizes(population))}
                    append_csv(directory / 'generations.csv', [row], list(row))
                    _save_checkpoint(checkpoint, {'signature': signature, 'population': population,
                        'random_state': random.getstate(), 'training_episodes': total,
                        'selection': selection,
                        'best_model_bytes': (directory / 'best.pkl').read_bytes() if selection else None})
                    print(f'seed-{seed:02d} generation={generation} cap={cap:g} committed; '
                          f'training={total}, new validation={stats["newly_evaluated_validation_episodes"]}, '
                          f'cached={stats["cached_validation_episodes"]}', flush=True)
                    generations_this_call += 1
                    if stop_after is not None and generations_this_call >= stop_after:
                        break
        if population.generation == settings['generations']:
            write_json(directory / 'complete.json', {'generations': population.generation,
                'signature': signature, 'training_episodes': total, 'selected': selection,
                'best_sha256': digest(directory / 'best.pkl') if selection else None,
                'scope': 'Validation-selected controller; no stability or final-test qualification implied.'})
    return directory
