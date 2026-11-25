"""Per-arm stock NEAT continuation on the exact prepared V1 batches."""
from datetime import datetime, timezone
import gzip
import json
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
from neat_quality_study.training import genotype_hash, trim_logs, validation_episodes
from neat_smooth_study.common import append_csv, digest, lock, read, write_json
from neat_smooth_study.engine import collect, initialize
from neat_smooth_study.metrics import quality, select
from .fitness import VERSION, score_batch
from .protocol import verify


def check_episodes(rows, hashes):
    if [row['track_hash'] for row in rows] != hashes:
        raise ValueError('Episode geometry/order differs from the prepared batch.')
    if any(row.get('cap_m_s') != 1. for row in rows):
        raise ValueError('Training episodes must use the frozen 1 m/s cap.')


def prepared_batch(run, generation, courses):
    directory = run / 'prepared/batches' / f'generation-{generation:04d}'
    hashes = read(directory / 'batch.json')['hashes']
    paths = [directory / f'{slot}.json' for slot in range(courses)]
    if len(hashes) != courses or len(set(hashes)) != courses:
        raise ValueError('Prepared batch has an invalid geometry count.')
    if [TrackSpec.load(path).geometry_hash for path in paths] != hashes:
        raise ValueError('Copied course geometry differs from the prepared batch.')
    reference_path = directory / 'reference.json'
    reference = read(reference_path)
    check_episodes(reference, hashes)
    return [str(path) for path in paths], hashes, reference, digest(reference_path)


def restore_outputs(directory, saved):
    next_generation = saved['population'].generation
    trim_logs(directory, next_generation)
    for folder in ('validation', 'champions', 'training-raw'):
        for path in (directory / folder).glob('generation-*'):
            if int(path.stem.split('-')[1]) >= next_generation:
                path.unlink()
    (directory / 'complete.json').unlink(missing_ok=True)
    if saved['best_model_bytes'] is None:
        (directory / 'best.pkl').unlink(missing_ok=True)
        (directory / 'selection.json').unlink(missing_ok=True)
    else:
        (directory / 'best.pkl').write_bytes(saved['best_model_bytes'])
        write_json(directory / 'selection.json', saved['selection'])


def training_signature(run, plan, arm, version, condition='quality-speed-followup'):
    prepared = Path(run) / 'prepared'
    return {'protocol_sha256': digest(prepared / 'protocol.json'),
            'parent_sha256': digest(prepared / plan['arms'][arm]['parent']),
            'initial_population_sha256': digest(prepared / plan['arms'][arm]['initial_population']),
            'reference_validation_sha256': digest(prepared / 'inputs/reference-validation.json'),
            'arm': arm, 'fitness_version': version, 'settings': plan['settings'],
            'evolution_seed': plan['evolution_seed'], 'condition': condition}


def train(run, arm, workers=3, resume=False, stop_after=None, *,
          verify_protocol=None, fitness_function=None, fitness_version=None,
          condition='quality-speed-followup'):
    verifier = verify if verify_protocol is None else verify_protocol
    scorer = score_batch if fitness_function is None else fitness_function
    version = VERSION if fitness_version is None else fitness_version
    run = Path(run).resolve()
    plan = verifier(run)
    if arm not in plan['arms']:
        raise ValueError(f'Unknown prepared arm: {arm}')
    if workers < 1 or (stop_after is not None and stop_after < 1):
        raise ValueError('Workers and stop_after must be positive.')
    settings = plan['settings']
    parent = run / 'prepared' / plan['arms'][arm]['parent']
    initial = run / 'prepared' / plan['arms'][arm]['initial_population']
    reference_path = run / 'prepared/inputs/reference-validation.json'
    reference = read(reference_path)
    signature = training_signature(run, plan, arm, version, condition)
    directory = run / 'development' / arm
    with lock(directory):
        if not resume and any(path.name != 'session.lock' for path in directory.iterdir()):
            raise FileExistsError('Use --resume for this existing arm.')
        for folder in ('checkpoints', 'champions', 'validation', 'validation-cache', 'training-raw'):
            (directory / folder).mkdir(exist_ok=True)
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        if resume:
            with gzip.open(checkpoint, 'rb') as stream:
                saved = pickle.load(stream)
            if saved['signature'] != signature:
                raise ValueError('Resume arm, protocol, initial population or fitness changed.')
            population = saved['population']
            random.setstate(saved['random_state'])
            total = saved['training_episodes']
            restore_outputs(directory, saved)
        else:
            with gzip.open(initial, 'rb') as stream:
                prepared = pickle.load(stream)
            population = prepared['population']
            if population.generation != 0 or len(population.population) != settings['population']:
                raise ValueError('Prepared initial population has an unexpected generation or size.')
            random.setstate(prepared['random_state'])
            total = 0
            _save_checkpoint(checkpoint, {'signature': signature, 'population': population,
                'random_state': random.getstate(), 'training_episodes': total,
                'best_model_bytes': None, 'selection': None, 'checkpoint_version': 1,
                'neat_version': '0.92'})
            write_json(directory / 'run.json', {**signature,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'validation_cache': 'Exact normalized genotype + arm/protocol/prepared initial population and config.',
                'training_reference': 'Exact V1 reference rows; no reference simulation.'})
        if not 0 <= population.generation <= settings['generations']:
            raise ValueError('Checkpoint generation is outside the frozen budget.')
        runtime = {'started_utc': datetime.now(timezone.utc).isoformat(), 'workers': workers,
                   'resume': resume, 'next_generation': population.generation}
        append_csv(directory / 'runtime.csv', [runtime], list(runtime))
        generations_this_call = 0
        if population.generation < settings['generations']:
            with mp.get_context('spawn').Pool(workers, initialize, (str(run), str(parent)),
                    maxtasksperchild=settings['worker_recycle_tasks']) as pool:
                while population.generation < settings['generations']:
                    generation = population.generation
                    started = time.perf_counter()
                    paths, hashes, baseline, reference_sha = prepared_batch(
                        run, generation, settings['courses_per_genome'])
                    stats = {}

                    def evaluate_generation(genomes, config):
                        tasks = [{'genome': genome, 'paths': paths, 'cap': 1.,
                                  'detailed': True, 'branch': 'old'} for _, genome in genomes]
                        evaluated = collect(pool, tasks, f'{arm} generation {generation} training')
                        if len(evaluated) != len(genomes):
                            raise ValueError('Training evaluator returned an incomplete population.')
                        scored, raw = [], []
                        for (key, genome), episodes in zip(genomes, evaluated):
                            check_episodes(episodes, hashes)
                            result = scorer(episodes, baseline)
                            genome.fitness = result['fitness']
                            scored.append({'generation': generation, 'genome_id': key, **result,
                                'violations': json.dumps(result['violations'], sort_keys=True)})
                            raw.append({'genome_id': key, 'episodes': episodes})
                        write_json(directory / 'training-raw' / f'generation-{generation:04d}.json',
                                   {'generation': generation, 'arm': arm, 'reference_sha256': reference_sha,
                                    'reference_episodes_new': 0, 'reference_episodes_reused': len(baseline),
                                    'genomes': raw})
                        append_csv(directory / 'genome-fitness.csv', scored, list(scored[0]))
                        episode_rows = [{'generation': generation, 'genome_id': genome.key,
                            **{key: value for key, value in row.items() if key not in ('components', 'straight_windows')}}
                            for (_, genome), episodes in zip(genomes, evaluated) for row in episodes]
                        append_csv(directory / 'training-episodes.csv', episode_rows, list(episode_rows[0]))
                        ranked = sorted((genome for _, genome in genomes), key=lambda genome: (-genome.fitness, genome.key))
                        chosen, identities = [], set()
                        for genome in ranked:
                            identity = genotype_hash(genome)
                            if identity not in identities:
                                identities.add(identity)
                                chosen.append((genome, identity))
                            if len(chosen) == settings['validation_candidates_per_generation']:
                                break
                        validation_rows, new_episodes, cached_episodes = [], 0, 0
                        for rank, (genome, identity) in enumerate(chosen):
                            model_path = directory / 'champions' / f'generation-{generation:04d}-rank-{rank}.pkl'
                            save_model(model_path, genome, config, {'generation': generation,
                                'candidate_rank': rank, 'condition': condition, 'arm': arm,
                                'fitness_version': version, 'protocol_sha256': signature['protocol_sha256'],
                                'parent_sha256': signature['parent_sha256'],
                                'initial_population_sha256': signature['initial_population_sha256'],
                                'robot': {**plan['robot'], 'max_speed': 1.}, 'episode': plan['episode'],
                                'genotype_sha256': identity})
                            cached_path = directory / 'validation-cache' / f'{identity}.json'
                            if cached_path.exists():
                                cached = read(cached_path)
                                ancestor = plan.get('inherited_validation_signatures', {}).get(arm)
                                valid_signature = cached['signature'] == signature or (
                                    ancestor is not None and cached['signature'] == ancestor)
                                if not valid_signature or cached['genotype_sha256'] != identity:
                                    raise ValueError('Validation cache belongs to another arm, config or protocol.')
                                primary, retention = cached['primary'], cached['retention']
                                cached_episodes += len(primary) + len(retention)
                            else:
                                primary, retention = validation_episodes(pool, run, genome=genome)
                                write_json(cached_path, {'signature': signature, 'genotype_sha256': identity,
                                    'primary': primary, 'retention': retention})
                                new_episodes += len(primary) + len(retention)
                            if any(row.get('cap_m_s') != 1. for row in primary) or any(row.get('cap_m_s') != .5 for row in retention):
                                raise ValueError('Validation episodes differ from the frozen caps.')
                            gate = quality(primary, retention, reference['primary'], reference['retention'])
                            active = {node[0] for node in NEATController(genome, config).network.node_evals}
                            outputs = config.genome_config.output_keys
                            gate.update(generation=generation, candidate_rank=rank, genome_id=genome.key,
                                arm=arm, genotype_sha256=identity, model_file=str(model_path.relative_to(directory)),
                                training_fitness=genome.fitness, drive_evaluated=outputs[0] in active,
                                steering_evaluated=outputs[1] in active)
                            validation_rows.append(gate)
                            print(f"arm={arm} generation={generation} candidate={rank} eligible={gate['eligible']} reasons={gate['reasons']}", flush=True)
                        write_json(directory / 'validation' / f'generation-{generation:04d}.json',
                                   {'candidates': validation_rows})
                        stats.update(training_episodes=len(genomes)*len(paths),
                            training_mean=mean(genome.fitness for _, genome in genomes), training_best=ranked[0].fitness,
                            quality_feasible_genomes=sum(row['quality_feasible'] for row in scored),
                            newly_evaluated_validation_episodes=new_episodes, cached_validation_episodes=cached_episodes,
                            validation_completion_rate=max(row['completed'] for row in validation_rows)/len(reference['primary']),
                            validation_candidates=len(validation_rows))

                    species = {key: len(species.members) for key, species in population.species.species.items()}
                    population.run(evaluate_generation, 1)
                    candidates = [candidate for path in sorted((directory / 'validation').glob('generation-*.json'))
                                  for candidate in read(path)['candidates']]
                    winner = select(candidates)
                    if winner:
                        shutil.copy2(directory / winner['model_file'], directory / 'best.pkl')
                        write_json(directory / 'selection.json', winner)
                    total += stats['training_episodes']
                    row = {'generation': generation, **stats, 'total_training_episodes': total,
                           'reference_episodes_new': 0, 'reference_episodes_reused': len(baseline),
                           'reference_sha256': reference_sha, 'batch_hashes': json.dumps(hashes),
                           'qualified_candidates': sum(candidate['eligible'] for candidate in candidates),
                           'wall_seconds': time.perf_counter()-started, 'species_sizes': json.dumps(species)}
                    append_csv(directory / 'generations.csv', [row], list(row))
                    _save_checkpoint(checkpoint, {'signature': signature, 'population': population,
                        'random_state': random.getstate(), 'training_episodes': total,
                        'best_model_bytes': (directory / 'best.pkl').read_bytes() if winner else None,
                        'selection': winner, 'checkpoint_version': 1, 'neat_version': '0.92'})
                    print(f'arm={arm} generation={generation} committed; reference episodes reused={len(baseline)}', flush=True)
                    generations_this_call += 1
                    if stop_after is not None and generations_this_call >= stop_after:
                        break
        if population.generation == settings['generations']:
            write_json(directory / 'complete.json', {'generations': population.generation, 'signature': signature,
                'training_episodes': total, 'qualified': (directory / 'best.pkl').exists(),
                'best_sha256': digest(directory / 'best.pkl') if (directory / 'best.pkl').exists() else None})
        return directory
