"""Stock NEAT evolution with a versioned genome-level quality-speed objective."""
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import shutil
import time

import numpy as np

from line_follower.controllers import NEATController, save_model
from line_follower.experiments import warm_start_population, _save_checkpoint
from line_follower.tracks import TrackSpec
from neat_smooth_study.common import digest, read, write_json, append_csv, lock
from neat_smooth_study.courses import generation_batch
from neat_smooth_study.engine import initialize, collect
from neat_smooth_study.metrics import quality, select
from .fitness import score_batch, VERSION
from .protocol import SETTINGS, evolution_seed, verify


def genotype_hash(genome):
    clone = deepcopy(genome)
    clone.key, clone.fitness = 0, None
    return hashlib.sha256(pickle.dumps(clone, protocol=4)).hexdigest()


def trim_logs(directory, next_generation):
    """Keep all runtime history, but only committed generation-dependent rows."""
    for path in directory.glob('*.csv'):
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or 'generation' not in fields:
                continue
            rows = [row for row in reader if row.get('generation', '').isdigit()
                    and int(row['generation']) < next_generation]
        temporary = path.with_suffix('.restore.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)


def validation_episodes(pool, run, genome=None, model=None, parameters=None):
    """Four-course tasks allow all workers to evaluate fixed validation courses."""
    run = Path(run)
    manifest = read(run / 'prepared/suite/suite.json')['splits']
    tasks, lengths = [], []
    for split, cap in (('validation', 1.), ('retention', .5)):
        entries = manifest[split]
        lengths.append(len(entries))
        for begin in range(0, len(entries), 4):
            chunk = entries[begin:begin+4]
            tasks.append({'genome': genome, 'model': model, 'parameters': parameters,
                          'paths': [str(run / 'prepared/suite' / e['path']) for e in chunk],
                          'groups': {e['geometry_hash']: e['group'] for e in chunk},
                          'cap': cap, 'detailed': True, 'branch': 'old'})
    evaluated = collect(pool, tasks, 'validation')
    rows = [row for chunk in evaluated for row in chunk]
    return rows[:lengths[0]], rows[lengths[0]:]


def train(run, workers=6, resume=False, stop_after=None):
    run = Path(run).resolve()
    plan = verify(run)
    directory = run / 'development/seed-01'
    parent = run / 'prepared/inputs/warm-start-g0083.pkl'
    reference_model = run / 'prepared/inputs/reference-g0134.pkl'
    reference = read(run / 'prepared/inputs/reference-validation.json')
    signature = {'protocol_sha256': digest(run / 'prepared/protocol.json'),
                 'parent_sha256': digest(parent), 'reference_sha256': digest(reference_model),
                 'evolution_seed': evolution_seed(), 'fitness_version': VERSION,
                 'settings': SETTINGS}
    with lock(directory):
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        for subfolder in ('checkpoints', 'champions', 'validation', 'validation-cache'):
            (directory / subfolder).mkdir(exist_ok=True)
        if resume:
            with gzip.open(checkpoint, 'rb') as stream:
                saved = pickle.load(stream)
            if saved['signature'] != signature:
                raise ValueError('Resume protocol, parent or fitness changed.')
            population = saved['population']
            random.setstate(saved['random_state'])
            seen, total = saved['seen_hashes'], saved['training_episodes']
            trim_logs(directory, population.generation)
            for path in (directory / 'validation').glob('generation-*.json'):
                if int(path.stem.split('-')[-1]) >= population.generation:
                    path.unlink()
            (directory / 'complete.json').unlink(missing_ok=True)
            if saved['best_model_bytes'] is None:
                (directory / 'best.pkl').unlink(missing_ok=True)
                (directory / 'selection.json').unlink(missing_ok=True)
            else:
                (directory / 'best.pkl').write_bytes(saved['best_model_bytes'])
                write_json(directory / 'selection.json', saved['selection'])
        else:
            if (directory / 'run.json').exists() or checkpoint.exists():
                raise FileExistsError('Use --resume for this existing development run.')
            random.seed(evolution_seed())
            population = warm_start_population(parent, SETTINGS['population'])
            seen, total = set(), 0
            write_json(directory / 'run.json', {**signature, 'started_utc': datetime.now(timezone.utc).isoformat(),
                       'validation_cache': 'Exact normalized genotype + frozen protocol; not stochastic sampling.'})
            _save_checkpoint(checkpoint, {'signature': signature, 'population': population,
                'random_state': random.getstate(), 'seen_hashes': seen, 'training_episodes': total,
                'best_model_bytes': None, 'selection': None, 'checkpoint_version': 2, 'neat_version': '0.92'})
        runtime = {'started_utc': datetime.now(timezone.utc).isoformat(), 'workers': workers,
                   'resume': resume, 'next_generation': population.generation}
        append_csv(directory / 'runtime.csv', [runtime], list(runtime))
        suite = read(run / 'prepared/suite/suite.json')['splits']
        bank_entries = sorted(suite['training'], key=lambda e: (e['kind'] != 'isolated', e['path']))
        bank = [TrackSpec.load(run / 'prepared/suite' / e['path']) for e in bank_entries]
        forbidden = set(plan['protected_geometry'])
        forbidden.update(e['geometry_hash'] for entries in suite.values() for e in entries)
        generations_this_call = 0
        with mp.get_context('spawn').Pool(workers, initialize, (str(run), str(parent)),
                maxtasksperchild=SETTINGS['worker_recycle_tasks']) as pool:
            while population.generation < SETTINGS['generations']:
                generation = population.generation
                started = time.perf_counter()
                tracks = generation_batch(evolution_seed(), generation, bank, forbidden | seen)
                paths = []
                for slot, track in enumerate(tracks):
                    path = directory / 'tracks' / f'generation-{generation:04d}' / f'{slot}.json'
                    track.save(path); paths.append(str(path))
                hashes = [t.geometry_hash for t in tracks]; seen.update(hashes)
                write_json(directory / 'tracks' / f'generation-{generation:04d}' / 'batch.json', {'hashes': hashes, 'paths': paths})
                baseline = collect(pool, [{'model': str(reference_model), 'paths': paths, 'detailed': True}],
                                   f'generation {generation} training reference')[0]
                write_json(directory / 'tracks' / f'generation-{generation:04d}' / 'reference.json', baseline)
                stats = {}

                def evaluate_generation(genomes, config):
                    tasks = [{'genome': genome, 'paths': paths, 'detailed': True, 'branch': 'old'}
                             for _, genome in genomes]
                    evaluated = collect(pool, tasks, f'generation {generation} training')
                    scored = []
                    for (key, genome), episodes in zip(genomes, evaluated):
                        result = score_batch(episodes, baseline)
                        genome.fitness = result['fitness']
                        scored.append({'generation': generation, 'genome_id': key, **result,
                                       'violations': json.dumps(result['violations'], sort_keys=True)})
                    append_csv(directory / 'genome-fitness.csv', scored, list(scored[0]))
                    episode_rows = [{'generation': generation, 'genome_id': genome.key,
                        **{k: v for k, v in row.items() if k not in ('components', 'straight_windows')}}
                        for (_, genome), episodes in zip(genomes, evaluated) for row in episodes]
                    append_csv(directory / 'training-episodes.csv', episode_rows, list(episode_rows[0]))
                    ranked = sorted((g for _, g in genomes), key=lambda g: (-g.fitness, g.key))
                    chosen, identities = [], set()
                    for genome in ranked:
                        identity = genotype_hash(genome)
                        if identity in identities:
                            continue
                        identities.add(identity); chosen.append((genome, identity))
                        if len(chosen) == SETTINGS['validation_candidates_per_generation']:
                            break
                    validation_rows, new_episodes = [], 0
                    for rank, (genome, identity) in enumerate(chosen):
                        model_path = directory / 'champions' / f'generation-{generation:04d}-rank-{rank}.pkl'
                        save_model(model_path, genome, config, {'generation': generation,
                            'candidate_rank': rank, 'condition': 'quality-speed-development',
                            'fitness_version': VERSION, 'protocol_sha256': signature['protocol_sha256'],
                            'parent_sha256': digest(parent), 'robot': {**plan['robot'], 'max_speed': 1.},
                            'episode': plan['episode'], 'genotype_sha256': identity})
                        cached_path = directory / 'validation-cache' / f'{identity}.json'
                        if cached_path.exists():
                            cached = read(cached_path)
                            if cached['protocol_sha256'] != signature['protocol_sha256'] or cached['genotype_sha256'] != identity:
                                raise ValueError('Validation cache belongs to another frozen input/policy.')
                            primary, retention = cached['primary'], cached['retention']
                        else:
                            primary, retention = validation_episodes(pool, run, genome=genome)
                            cached = {'protocol_sha256': signature['protocol_sha256'], 'genotype_sha256': identity,
                                      'primary': primary, 'retention': retention}
                            write_json(cached_path, cached)
                            new_episodes += len(primary) + len(retention)
                        gate = quality(primary, retention, reference['primary'], reference['retention'])
                        active = {node[0] for node in NEATController(genome, config).network.node_evals}
                        gate.update(generation=generation, candidate_rank=rank, genome_id=genome.key,
                            genotype_sha256=identity, model_file=str(model_path.relative_to(directory)),
                            training_fitness=genome.fitness, drive_evaluated=config.genome_config.output_keys[0] in active,
                            steering_evaluated=config.genome_config.output_keys[1] in active)
                        validation_rows.append(gate)
                        print(f"generation={generation} candidate={rank} eligible={gate['eligible']} reasons={gate['reasons']}", flush=True)
                    write_json(directory / 'validation' / f'generation-{generation:04d}.json', {'candidates': validation_rows})
                    stats.update(training_episodes=len(genomes)*len(paths), training_mean=float(np.mean([g.fitness for _, g in genomes])),
                        training_best=ranked[0].fitness, quality_feasible_genomes=sum(row['quality_feasible'] for row in scored),
                        newly_evaluated_validation_episodes=new_episodes,
                        validation_completion_rate=max(r['completed'] for r in validation_rows)/60.,
                        validation_candidates=len(validation_rows))

                species = {key: len(s.members) for key, s in population.species.species.items()}
                population.run(evaluate_generation, 1)
                candidates = [candidate for path in sorted((directory / 'validation').glob('generation-*.json'))
                              for candidate in read(path)['candidates']]
                winner = select(candidates)
                if winner:
                    shutil.copy2(directory / winner['model_file'], directory / 'best.pkl')
                    write_json(directory / 'selection.json', winner)
                total += stats['training_episodes']
                row = {'generation': generation, **stats, 'total_training_episodes': total,
                       'reference_episodes': len(baseline), 'qualified_candidates': sum(r['eligible'] for r in candidates),
                       'wall_seconds': time.perf_counter() - started, 'species_sizes': json.dumps(species)}
                append_csv(directory / 'generations.csv', [row], list(row))
                _save_checkpoint(checkpoint, {'signature': signature, 'population': population,
                    'random_state': random.getstate(), 'seen_hashes': seen, 'training_episodes': total,
                    'best_model_bytes': (directory / 'best.pkl').read_bytes() if winner else None,
                    'selection': winner, 'checkpoint_version': 2, 'neat_version': '0.92'})
                print(f"generation={generation} committed; feasible_training_genomes={stats['quality_feasible_genomes']}; qualified_validation={row['qualified_candidates']}", flush=True)
                generations_this_call += 1
                if stop_after is not None and generations_this_call >= stop_after:
                    break
        if population.generation == SETTINGS['generations']:
            write_json(directory / 'complete.json', {'generations': population.generation, 'signature': signature,
                'training_episodes': total, 'qualified': (directory / 'best.pkl').exists(),
                'best_sha256': digest(directory / 'best.pkl') if (directory / 'best.pkl').exists() else None})
        return directory
