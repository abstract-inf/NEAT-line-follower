"""Recompute committed pilot fitness from actual stored episode measurements."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle


ROOT = Path(__file__).resolve().parents[1]


def audit(run, output):
    run = Path(run).resolve()
    hashes = {}
    def captured(path):
        data = path.read_bytes()
        hashes[path.relative_to(run).as_posix()] = hashlib.sha256(data).hexdigest()
        return data
    plan = json.loads(captured(run / 'prepared/protocol.json'))
    fingerprint = hashes['prepared/protocol.json']
    if fingerprint != (run / 'prepared/protocol.sha256').read_text().strip():
        raise ValueError('Pilot protocol differs.')
    for relative, expected in plan['prepared_files'].items():
        if hashlib.sha256((run / 'prepared' / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen pilot input differs: {relative}')
    for relative, expected in plan['source_files'].items():
        if hashlib.sha256((run / 'prepared/source' / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen pilot source differs: {relative}')
    namespace = {}
    exec(compile(captured(run / 'prepared/source/neat_simple_study/fitness.py'), '<frozen fitness>', 'exec'), namespace)
    scorer = namespace['score_batch']
    results, initial_episodes = {}, {}
    for arm, definition in plan['arms'].items():
        directory = run / 'development' / arm
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        if not checkpoint.exists():
            results[arm] = {'committed_generations': 0, 'complete': False}
            continue
        state = pickle.loads(gzip.decompress(captured(checkpoint)))
        generation_count = state['population'].generation
        if state['signature']['protocol_sha256'] != fingerprint or state['signature']['arm'] != arm:
            raise ValueError('Population checkpoint belongs to another arm/protocol.')
        data = captured(directory / 'genome-fitness.csv').decode('utf-8') if generation_count else 'generation,genome_id,fitness\n'
        records = {(int(row['generation']), int(row['genome_id'])): row for row in csv.DictReader(io.StringIO(data))
                   if row.get('generation', '').isdigit() and int(row['generation']) < generation_count}
        checked, episodes = 0, 0
        for generation in range(generation_count):
            path = directory / 'training-raw' / f'generation-{generation:04d}.json'
            data = captured(path)
            if generation < plan.get('inherited_generations', 0):
                key = path.relative_to(run).as_posix()
                if hashlib.sha256(data).hexdigest() != plan['historical_file_hashes'][key]:
                    raise ValueError('Inherited training data differs from the sealed parent.')
            raw = json.loads(data)
            expected = json.loads((run / 'prepared/batches' / f'generation-{generation:04d}/batch.json').read_text())['hashes']
            for genome in raw['genomes']:
                rows = genome['episodes']
                if [row['track_hash'] for row in rows] != expected or any(row['cap_m_s'] != 1. for row in rows):
                    raise ValueError('Training measurements differ from prepared geometry/cap.')
                result = scorer(rows, smoothness_weight=definition['smoothness_weight'])
                recorded = records[(generation, genome['genome_id'])]
                if result['fitness'] != float(recorded['fitness']) or result['fitness_version'] != recorded['fitness_version']:
                    raise ValueError(f'Logged fitness differs: {arm}, generation {generation}, genome {genome["genome_id"]}')
                checked += 1
                episodes += len(rows)
            if generation == 0:
                initial_episodes[arm] = raw['genomes']
        if checked != len(records) or episodes != state['training_episodes']:
            raise ValueError('Actual per-genome records and committed training counts differ.')
        complete = directory / 'complete.json'
        if complete.exists():
            completed = json.loads(captured(complete))
            if completed['generations'] != generation_count or completed['training_episodes'] != episodes:
                raise ValueError('Completion marker disagrees with the committed measurements.')
        results[arm] = {'committed_generations': generation_count, 'complete': complete.exists(),
                        'actual_genome_scores_recomputed': checked, 'actual_training_episodes': episodes,
                        'inherited_generations': plan.get('inherited_generations', 0),
                        'score_discrepancies': 0}
    equal = None
    if len(initial_episodes) == len(plan['arms']):
        samples = list(initial_episodes.values())
        equal = all(sample == samples[0] for sample in samples[1:])
        if not equal:
            raise ValueError('Identical generation-zero populations/courses produced different stored trajectories.')
    result = {'scope': 'Direct recomputation of actual committed breeding scores, not learning-performance or independence proof.',
              'complete': all(record['complete'] for record in results.values()),
              'protocol_sha256': fingerprint, 'arms': results,
              'all_arms_generation_zero_trajectories_exactly_equal': equal,
              'input_hashes': hashes, 'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('complete', 'arms', 'all_arms_generation_zero_trajectories_exactly_equal')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/neat-simple-pilot-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/neat-simple-pilot-v1/fitness-recomputation-audit.json')
    args = parser.parse_args()
    audit(args.run, args.output)
