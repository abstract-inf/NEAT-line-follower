"""Freeze two development arms against the completed V1 course schedule."""
from pathlib import Path
import random
import shutil

from line_follower.controllers import read_model_artifact
from line_follower.experiments import warm_start_population, _save_checkpoint
from line_follower.tracks import TrackSpec
from neat_quality_study.protocol import source_inventory as original_inventory, verify as verify_original
from neat_quality_study.training import genotype_hash
from neat_smooth_study.common import dependencies, digest, read, write_json, lock
from .fitness import VERSION, LIMITS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/neat-quality-followup-v1'
EVOLUTION_SEED = 7036297164298145801
SETTINGS = {'generations': 50, 'population': 100, 'courses_per_genome': 6,
            'validation_candidates_per_generation': 3, 'worker_recycle_tasks': 128,
            'mask_cache_bytes': 67108864, 'cap_m_s': 1.}
ARMS = {name: {'parent': f'inputs/{name}-parent.pkl',
               'initial_population': f'initial/{name}.pkl.gz'} for name in ('smooth', 'fast')}


def source_inventory(root=ROOT):
    inventory = original_inventory(root)
    paths = [root / 'run_neat_quality_followup.py']
    paths.extend((root / 'neat_quality_followup').rglob('*.py'))
    inventory.update({path.relative_to(root).as_posix(): digest(path) for path in paths})
    return dict(sorted(inventory.items()))


def check_batches(prepared):
    """Check the copied ordering against track geometry and saved reference rows."""
    for generation in range(SETTINGS['generations']):
        directory = prepared / 'batches' / f'generation-{generation:04d}'
        batch = read(directory / 'batch.json')
        tracks = [TrackSpec.load(directory / f'{slot}.json') for slot in range(6)]
        hashes = [track.geometry_hash for track in tracks]
        original_paths = [Path(path.replace('\\', '/')) for path in batch['paths']]
        if (len(set(hashes)) != 6 or hashes != batch['hashes'] or
                [path.name for path in original_paths] != [f'{slot}.json' for slot in range(6)] or
                any(path.parent.name != directory.name for path in original_paths)):
            raise ValueError(f'Frozen batch geometry/order differs: {generation}')
        reference = read(directory / 'reference.json')
        if [row['track_hash'] for row in reference] != hashes or any(row['cap_m_s'] != 1. for row in reference):
            raise ValueError(f'Frozen reference geometry/order/cap differs: {generation}')


def check_validation(prepared):
    reference = read(prepared / 'inputs/reference-validation.json')
    if reference['parent_sha256'] != digest(prepared / 'inputs/reference-parent.pkl'):
        raise ValueError('Cached validation reference ancestry differs.')
    suite = read(prepared / 'suite/suite.json')['splits']
    for split, key, count, cap in (('validation', 'primary', 60, 1.), ('retention', 'retention', 20, .5)):
        expected = {entry['geometry_hash']: entry['group'] for entry in suite[split]}
        rows = reference[key]
        if len(expected) != count or len(rows) != count or {row['track_hash'] for row in rows} != set(expected):
            raise ValueError('Shared validation/reference geometries differ.')
        if any(row['cap_m_s'] != cap or row['group'] != expected[row['track_hash']] for row in rows):
            raise ValueError('Shared validation reference caps/groups differ.')


def verify(run):
    prepared = Path(run) / 'prepared'
    plan = read(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != digest(prepared / 'protocol.json'):
        raise ValueError('Follow-up protocol changed after commitment.')
    if (plan['study'] != 'neat-quality-followup-v1' or plan['settings'] != SETTINGS or
            plan['fitness_version'] != VERSION or plan['quality_limits'] != LIMITS or
            plan['arms'] != ARMS or plan['evolution_seed'] != EVOLUTION_SEED):
        raise ValueError('Follow-up settings, fitness or arm signature changed.')
    if plan['dependencies'] != dependencies() or source_inventory() != plan['source_files']:
        raise ValueError('Runtime source or dependencies differ from the frozen protocol.')
    if source_inventory(prepared / 'source') != plan['source_files']:
        raise ValueError('Frozen execution source changed.')
    for relative, expected in plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError(f'Frozen follow-up input changed: {relative}')
    if digest(prepared / 'inputs/v1-protocol.json') != plan['source_v1_protocol_sha256']:
        raise ValueError('Completed V1 protocol signature differs.')
    return plan


def prepare(run):
    run = Path(run).resolve()
    if (run / 'prepared/protocol.json').exists():
        verify(run)
        print('Existing follow-up freeze verified without rewriting it.', flush=True)
        return
    original = ROOT / 'runs/neat-quality-speed-v1'
    prior = verify_original(original)
    completed = read(original / 'development/seed-01/complete.json')
    prior_hash = digest(original / 'prepared/protocol.json')
    if (len(prior['source_files']) != 49 or completed['generations'] != 50 or
            completed['signature']['protocol_sha256'] != prior_hash or
            completed['signature']['evolution_seed'] != EVOLUTION_SEED):
        raise ValueError('A completed, unchanged 50-generation V1 study is required.')
    for relative, expected in prior['source_files'].items():
        if digest(ROOT / relative) != expected or digest(original / 'prepared/source' / relative) != expected:
            raise ValueError(f'Original V1 execution source differs: {relative}')
    with lock(run):
        prepared = run / 'prepared'
        if prepared.exists():
            raise FileExistsError('Preserve partial preparation; choose a new run directory.')
        inputs = prepared / 'inputs'; inputs.mkdir(parents=True)
        originals = {
            'smooth-parent.pkl': original / 'prepared/inputs/warm-start-g0083.pkl',
            'fast-parent.pkl': original / 'prepared/inputs/reference-g0134.pkl',
            'reference-parent.pkl': original / 'prepared/inputs/reference-g0134.pkl',
            'reference-validation.json': original / 'prepared/inputs/reference-validation.json',
            'v1-protocol.json': original / 'prepared/protocol.json',
            'v1-complete.json': original / 'development/seed-01/complete.json',
            'v1-generation-0000.json': original / 'development/seed-01/validation/generation-0000.json',
        }
        gates = read(originals['v1-generation-0000.json'])['candidates']
        if len(gates) != 3:
            raise ValueError('Three saved V1 generation-zero candidates are required.')
        for gate in gates:
            originals[f"v1-generation-0000-rank-{gate['candidate_rank']}.pkl"] = original / 'development/seed-01' / gate['model_file']
        for code, cap in (('100', 1.), ('050', .5)):
            directory = ROOT / 'runs/neat-smooth-study-v1/pid' / f'cap-{code}'
            selection = read(directory / 'selection-freeze.json')
            if (not selection['eligible'] or selection['sha256'] != digest(directory / 'pid-fullspeed.json') or
                    read(directory / 'pid-fullspeed.json')['base'] != 1. or read(directory / 'protocol.json')['cap'] != cap or
                    selection['protocol_sha256'] != digest(directory / 'protocol.json')):
                raise ValueError('Validated cap-specific full-speed historical PID snapshots are required.')
            for filename in ('pid-fullspeed.json', 'selection-freeze.json', 'protocol.json'):
                originals[f'pid-{code}-{filename}'] = directory / filename
        for name, source in originals.items():
            shutil.copy2(source, inputs / name)
        for arm, generation in (('smooth', 83), ('fast', 134)):
            metadata = read_model_artifact(inputs / f'{arm}-parent.pkl')['metadata']
            if metadata['generation'] != generation or metadata['robot']['max_speed'] != 1.:
                raise ValueError('Unexpected follow-up parent generation or native cap.')
        shutil.copytree(original / 'prepared/suite', prepared / 'suite')
        for generation in range(50):
            name = f'generation-{generation:04d}'
            source = original / 'development/seed-01/tracks' / name
            destination = prepared / 'batches' / name
            destination.mkdir(parents=True)
            for filename in [*(f'{slot}.json' for slot in range(6)), 'reference.json', 'batch.json']:
                shutil.copy2(source / filename, destination / filename)
        check_batches(prepared); check_validation(prepared)
        inventories, evidence = {}, []
        for arm in ARMS:
            random.seed(EVOLUTION_SEED)
            population = warm_start_population(inputs / f'{arm}-parent.pkl', SETTINGS['population'])
            inventories[arm] = {str(key): genotype_hash(genome) for key, genome in sorted(population.population.items())}
            if arm == 'smooth':
                for gate in gates:
                    saved = read_model_artifact(inputs / f"v1-generation-0000-rank-{gate['candidate_rank']}.pkl")
                    identity = genotype_hash(saved['genome'])
                    if identity != gate['genotype_sha256'] or inventories[arm].get(str(gate['genome_id'])) != identity:
                        raise ValueError('Smooth initial reconstruction differs from saved V1 generation-zero evidence.')
                    evidence.append({'genome_id': gate['genome_id'], 'genotype_sha256': identity})
            path = prepared / ARMS[arm]['initial_population']; path.parent.mkdir(exist_ok=True)
            _save_checkpoint(path, {'population': population, 'random_state': random.getstate()})
        write_json(prepared / 'initial/genotypes.json', {'evolution_seed': EVOLUTION_SEED, 'arms': inventories,
            'smooth_v1_matches': evidence, 'scope': 'Three saved generation-zero genotypes match; V1 did not preserve its full initial population, so complete historical identity is not claimed.'})
        inventory = source_inventory()
        for relative in inventory:
            target = prepared / 'source' / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        plan = {'study': 'neat-quality-followup-v1', 'stage': 'development', 'settings': SETTINGS,
            'fitness_version': VERSION, 'quality_limits': LIMITS, 'evolution_seed': EVOLUTION_SEED,
            'arms': ARMS, 'dependencies': dependencies(), 'source_files': inventory,
            'source_v1_protocol_sha256': prior_hash, 'robot': prior['robot'], 'episode': prior['episode'],
            'input_sources': {name: str(path) for name, path in originals.items()},
            'objective': 'Incomplete fitness unchanged; completed fitness=10*m+9*S/(1+V). Original final acceptance gates unchanged.',
            'reference_budget_per_arm': {'reused': 300, 'new': 0},
            'scope': 'Controlled development contrasts: strict V1 smooth versus revised smooth reward; revised smooth versus native-fast initialization. No interaction, independence, confirmation or fresh-test claim.',
            'pid_scope': 'Historical validated cap-specific baseline provenance only. Proper retuning and subsequent frozen shared comparison are separate; no PID evaluation is launched here.'}
        plan['prepared_files'] = {path.relative_to(prepared).as_posix(): digest(path)
            for path in prepared.rglob('*') if path.is_file() and 'source' not in path.relative_to(prepared).parts}
        write_json(prepared / 'protocol.json', plan)
        (prepared / 'protocol.sha256').write_text(digest(prepared / 'protocol.json') + '\n', encoding='utf-8')
        verify(run)
        print('Frozen follow-up inputs, 50 copied batches and both initial populations; no evolution launched.', flush=True)
