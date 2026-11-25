"""One prospective joint-goal continuation; all inspected data are development data."""
from copy import deepcopy
from pathlib import Path
import pickle
import random
import shutil
from statistics import mean
import subprocess
import sys

import numpy as np

from line_follower.controllers import NEATController, read_model_artifact, save_model
from line_follower.experiments import _save_checkpoint, warm_start_population
from neat_quality_followup.protocol import source_inventory as previous_inventory, check_batches, check_validation
from neat_quality_study.training import genotype_hash
from neat_smooth_study.common import dependencies, digest, lock, read, write_json
from .fitness import VERSION, LIMITS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/neat-joint-study-v1'
EVOLUTION_SEED = 7036297164298145801
SETTINGS = {'generations': 50, 'population': 100, 'courses_per_genome': 6,
            'validation_candidates_per_generation': 3, 'worker_recycle_tasks': 128,
            'mask_cache_bytes': 67108864, 'cap_m_s': 1.}
ARMS = {'joint': {'parent': 'inputs/initialization-template.pkl', 'initial_population': 'initial/joint.pkl.gz'}}
POWERS = {'bias_mutate_power': (.12, .03), 'weight_mutate_power': (.14, .035)}
EXPECTED_PARENT = '064ad6554b353318b29b9d3690873597befd5040836a5104dfab4d69574795bf'


def source_inventory(root=ROOT):
    inventory = previous_inventory(root)
    paths = [root / 'run_neat_joint_study.py', *(root / 'neat_joint_study').glob('*.py')]
    inventory.update({path.relative_to(root).as_posix(): digest(path) for path in paths})
    return dict(sorted(inventory.items()))


def select_parent(previous):
    directory = previous / 'development/fast'
    reference_path = previous / 'prepared/inputs/reference-validation.json'
    reference = read(reference_path)['primary']
    reference_error = mean(row['geometry_lateral_rmse_m'] for row in reference)
    reference_jitter = mean(row['command_jitter'] for row in reference)
    records, sources = [], {str(reference_path): digest(reference_path)}
    for path in sorted((directory / 'validation').glob('generation-*.json')):
        sources[str(path)] = digest(path)
        for candidate in read(path)['candidates']:
            cached_path = directory / 'validation-cache' / f"{candidate['genotype_sha256']}.json"
            cached = read(cached_path)
            sources[str(cached_path)] = digest(cached_path)
            rows, retention = cached['primary'], cached['retention']
            windows = [window for row in rows for window in row['straight_windows']]
            observed = windows and all(window['observed'] for window in windows)
            speed = min((window['speed_m_s'] for window in windows), default=0.) if observed else 0.
            jitter = mean(row['command_jitter'] for row in rows)
            components = {
                'geometric_rmse': max(0., mean(row['geometry_lateral_rmse_m'] for row in rows)/reference_error-1),
                'command_jitter': max(0., jitter/reference_jitter-1),
                'straight_yaw': max(0., max((window['yaw_rms_rad_s'] for window in windows), default=0.)/.15-1) if observed else None,
                'straight_steering': max(0., max((window['steering_rms'] for window in windows), default=0.)/.02-1) if observed else None}
            eligible = (len(rows) == sum(row['completed'] for row in rows) == 60 and
                        len(retention) == sum(row['completed'] for row in retention) == 20 and
                        candidate['paired_time_ratio'] is not None and candidate['paired_time_ratio'] <= 1.05 and
                        observed and speed >= .95)
            records.append({**candidate, 'parent_speed_time_eligible': bool(eligible),
                            'minimum_straight_speed_m_s': speed, 'maximum_quality_excess': max(components.values()) if observed else None,
                            'quality_excess_components': components, 'selection_jitter': jitter,
                            'cache_file': str(cached_path.relative_to(previous))})
    candidates = [row for row in records if row['parent_speed_time_eligible']]
    if not candidates:
        raise ValueError('No full-completion, time/speed-qualified development parent.')
    selected = min(candidates, key=lambda row: (row['maximum_quality_excess'], row['selection_jitter'],
                                               row['generation'], row['candidate_rank']))
    if (selected['generation'], selected['candidate_rank'], selected['genotype_sha256']) != (33, 2, EXPECTED_PARENT):
        raise ValueError('The declared parent-selection audit did not reproduce fast g33 rank 2.')
    return {'selected': selected, 'records': records, 'input_sha256': sources,
            'scope': 'Retrospective development-parent choice; quality violations remain, no qualifying winner is claimed.'}


def initialization_template(original, target):
    artifact = read_model_artifact(original)
    config = deepcopy(artifact['config'])
    for field, (before, after) in POWERS.items():
        if getattr(config.genome_config, field) != before:
            raise ValueError(f'Unexpected original mutation power: {field}')
        setattr(config.genome_config, field, after)
    restored = deepcopy(config)
    for field, (before, _) in POWERS.items():
        setattr(restored.genome_config, field, before)
    if pickle.dumps(restored, protocol=4) != pickle.dumps(artifact['config'], protocol=4):
        raise ValueError('Configuration changes extend beyond the two mutation powers.')
    save_model(target, artifact['genome'], config,
               {**artifact['metadata'], 'arm': 'joint', 'original_parent_sha256': digest(original),
                'initialization_template': True, 'mutation_power_factor': .25})
    template = read_model_artifact(target)
    controllers = [NEATController(item['genome'], item['config']) for item in (artifact, template)]
    rng = random.Random(2026112501)
    observations = [[int(rng.random() < .2) for _ in range(15)] for _ in range(100)]
    observations[0], observations[1] = [0]*15, [1]*15
    actions = [[controller.act(np.array(observation), 1/60).tolist() for observation in observations]
               for controller in controllers]
    identical = np.array_equal(actions[0], actions[1])
    identity = genotype_hash(artifact['genome'])
    if identity != genotype_hash(template['genome']) or not identical:
        raise ValueError('The initialization template changed genotype or deterministic policy actions.')
    return {'genotype_sha256': identity, 'actions_exactly_equal': bool(identical), 'sensor_steps': 100,
            'changed_config_fields': {key: {'before': value[0], 'after': value[1]} for key, value in POWERS.items()},
            'only_declared_config_fields_changed': True, 'observations': observations,
            'original_actions': actions[0], 'template_actions': actions[1]}


def verify(run):
    prepared = Path(run) / 'prepared'
    plan = read(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != digest(prepared / 'protocol.json'):
        raise ValueError('Joint protocol changed after commitment.')
    if (plan['study'] != 'neat-joint-study-v1' or plan['settings'] != SETTINGS or plan['arms'] != ARMS or
            plan['evolution_seed'] != EVOLUTION_SEED or plan['fitness_version'] != VERSION or
            plan['quality_limits'] != LIMITS or plan['mutation_powers'] != {key: list(value) for key, value in POWERS.items()}):
        raise ValueError('Joint scientific settings changed.')
    if plan['dependencies'] != dependencies() or source_inventory() != plan['source_files']:
        raise ValueError('Joint runtime source/dependencies differ from the frozen protocol.')
    if source_inventory(prepared / 'source') != plan['source_files']:
        raise ValueError('Joint frozen execution source changed.')
    for relative, expected in plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError(f'Joint frozen input changed: {relative}')
    return plan


def prepare(run):
    run = Path(run).resolve()
    if (run / 'prepared/protocol.json').exists():
        verify(run)
        print('Existing joint study verified; no preparation regenerated.', flush=True)
        return
    previous = ROOT / 'runs/neat-quality-followup-v1'
    frozen = previous / 'prepared/source'
    command = [sys.executable, str(frozen / 'run_neat_quality_followup.py'), 'verify', '--run', str(previous), '--frozen-source']
    checked = subprocess.run(command, cwd=frozen, capture_output=True, text=True, check=True)
    for arm in ('smooth', 'fast'):
        complete = read(previous / 'development' / arm / 'complete.json')
        if complete['generations'] != 50 or complete['signature']['protocol_sha256'] != digest(previous / 'prepared/protocol.json'):
            raise ValueError('Both parent development arms must be complete and frozen.')
    audit = select_parent(previous)
    selected = audit['selected']
    prior = read(previous / 'prepared/protocol.json')
    with lock(run):
        prepared = run / 'prepared'
        if prepared.exists():
            raise FileExistsError('Preserve partial preparation; use a new run version.')
        inputs = prepared / 'inputs'; inputs.mkdir(parents=True)
        copies = {'original-parent.pkl': previous / 'development/fast' / selected['model_file'],
                  'parent-cache.json': previous / selected['cache_file'],
                  'reference-parent.pkl': previous / 'prepared/inputs/reference-parent.pkl',
                  'reference-validation.json': previous / 'prepared/inputs/reference-validation.json',
                  'parent-protocol.json': previous / 'prepared/protocol.json',
                  'parent-complete.json': previous / 'development/fast/complete.json'}
        for name, path in copies.items():
            shutil.copy2(path, inputs / name)
        write_json(prepared / 'parent-selection-audit.json', audit)
        write_json(prepared / 'parent-verification.json', {'command': command, 'returncode': checked.returncode,
            'stdout': checked.stdout, 'stderr': checked.stderr, 'source': 'Frozen parent entry point, not updated live helper.'})
        identity = initialization_template(inputs / 'original-parent.pkl', inputs / 'initialization-template.pkl')
        write_json(prepared / 'initialization-action-identity.json', identity)
        shutil.copytree(previous / 'prepared/suite', prepared / 'suite')
        shutil.copytree(previous / 'prepared/batches', prepared / 'batches')
        check_batches(prepared); check_validation(prepared)
        old_rng = random.getstate()
        try:
            random.seed(EVOLUTION_SEED)
            population = warm_start_population(inputs / 'initialization-template.pkl', SETTINGS['population'])
            initial = prepared / ARMS['joint']['initial_population']; initial.parent.mkdir()
            _save_checkpoint(initial, {'population': population, 'random_state': random.getstate()})
        finally:
            random.setstate(old_rng)
        inventory = {str(key): genotype_hash(genome) for key, genome in sorted(population.population.items())}
        if inventory.get('1') != identity['genotype_sha256']:
            raise ValueError('Initial key 1 is not the exact selected parent.')
        write_json(prepared / 'initial/genotypes.json', {'evolution_seed': EVOLUTION_SEED, 'genotypes': inventory})
        source_files = source_inventory()
        for relative in source_files:
            target = prepared / 'source' / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        plan = {'study': 'neat-joint-study-v1', 'stage': 'development', 'settings': SETTINGS, 'arms': ARMS,
                'fitness_version': VERSION, 'quality_limits': LIMITS, 'evolution_seed': EVOLUTION_SEED,
                'mutation_powers': {key: list(value) for key, value in POWERS.items()},
                'dependencies': dependencies(), 'source_files': source_files, 'robot': prior['robot'], 'episode': prior['episode'],
                'selected_parent': {key: selected[key] for key in ('generation', 'candidate_rank', 'genotype_sha256')},
                'reference_budget': {'new': 0, 'reused': 300},
                'scope': 'One 50-generation development fine-tune changing parent, joint reward, and two mutation powers together; no isolated causal or independent-test claim.',
                'validation': 'Same consulted 60 primary/20 retention courses; unchanged quality/select gates; top three distinct training genotypes.',
                'source_inputs': {name: str(path) for name, path in copies.items()}}
        plan['prepared_files'] = {path.relative_to(prepared).as_posix(): digest(path) for path in prepared.rglob('*')
                                  if path.is_file() and 'source' not in path.relative_to(prepared).parts}
        write_json(prepared / 'protocol.json', plan)
        (prepared / 'protocol.sha256').write_text(digest(prepared / 'protocol.json') + '\n', encoding='utf-8')
        verify(run)
        print('Prepared one joint arm: 50 generations, 100 initial genomes, 300 copied courses; no evolution launched.', flush=True)
