"""Matched continuation to 50 total generations; preparation never evolves."""
from copy import deepcopy
import gzip
from pathlib import Path
import pickle
import shutil

from line_follower.experiments import _save_checkpoint
from line_follower.tracks import TrackSpec
from neat_smooth_study.common import dependencies, digest, lock, read, write_json
from run_neat_simple_study import ARMS, SETTINGS as PILOT_SETTINGS, source_inventory as pilot_inventory
from .fitness import VERSION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/neat-simple-pilot-50-v1'
PARENT_RUN = ROOT / 'runs/neat-simple-pilot-v1'
BATCH_ANCESTOR_RUN = ROOT / 'runs/neat-quality-followup-v1'
INHERITED_GENERATIONS = 20
SETTINGS = {**PILOT_SETTINGS, 'generations': 50}
CONDITION = 'simple-reward-development'


def source_inventory(root=ROOT):
    inventory = pilot_inventory(root)
    for name in ('run_neat_simple_continue.py', 'run_neat_simple_confirm.py'):
        if (root / name).exists():
            inventory[name] = digest(root / name)
    return dict(sorted(inventory.items()))


def _read_checkpoint(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def _copy_history_file(source, target):
    # CopyFile2 can reject long Windows paths; only exact bytes are required.
    target.write_bytes(source.read_bytes())


def _file_hashes(directory):
    return {path.relative_to(directory).as_posix(): digest(path)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def _checked_parent(parent):
    prepared = parent / 'prepared'
    plan = read(prepared / 'protocol.json')
    fingerprint = digest(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != fingerprint:
        raise ValueError('Parent pilot protocol changed.')
    if (plan['study'] != 'neat-simple-pilot-v1' or plan['fitness_version'] != VERSION
            or plan['settings'] != {**SETTINGS, 'generations': INHERITED_GENERATIONS}
            or plan['dependencies'] != dependencies() or set(plan['arms']) != set(ARMS)):
        raise ValueError('The matched completed parent pilot is required.')
    for relative, expected in plan['source_files'].items():
        if digest(prepared / 'source' / relative) != expected:
            raise ValueError('Parent frozen source changed.')
    for relative, expected in plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError('Parent prepared input changed.')
    checkpoints = {}
    for arm, weight in ARMS.items():
        complete = read(parent / 'development' / arm / 'complete.json')
        state = _read_checkpoint(parent / 'development' / arm / 'checkpoints/latest.pkl.gz')
        signature = state['signature']
        if (complete['generations'] != INHERITED_GENERATIONS
                or state['population'].generation != INHERITED_GENERATIONS
                or complete['signature'] != signature or signature['protocol_sha256'] != fingerprint
                or signature['arm'] != arm or signature['fitness_version'] != VERSION
                or signature['settings'] != plan['settings']
                or signature['condition'] != CONDITION
                or signature['evolution_seed'] != plan['evolution_seed']
                or signature['initial_population_sha256'] != digest(prepared / 'initial/common.pkl.gz')
                or signature['parent_sha256'] != digest(prepared / 'inputs/fast-parent.pkl')
                or signature['reference_validation_sha256'] != digest(prepared / 'inputs/reference-validation.json')
                or complete['training_episodes'] != state['training_episodes']
                or plan['arms'][arm]['smoothness_weight'] != weight):
            raise ValueError(f'Parent {arm} is not the matching completed boundary checkpoint.')
        checkpoints[arm] = state
    return plan, checkpoints


def _history_files(directory):
    return [path for path in sorted(directory.rglob('*')) if path.is_file()
            and 'checkpoints' not in path.relative_to(directory).parts
            and path.name != 'session.lock' and not path.name.endswith('.tmp')]


def verify(run):
    prepared = Path(run) / 'prepared'
    plan = read(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != digest(prepared / 'protocol.json'):
        raise ValueError('Continuation protocol changed after commitment.')
    arms = {arm: {'smoothness_weight': weight, 'parent': 'inputs/fast-parent.pkl',
                  'initial_population': f'initial/{arm}.pkl.gz'} for arm, weight in ARMS.items()}
    if (plan['study'] != 'neat-simple-pilot-50-v1' or plan['fitness_version'] != VERSION
            or plan['settings'] != SETTINGS or plan['arms'] != arms
            or plan['inherited_generations'] != INHERITED_GENERATIONS
            or plan['dependencies'] != dependencies()):
        raise ValueError('Continuation arms, 50-total budget or objective changed.')
    parent = Path(plan['parent_run'])
    parent_plan = read(prepared / 'parent-protocol.json')
    states = {arm: _read_checkpoint(prepared / f'initial/{arm}.pkl.gz') for arm in ARMS}
    if (digest(prepared / 'parent-protocol.json') != plan['parent_protocol_sha256']
            or plan['evolution_seed'] != parent_plan['evolution_seed']
            or plan['inherited_validation_signatures'] != {arm: state['signature'] for arm, state in states.items()}):
        raise ValueError('Continuation ancestry or inherited cache signatures changed.')
    for relative, expected in parent_plan['source_files'].items():
        if digest(prepared / 'parent-source' / relative) != expected:
            raise ValueError('Sealed parent source differs.')
    for relative, expected in parent_plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError('Sealed original pilot input differs.')
    if source_inventory() != plan['source_files'] or source_inventory(prepared / 'source') != plan['source_files']:
        raise ValueError('Continuation execution source differs from the freeze.')
    for relative, expected in plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError(f'Frozen continuation input changed: {relative}')
    for relative, expected in plan['historical_file_hashes'].items():
        historical = prepared / 'history' / Path(*Path(relative).parts[1:])
        if digest(historical) != expected:
            raise ValueError(f'Sealed historical output changed: {relative}')
        if parent.exists() and digest(parent / relative) != expected:
            raise ValueError(f'Historical parent output changed: {relative}')
    for arm, expected in plan['parent_checkpoint_sha256'].items():
        if digest(prepared / f'initial/{arm}.pkl.gz') != expected:
            raise ValueError('Sealed historical checkpoint changed.')
        if parent.exists() and digest(parent / 'development' / arm / 'checkpoints/latest.pkl.gz') != expected:
            raise ValueError('Historical parent checkpoint changed.')
    return plan


def _install_boundary_checkpoints(run, plan):
    from neat_quality_followup.training import training_signature
    for arm in ARMS:
        directory = run / 'development' / arm
        checkpoint = directory / 'checkpoints/latest.pkl.gz'
        if checkpoint.exists():
            continue  # Preparation must never overwrite a child that has advanced.
        history = run / 'prepared/history' / arm
        directory.mkdir(parents=True, exist_ok=True)
        for source in history.rglob('*'):
            relative = source.relative_to(history)
            if source.is_file() and relative.as_posix() not in ('run.json', 'complete.json'):
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_history_file(source, target)
        state = _read_checkpoint(run / 'prepared' / plan['arms'][arm]['initial_population'])
        original = pickle.dumps({key: value for key, value in state.items() if key != 'signature'}, protocol=4)
        updated = {**state, 'signature': training_signature(run, plan, arm, VERSION, condition=CONDITION)}
        checkpoint.parent.mkdir(exist_ok=True)
        _save_checkpoint(checkpoint, updated)
        restored = _read_checkpoint(checkpoint)
        if pickle.dumps({key: value for key, value in restored.items() if key != 'signature'}, protocol=4) != original:
            raise ValueError('Boundary installation changed population, RNG or inherited state.')
        write_json(directory / 'run.json', {**updated['signature'],
            'inherited_generations': INHERITED_GENERATIONS, 'new_generations_completed': 0,
            'parent_protocol_sha256': plan['parent_protocol_sha256'],
            'scope': 'Prepared continuation boundary only; inherited episodes are not new training.'})


def prepare(run):
    run = Path(run).resolve()
    with lock(run):
        if (run / 'prepared/protocol.json').exists():
            plan = verify(run)
            _install_boundary_checkpoints(run, plan)
            return
        parent = PARENT_RUN.resolve()
        parent_plan, states = _checked_parent(parent)
        prepared = run / 'prepared'
        if prepared.exists():
            raise FileExistsError('Preserve partial preparation; choose another run directory.')
        prepared.mkdir()
        shutil.copy2(parent / 'prepared/protocol.json', prepared / 'parent-protocol.json')
        shutil.copytree(parent / 'prepared/source', prepared / 'parent-source')
        for folder in ('inputs', 'suite', 'initial'):
            shutil.copytree(parent / 'prepared' / folder, prepared / folder)
        common = _read_checkpoint(prepared / 'initial/common.pkl.gz')['population']
        if common.generation != 0 or len(common.population) != SETTINGS['population']:
            raise ValueError('The original common initial population changed.')
        preflight_path = parent / 'preflight/physics-replay.json'
        replay = read(preflight_path)
        if not replay['all_match'] or replay['protocol_sha256'] != digest(parent / 'prepared/protocol.json'):
            raise ValueError('The parent physics preflight is not the frozen matched replay.')
        shutil.copy2(preflight_path, prepared / 'inputs/inherited-preflight.json')
        historical, parent_checkpoints = {}, {}
        for arm in ARMS:
            source = parent / 'development' / arm
            checkpoint = source / 'checkpoints/latest.pkl.gz'
            parent_checkpoints[arm] = digest(checkpoint)
            shutil.copy2(checkpoint, prepared / 'initial' / f'{arm}.pkl.gz')
            for path in _history_files(source):
                historical[path.relative_to(parent).as_posix()] = digest(path)
                target = prepared / 'history' / arm / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_history_file(path, target)
        suite = read(prepared / 'suite/suite.json')['splits']
        held_out = {entry['geometry_hash'] for split in ('validation', 'retention') for entry in suite[split]}
        batch_plan = read(BATCH_ANCESTOR_RUN / 'prepared/protocol.json')
        for generation in range(SETTINGS['generations']):
            relative = Path('batches') / f'generation-{generation:04d}'
            source = BATCH_ANCESTOR_RUN / 'prepared' / relative
            for name, fingerprint in _file_hashes(source).items():
                if batch_plan['prepared_files'].get((relative / name).as_posix()) != fingerprint:
                    raise ValueError('A continuation batch differs from its frozen original source.')
            if generation < INHERITED_GENERATIONS and _file_hashes(source) != _file_hashes(parent / 'prepared' / relative):
                raise ValueError('The first inherited course batches differ from the completed pilot.')
            shutil.copytree(source, prepared / relative)
            hashes = read(prepared / relative / 'batch.json')['hashes']
            actual = [TrackSpec.load(prepared / relative / f'{slot}.json').geometry_hash for slot in range(6)]
            if hashes != actual or len(set(hashes)) != 6 or set(hashes) & held_out:
                raise ValueError('Continuation batch geometry/order or validation separation differs.')
        sources = source_inventory()
        for relative in sources:
            target = prepared / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        files = {path.relative_to(prepared).as_posix(): digest(path) for path in sorted(prepared.rglob('*'))
                 if path.is_file() and 'source' not in path.relative_to(prepared).parts}
        plan = {**deepcopy(parent_plan), 'study': 'neat-simple-pilot-50-v1', 'settings': SETTINGS,
            'arms': {arm: {'smoothness_weight': weight, 'parent': 'inputs/fast-parent.pkl',
                            'initial_population': f'initial/{arm}.pkl.gz'} for arm, weight in ARMS.items()},
            'parent_run': str(parent), 'parent_protocol_sha256': digest(parent / 'prepared/protocol.json'),
            'inherited_generations': INHERITED_GENERATIONS,
            'inherited_validation_signatures': {arm: state['signature'] for arm, state in states.items()},
            'historical_file_hashes': historical, 'parent_checkpoint_sha256': parent_checkpoints,
            'source_files': sources, 'prepared_files': files,
            'batch_source_protocol_sha256': digest(BATCH_ANCESTOR_RUN / 'prepared/protocol.json'),
            'controls': 'All arms retain their exact generation-20 population/RNG and use the same 50 preserved six-course batches; no restart or operator change.',
            'scope': 'Matched warm-start development continuation: 50 total generations per arm, including 20 inherited. No independent-seed or final-test claim.',
            'budget': {'cumulative_generations_per_arm': SETTINGS['generations'],
                       'inherited_generations_per_arm': INHERITED_GENERATIONS,
                       'new_generations_per_arm': SETTINGS['generations'] - INHERITED_GENERATIONS,
                       'nominal_new_training_per_arm': (SETTINGS['generations'] - INHERITED_GENERATIONS) * SETTINGS['population'] * 6,
                       'nominal_cumulative_training_per_arm': SETTINGS['generations'] * SETTINGS['population'] * 6,
                       'nominal_training_per_arm': SETTINGS['generations'] * SETTINGS['population'] * 6,
                       'maximum_validation_slots_per_arm': SETTINGS['generations'] * SETTINGS['validation_candidates_per_generation'] * 80,
                       'maximum_new_validation_slots_per_arm': (SETTINGS['generations'] - INHERITED_GENERATIONS) * SETTINGS['validation_candidates_per_generation'] * 80,
                       'arms': len(ARMS), 'automatic_extensions': False}}
        write_json(prepared / 'protocol.json', plan)
        (prepared / 'protocol.sha256').write_text(digest(prepared / 'protocol.json') + '\n')
        verify(run)
        _install_boundary_checkpoints(run, plan)
    print('Prepared exact inherited generation-20 boundaries; no new evolution. Hard stop: 50 total per arm.')
