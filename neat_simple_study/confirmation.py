"""Prospective five-seed protocol; preparation requires a completed pilot decision."""
import base64
import gzip
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import pickle
import shutil

from line_follower.controllers import save_model
from line_follower.experiments import _save_checkpoint
from line_follower.suites import derived_seed
from line_follower.tracks import TrackSpec
from neat_quality_study.training import genotype_hash
from neat_smooth_study.common import dependencies, digest, lock, read, write_json
from neat_smooth_study.courses import make_suite, protected_geometry
from .curriculum import GENERATIONS, SPEED_SCHEDULE, BATCH_GROUPS, fresh_population, generation_batch
from .fitness import VERSION
from run_neat_simple_study import ARMS, source_inventory as pilot_inventory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/neat-simple-confirm-v1'
STUDY_SEED = 2026112503
PILOT_STUDY = 'neat-simple-pilot-50-v1'
SEEDS = tuple(range(1, 6))
SETTINGS = {'generations': GENERATIONS, 'population': 100, 'courses_per_genome': 6,
            'validation_candidates': 1, 'worker_recycle_tasks': 128,
            'mask_cache_bytes': 64 * 1024**2}
VALIDATION = {'validation': {'mixed': 30, 'speed': 20, 'corner': 10},
              'retention': {'mixed': 15, 'speed': 10, 'corner': 5},
              'training': {'speed': 20}}
TEST_RULE = {'namespace': 'simple-confirm-heldout-2026112503',
             'counts': {'mixed': 90, 'speed': 60, 'corner': 30}, 'caps': [1., .5]}


def source_inventory(root=ROOT):
    files = pilot_inventory(root)
    files['run_neat_simple_confirm.py'] = digest(root / 'run_neat_simple_confirm.py')
    return dict(sorted(files.items()))


def neat_sources():
    distribution = importlib.metadata.distribution('neat-python')
    if distribution.version != '0.92':
        raise ValueError('This protocol requires stock NEAT-Python 0.92.')
    files = {}
    for path in distribution.files:
        name = str(path).replace('\\', '/')
        if not name.startswith('neat/') or path.suffix != '.py':
            continue
        data = distribution.locate_file(path).read_bytes()
        fingerprint = hashlib.sha256(data).digest()
        expected = path.hash
        if (expected is None or expected.mode != 'sha256'
                or base64.urlsafe_b64encode(fingerprint).decode().rstrip('=') != expected.value):
            raise ValueError(f'Installed NEAT source differs from its installation RECORD: {name}')
        files[name] = fingerprint.hex()
    if not files:
        raise ValueError('Cannot verify stock NEAT runtime source.')
    return files


def verify(run):
    prepared = Path(run) / 'prepared'
    plan = read(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != digest(prepared / 'protocol.json'):
        raise ValueError('Confirmation protocol changed after commitment.')
    if (plan['study'] != 'neat-simple-confirm-v1' or plan['settings'] != SETTINGS
            or plan['fitness_version'] != VERSION or plan['test_rule'] != TEST_RULE
            or plan['speed_schedule'] != [list(stage) for stage in SPEED_SCHEDULE]
            or sorted(map(int, plan['seeds'])) != list(SEEDS)):
        raise ValueError('Confirmation budget, objective, curriculum or seeds changed.')
    if (plan['dependencies'] != dependencies() or plan['neat_sources'] != neat_sources()
            or plan['python_version'] != platform.python_version()):
        raise ValueError('Dependencies or stock NEAT source differ from the freeze.')
    if source_inventory() != plan['source_files'] or source_inventory(prepared / 'source') != plan['source_files']:
        raise ValueError('Confirmation execution source changed.')
    for relative, expected in plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError(f'Frozen confirmation input changed: {relative}')
    decision = read(prepared / 'inputs/development-decision.json')
    _pilot_paths(decision)
    if plan['smoothness_weight'] != decision['smoothness_weight']:
        raise ValueError('Confirmation coefficient differs from the declared development choice.')
    return plan


def _pilot_paths(decision):
    if decision.get('pilot_study') != PILOT_STUDY:
        raise ValueError('Explicitly identify the approved neat-simple-pilot-50-v1 study.')
    return ROOT / 'runs' / PILOT_STUDY, ROOT / 'results' / PILOT_STUDY


def checked_decision(path):
    """Only a transparent completed-development choice can start confirmation."""
    decision = read(path)
    from neat_quality_followup.training import training_signature
    pilot, output = _pilot_paths(decision)
    plan = read(pilot / 'prepared/protocol.json')
    fingerprint = digest(pilot / 'prepared/protocol.json')
    if (decision['pilot_protocol_sha256'] != fingerprint
            or (pilot / 'prepared/protocol.sha256').read_text().strip() != fingerprint):
        raise ValueError('Decision belongs to a different pilot.')
    if (plan['study'] != PILOT_STUDY or plan['fitness_version'] != VERSION
            or plan['settings']['generations'] != 50 or set(plan['arms']) != set(ARMS)
            or any(plan['arms'][arm]['smoothness_weight'] != weight for arm, weight in ARMS.items())):
        raise ValueError('The matching three-arm, 50-total-generation compact study is required.')
    for relative, expected in plan['source_files'].items():
        if digest(pilot / 'prepared/source' / relative) != expected:
            raise ValueError('Frozen development source changed.')
    for relative, expected in plan['prepared_files'].items():
        if digest(pilot / 'prepared' / relative) != expected:
            raise ValueError(f'Frozen development input changed: {relative}')
    terminal_files = {'protocol': pilot / 'prepared/protocol.json'}
    for arm in ARMS:
        directory = pilot / 'development' / arm
        complete_path, checkpoint_path = directory / 'complete.json', directory / 'checkpoints/latest.pkl.gz'
        complete = read(complete_path)
        with gzip.open(checkpoint_path, 'rb') as stream:
            checkpoint = pickle.load(stream)
        signature = training_signature(pilot, plan, arm, VERSION, condition='simple-reward-development')
        if (complete['generations'] != 50 or checkpoint['population'].generation != 50
                or complete['signature'] != signature or checkpoint['signature'] != signature
                or complete['training_episodes'] != checkpoint['training_episodes']):
            raise ValueError('Complete all three matching 50-total-generation arms first.')
        terminal_files[f'{arm}/complete'] = complete_path
        terminal_files[f'{arm}/checkpoint'] = checkpoint_path
    weight = decision['smoothness_weight']
    if type(weight) not in (int, float) or weight not in (0., 1., 10.) or not decision.get('rationale'):
        raise ValueError('Declare a tested coefficient and its development rationale.')
    report = read(output / 'report.json')
    if (report['complete'] is not True or report['protocol_sha256'] != fingerprint
            or report['snapshot_sha256'] != decision['pilot_snapshot_sha256']
            or digest(output / 'report.json') != decision['pilot_report_sha256']
            or set(report['arms']) != set(ARMS)
            or any(row['complete'] is not True or row['committed_generations'] != 50 for row in report['arms'].values())):
        raise ValueError('Decision must identify the complete captured pilot analysis.')
    address = decision['pilot_snapshot_sha256']
    if not isinstance(address, str) or len(address) != 64 or any(character not in '0123456789abcdef' for character in address):
        raise ValueError('Pilot snapshot must be identified by its SHA-256 address.')
    snapshot = output / 'snapshots' / address
    if digest(snapshot / 'manifest.json') != snapshot.name:
        raise ValueError('Pilot analysis snapshot identity differs.')
    inputs = read(snapshot / 'manifest.json')['inputs']
    for role, entry in inputs.items():
        if Path(entry['file']).name != entry['file'] or digest(snapshot / entry['file']) != entry['sha256']:
            raise ValueError(f'Pilot analysis snapshot payload changed: {role}')
    if not (set(terminal_files) | {'analysis-source'}) <= set(inputs):
        raise ValueError('Pilot snapshot lacks terminal checkpoints or archived analysis source.')
    for role, source in terminal_files.items():
        if inputs[role]['sha256'] != digest(source):
            raise ValueError('Captured pilot evidence differs from its terminal study state.')
    if digest(snapshot / 'analysis.py') != inputs['analysis-source']['sha256']:
        raise ValueError('Archived pilot analysis source changed.')
    return decision, plan


def prepare(run, decision_path):
    run = Path(run).resolve()
    if (run / 'prepared/protocol.json').exists():
        verify(run)
        return
    decision, pilot = checked_decision(decision_path)
    pilot_run, pilot_output = _pilot_paths(decision)
    with lock(run):
        prepared = run / 'prepared'
        if prepared.exists():
            raise FileExistsError('Preserve the partial preparation; do not overwrite it.')
        inputs = prepared / 'inputs'
        inputs.mkdir(parents=True)
        shutil.copy2(decision_path, inputs / 'development-decision.json')
        shutil.copy2(pilot_output / 'report.json', inputs / 'pilot-report.json')
        shutil.copy2(pilot_output / 'snapshots' / decision['pilot_snapshot_sha256'] / 'manifest.json',
                     inputs / 'pilot-analysis-manifest.json')
        old_inputs = pilot_run / 'prepared/inputs'
        for name in ('accepted-prototype.pkl', 'pid-100.json', 'pid-050.json', 'controller-manifest.json'):
            shutil.copy2(old_inputs / name, inputs / name)
        previous = read(inputs / 'controller-manifest.json')['controllers']
        for name, old_name in (('accepted-prototype.pkl', 'neat-prototype.pkl'), ('pid-100.json', 'pid-100.json'), ('pid-050.json', 'pid-050.json')):
            if digest(inputs / name) != previous[old_name]['sha256']:
                raise ValueError('Frozen development controls differ.')
        protected = protected_geometry(ROOT)
        write_json(inputs / 'protected-hashes.json', sorted(protected))
        used = set(protected)
        suite = make_suite(prepared / 'suite', 'simple-confirm-development-2026112503', VALIDATION, used)
        bank_entries = [entry for entry in suite['splits']['training'] if entry['kind'] == 'isolated']
        bank_entries += [entry for entry in suite['splits']['training'] if entry['kind'] == 'post_turn']
        bank = [TrackSpec.load(prepared / 'suite' / entry['path']) for entry in bank_entries]
        bank_hashes = {track.geometry_hash for track in bank}
        seeds, initial_identities = {}, set()
        (prepared / 'initial').mkdir()
        for seed in SEEDS:
            evolution_seed = derived_seed(STUDY_SEED, 'fresh-evolution', seed)
            track_seed = derived_seed(STUDY_SEED, 'fresh-tracks', seed)
            population, rng = fresh_population(evolution_seed)
            identities = {genotype_hash(genome) for genome in population.population.values()}
            if len(identities) != 100 or identities & initial_identities:
                raise ValueError('All 500 initial random genomes must have distinct genotypes.')
            initial_identities.update(identities)
            initial = f'initial/seed-{seed:02d}.pkl.gz'
            template = f'initial/seed-{seed:02d}-template.pkl'
            _save_checkpoint(prepared / initial, {'population': population, 'random_state': rng})
            save_model(prepared / template, next(iter(population.population.values())), population.config,
                       {'condition': 'random-config-template-only', 'seed': seed, 'trained': False})
            seeds[str(seed)] = {'initial_population': initial, 'controller_template': template,
                                'evolution_seed': evolution_seed, 'track_seed': track_seed}
            for generation in range(GENERATIONS):
                tracks = generation_batch(track_seed, generation, bank, used - bank_hashes)
                entries = []
                for slot, (track, group) in enumerate(zip(tracks, BATCH_GROUPS)):
                    relative = f'batches/seed-{seed:02d}/generation-{generation:04d}/{slot}.json'
                    track.save(prepared / relative)
                    entries.append({'path': relative, 'geometry_hash': track.geometry_hash, 'group': group})
                    used.add(track.geometry_hash)
                write_json(prepared / f'batches/seed-{seed:02d}/generation-{generation:04d}/batch.json',
                           {'generation': generation, 'entries': entries,
                            'cap_m_s': next(cap for start, cap in reversed(SPEED_SCHEDULE) if generation >= start)})
            print(f'Prepared seed {seed}: fresh 100-genome population and {GENERATIONS} deterministic six-course batches.', flush=True)
        write_json(inputs / 'all-development-hashes.json', sorted(used))
        source_files = source_inventory()
        for relative in source_files:
            target = prepared / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        files = {path.relative_to(prepared).as_posix(): digest(path) for path in sorted(prepared.rglob('*'))
                 if path.is_file() and 'source' not in path.relative_to(prepared).parts}
        protocol = {'study': 'neat-simple-confirm-v1', 'fitness_version': VERSION,
                    'smoothness_weight': decision['smoothness_weight'], 'settings': SETTINGS,
                    'speed_schedule': [list(stage) for stage in SPEED_SCHEDULE], 'seeds': seeds,
                    'robot': pilot['robot'], 'episode': pilot['episode'],
                    'dependencies': dependencies(), 'neat_sources': neat_sources(),
                    'python_version': platform.python_version(), 'preparation_platform': platform.platform(),
                    'source_files': source_files, 'prepared_files': files, 'test_rule': TEST_RULE,
                    'scope': 'Five independent from-scratch populations. Single objective and fixed NEAT settings throughout; no intermediate winner-based restart.',
                    'selection': 'One training champion per generation. Highest primary validation compact fitness during generations 250-399; tie earliest. Retention is measured separately, not a hidden quality gate.',
                    'validation': '60 primary at current training cap, 30 retention at 0.5 m/s, all shared fixed geometries. Final test mixture matches the 3:2:1 course-group ratio.',
                    'measurement': 'Training skips independent geometric RMSE; objective uses only progress/completion/time/cap/command jitter. Detailed validation and assessment measure geometry/yaw/straight speeds. Physics and policy are unchanged.',
                    'budget': {'nominal_training_episodes_per_seed': GENERATIONS * 100 * 6,
                               'maximum_validation_slots_per_seed': GENERATIONS * 90,
                               'seeds': 5, 'automatic_extensions': False},
                    'limits': 'Budget is not claimed minimal or optimal. No hardware claim; controller comparison does not isolate curriculum/reward effects. All seeds and failed outcomes must be reported.'}
        write_json(prepared / 'protocol.json', protocol)
        (prepared / 'protocol.sha256').write_text(digest(prepared / 'protocol.json') + '\n')
    verify(run)
    print('Five-seed confirmation prepared; evolution has not started.')


def preflight(run, workers=6):
    """Verify that skipping unused geometric measurements preserves control."""
    import gzip
    import math
    import pickle
    from neat_smooth_study.engine import evaluate_tasks
    from .fitness import score_batch
    if workers not in (6, 7, 8):
        raise ValueError('Use six to eight workers.')
    run = Path(run).resolve()
    plan = verify(run)
    prepared = run / 'prepared'
    seed = plan['seeds']['1']
    with gzip.open(prepared / seed['initial_population'], 'rb') as stream:
        population = pickle.load(stream)['population']
    genome = next(iter(population.population.values()))
    batch = read(prepared / 'batches/seed-01/generation-0000/batch.json')
    entries = batch['entries']
    tasks = []
    for cap in (.5, 1.):
        for policy in ({'genome': genome}, {'model': str(prepared / 'inputs/accepted-prototype.pkl')}):
            for detailed in (False, True):
                for entry in entries:
                    tasks.append({**policy, 'paths': [str(prepared / entry['path'])],
                                  'groups': {entry['geometry_hash']: entry['group']},
                                  'cap': cap, 'detailed': detailed, 'branch': 'old'})
    observed = [rows[0] for rows in evaluate_tasks(run, tasks, workers, prepared / seed['controller_template'])]
    relevant = ('progress', 'elapsed_s', 'completion_time_s', 'control_steps', 'physics_steps',
                'command_jitter', 'average_forward_speed_m_s', 'peak_forward_speed_m_s',
                'mean_action_L', 'mean_action_R')
    checks = []
    for offset in range(0, len(observed), 12):
        fast, detailed = observed[offset:offset+6], observed[offset+6:offset+12]
        for a, b in zip(fast, detailed):
            if any(a[field] != b[field] for field in ('track_hash', 'completed', 'termination', 'error')):
                raise ValueError('Detailed instrumentation changed controller/physics outcomes.')
            for field in relevant:
                if a[field] is None or b[field] is None:
                    equal = a[field] is b[field]
                else:
                    equal = math.isclose(a[field], b[field], rel_tol=1e-12, abs_tol=1e-12)
                if not equal:
                    raise ValueError(f'Detailed instrumentation changed {field}.')
        a = score_batch(fast, smoothness_weight=plan['smoothness_weight'])
        b = score_batch(detailed, smoothness_weight=plan['smoothness_weight'])
        if a != b:
            raise ValueError('Detailed instrumentation changed compact fitness.')
        checks.append({'cap_m_s': fast[0]['cap_m_s'], 'score': a, 'all_relevant_metrics_equal': True})
    write_json(run / 'preflight/instrumentation-invariance.json', {
        'protocol_sha256': digest(prepared / 'protocol.json'), 'workers': workers,
        'new_episodes': len(observed), 'all_match': True, 'checks': checks, 'episodes': observed})
    print('All 48 real detailed/non-detailed episodes matched on fitness-relevant behavior.')
