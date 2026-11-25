"""Frozen inputs and source for a separate quality-speed development study."""
from pathlib import Path
import json
import shutil

from line_follower.suites import derived_seed
from neat_smooth_study.common import dependencies, digest, read, write_json, lock
from neat_smooth_study.courses import protected_geometry
from .fitness import VERSION, LIMITS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/neat-quality-speed-v1'
SETTINGS = {
    'generations': 50, 'population': 100, 'courses_per_genome': 6,
    'validation_candidates_per_generation': 3, 'cap_m_s': 1.,
    'mask_cache_bytes': 64 * 1024**2, 'worker_recycle_tasks': 128,
    'study_seed': 2026112401,
}


def source_inventory(root=ROOT):
    paths = [root / 'run_neat_quality_study.py', root / 'requirements-lock.txt']
    for folder in ('line_follower', 'neat_smooth_study', 'neat_quality_study', 'configs'):
        paths.extend(p for p in (root / folder).rglob('*')
                     if p.is_file() and p.suffix in {'.py', '.json', '.txt', '.xml'})
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def evolution_seed(seed=1):
    return derived_seed(SETTINGS['study_seed'], 'quality-speed-development', seed)


def verify(run):
    prepared = Path(run) / 'prepared'
    plan = read(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != digest(prepared / 'protocol.json'):
        raise ValueError('Quality-speed protocol changed after commitment.')
    if plan['settings'] != SETTINGS or plan['fitness_version'] != VERSION or plan['quality_limits'] != LIMITS:
        raise ValueError('Fitness/settings changed; use a new experiment version.')
    if plan['dependencies'] != dependencies() or source_inventory() != plan['source_files']:
        raise ValueError('Runtime source or dependencies differ from the frozen protocol.')
    for path, expected in plan['prepared_files'].items():
        if digest(prepared / path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    if source_inventory(prepared / 'source') != plan['source_files']:
        raise ValueError('Frozen execution source changed.')
    return plan


def prepare(run):
    run = Path(run).resolve()
    if (run / 'prepared/protocol.json').exists():
        verify(run)
        print('Existing quality-speed protocol verified without rewriting it.', flush=True)
        return
    old = ROOT / 'runs/neat-smooth-study-v1'
    prior = read(old / 'prepared/protocol.json')
    if dependencies()['neat-python'] != '0.92' or dependencies()['mujoco'] != '4.0.0':
        raise ValueError('Use the recorded stock NEAT-Python 0.92 / MuJoCo 4.0.0 environment.')
    if dependencies() != prior['dependencies']:
        raise ValueError('The controlled predecessor dependency set changed.')
    for relative, expected in prior['source_files'].items():
        if relative.startswith('line_follower/') and digest(ROOT / relative) != expected:
            raise ValueError(f'Controller/physics source differs from the preserved baseline: {relative}')
    with lock(run):
        prepared = run / 'prepared'
        if prepared.exists():
            raise FileExistsError('Partial preparation exists; preserve it and choose a new run directory.')
        inputs = prepared / 'inputs'
        inputs.mkdir(parents=True)
        originals = {
            'warm-start-g0083.pkl': old / 'pilot/seed-01/smooth/champions/generation-0083.pkl',
            'reference-g0134.pkl': old / 'prepared/inputs/seed-01.pkl',
            'reference-validation.json': old / 'pilot/references/seed-01.json',
            'pid-fullspeed-100.json': old / 'pid/cap-100/pid-fullspeed.json',
            'pid-selection.json': old / 'pid/cap-100/selection-freeze.json',
            'geometry-freeze.json': old / 'prepared/inputs/geometry-freeze.json',
        }
        selection = read(originals['pid-selection.json'])
        if not selection['eligible'] or selection['sha256'] != digest(originals['pid-fullspeed-100.json']):
            raise ValueError('A validated full-speed PID must be frozen first.')
        from line_follower.controllers import read_model_artifact
        parent = read_model_artifact(originals['warm-start-g0083.pkl'])
        if parent['metadata']['generation'] != 83 or parent['metadata']['robot']['max_speed'] != 1.:
            raise ValueError('Unexpected development warm start.')
        cached_reference = read(originals['reference-validation.json'])
        if cached_reference['parent_sha256'] != digest(originals['reference-g0134.pkl']):
            raise ValueError('Cached parent validation has different ancestry.')
        for name, source in originals.items():
            shutil.copy2(source, inputs / name)
        # Deliberately reuse development/selection data. Fresh final test is not prepared here.
        shutil.copytree(old / 'prepared/pilot/suite', prepared / 'suite')
        calibration = ROOT / 'results/neat-quality-speed-v1/fitness-calibration.json'
        if not calibration.exists():
            raise ValueError('Record the archived-data fitness calibration before training preparation.')
        shutil.copy2(calibration, prepared / 'fitness-calibration.json')
        inventory = source_inventory()
        for relative in inventory:
            target = prepared / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        plan = {
            'study': 'neat-quality-speed-v1', 'stage': 'development',
            'settings': SETTINGS, 'fitness_version': VERSION, 'quality_limits': LIMITS,
            'dependencies': dependencies(), 'source_files': inventory,
            'robot': prior['robot'], 'episode': prior['episode'],
            'evolution_seed': evolution_seed(),
            'input_sources': {name: str(path) for name, path in originals.items()},
            'protected_geometry': sorted(protected_geometry(ROOT)),
            'quality_reference': 'Frozen fast g134 on each identical training batch; mean independent geometric RMSE and jitter, with disclosed 5 mm / 0.0005 training floors.',
            'objective': 'Completion count first; among full-completion batches, quality feasibility then speed. Graded violations retain breeding direction. No inherited continuous jitter/legacy bonus enters breeding fitness.',
            'speed_utility': '.75*min(minimum_required_straight_speed/.95,1) + .25*mean(min(completed_route_speed/1mps,1)); missing window speed=0.',
            'validation': 'Top 3 distinct training genotypes each generation; 60 primary +20 retention courses; exact repeat genotypes reuse deterministic validation cache. Original g134 acceptance and 5% paired-time allowance unchanged.',
            'selection_scope': 'Previously consulted development validation, not final-test evidence. This pilot tests a fitness/aggregation revision and a g83 warm start together.',
            'confirmation': 'Independent consistent five-lineage protocol must be frozen after development; this one seed does not establish repeatability.',
            'comparison': 'Both controller families retain the same sensor interface, physics, cap and recovery semantics. PID base=1; no neural output filter, imitation target or controller access to route geometry.',
            'data_policy': 'Preserve all attempts, rejected candidates, raw episodes, source/inputs, mutation RNG, checkpoint and worker history. Committed episode totals exclude unknown partial work from a crash; interruptions are recorded.',
        }
        plan['prepared_files'] = {
            p.relative_to(prepared).as_posix(): digest(p) for p in prepared.rglob('*')
            if p.is_file() and 'source' not in p.relative_to(prepared).parts}
        write_json(prepared / 'protocol.json', plan)
        (prepared / 'protocol.sha256').write_text(digest(prepared / 'protocol.json') + '\n', encoding='utf-8')
        verify(run)
        print('Frozen quality-speed development protocol, source, warm start, baseline and shared courses.', flush=True)
