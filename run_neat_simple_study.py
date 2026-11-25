"""A bounded development comparison of one-coefficient NEAT task fitness."""
import argparse
from functools import partial
import gzip
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

from line_follower.tracks import TrackSpec
from neat_smooth_study.common import dependencies, digest, lock, read, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / 'runs/neat-simple-pilot-v1'
VERSION = 'completion-speed-rms-1.0'
ARMS = {'lambda-0': 0., 'lambda-1': 1., 'lambda-10': 10.}
SETTINGS = {'generations': 20, 'population': 100, 'courses_per_genome': 6,
            'validation_candidates_per_generation': 3, 'worker_recycle_tasks': 128,
            'mask_cache_bytes': 64 * 1024**2, 'cap_m_s': 1.}
SOURCE_FOLDERS = ('line_follower', 'neat_smooth_study', 'neat_quality_study',
                  'neat_quality_followup', 'neat_simple_study', 'configs')


def source_inventory(root=ROOT):
    paths = [root / 'run_neat_simple_study.py', root / 'requirements-lock.txt']
    for folder in SOURCE_FOLDERS:
        paths.extend(path for path in (root / folder).rglob('*')
                     if path.is_file() and path.suffix in {'.py', '.txt', '.json', '.xml'})
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(paths)}


def verify(run):
    prepared = Path(run) / 'prepared'
    plan = read(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != digest(prepared / 'protocol.json'):
        raise ValueError('Simple pilot protocol changed after commitment.')
    expected_arms = {name: {'smoothness_weight': weight, 'parent': 'inputs/fast-parent.pkl',
                           'initial_population': 'initial/common.pkl.gz'}
                     for name, weight in ARMS.items()}
    if (plan['study'] != 'neat-simple-pilot-v1' or plan['fitness_version'] != VERSION
            or plan['settings'] != SETTINGS or plan['arms'] != expected_arms):
        raise ValueError('Pilot arms, budget or objective changed; use a new version.')
    if plan['dependencies'] != dependencies():
        raise ValueError('Dependencies differ from the frozen pilot.')
    if source_inventory() != plan['source_files'] or source_inventory(prepared / 'source') != plan['source_files']:
        raise ValueError('Pilot execution source differs from the freeze.')
    for relative, expected in plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError(f'Pilot input changed: {relative}')
    return plan


def verify_ancestor(original):
    prepared = original / 'prepared'
    plan = read(prepared / 'protocol.json')
    protocol_hash = digest(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != protocol_hash:
        raise ValueError('Ancestor protocol bytes changed.')
    if plan['dependencies'] != dependencies():
        raise ValueError('Ancestor dependencies differ.')
    for relative, expected in plan['source_files'].items():
        if digest(prepared / 'source' / relative) != expected:
            raise ValueError(f'Ancestor frozen source changed: {relative}')
    for relative, expected in plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError(f'Ancestor prepared input changed: {relative}')
    complete = read(original / 'development/fast/complete.json')
    signature = complete['signature']
    if (complete['generations'] != 50 or signature['protocol_sha256'] != protocol_hash
            or signature['initial_population_sha256'] != digest(prepared / 'initial/fast.pkl.gz')
            or signature['parent_sha256'] != digest(prepared / 'inputs/fast-parent.pkl')):
        raise ValueError('A completed, matching native-fast development parent is required.')
    return plan


def prepare(run):
    run = Path(run).resolve()
    if (run / 'prepared/protocol.json').exists():
        verify(run)
        print('Existing simple pilot verified; no preparation overwritten.')
        return
    original = ROOT / 'runs/neat-quality-followup-v1'
    prior = verify_ancestor(original)
    original_prepared = original / 'prepared'
    with lock(run):
        prepared = run / 'prepared'
        if prepared.exists():
            raise FileExistsError('Preserve partial preparation; choose a different run directory.')
        inputs = prepared / 'inputs'
        inputs.mkdir(parents=True)
        for name in ('fast-parent.pkl', 'reference-parent.pkl', 'reference-validation.json'):
            shutil.copy2(original_prepared / 'inputs' / name, inputs / name)
        for name, source in {
            'accepted-prototype.pkl': ROOT / 'results/neat-pid-finalization-20261125/frozen/neat-prototype.pkl',
            'pid-100.json': ROOT / 'results/neat-pid-finalization-20261125/frozen/pid-100.json',
            'pid-050.json': ROOT / 'results/neat-pid-finalization-20261125/frozen/pid-050.json',
            'controller-manifest.json': ROOT / 'results/neat-pid-finalization-20261125/frozen/manifest.json',
        }.items():
            shutil.copy2(source, inputs / name)
        controller_manifest = read(inputs / 'controller-manifest.json')['controllers']
        for frozen_name, manifest_name in (('accepted-prototype.pkl', 'neat-prototype.pkl'),
                                           ('pid-100.json', 'pid-100.json'), ('pid-050.json', 'pid-050.json')):
            if digest(inputs / frozen_name) != controller_manifest[manifest_name]['sha256']:
                raise ValueError('A development control differs from its accepted freeze.')
        shutil.copy2(original_prepared / 'protocol.json', inputs / 'ancestor-protocol.json')
        (prepared / 'initial').mkdir()
        shutil.copy2(original_prepared / 'initial/fast.pkl.gz', prepared / 'initial/common.pkl.gz')
        with gzip.open(prepared / 'initial/common.pkl.gz', 'rb') as stream:
            initial = pickle.load(stream)
        population = initial['population']
        if population.generation != 0 or len(population.population) != SETTINGS['population']:
            raise ValueError('Pilot must begin with the exact prepared 100-genome population.')
        config = population.config.genome_config
        if config.bias_mutate_power != .12 or config.weight_mutate_power != .14:
            raise ValueError('Pilot uses constant original mutation powers, not the later fine-tuning powers.')
        shutil.copytree(original_prepared / 'suite', prepared / 'suite')
        suite = read(prepared / 'suite/suite.json')['splits']
        held_out = [entry['geometry_hash'] for split in ('validation', 'retention') for entry in suite[split]]
        if len(set(held_out)) != 80 or len(suite['validation']) != 60 or len(suite['retention']) != 20:
            raise ValueError('Pilot requires 60+20 distinct development validation courses.')
        for generation in range(SETTINGS['generations']):
            source = original_prepared / 'batches' / f'generation-{generation:04d}'
            target = prepared / 'batches' / source.name
            shutil.copytree(source, target)
            hashes = read(target / 'batch.json')['hashes']
            if [TrackSpec.load(target / f'{slot}.json').geometry_hash for slot in range(6)] != hashes:
                raise ValueError('Pilot batch ordering differs from its ancestor.')
            if set(hashes) & set(held_out):
                raise ValueError('Pilot training and validation courses overlap.')
        source_files = source_inventory()
        for relative in source_files:
            target = prepared / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        prepared_files = {path.relative_to(prepared).as_posix(): digest(path)
                          for path in sorted(prepared.rglob('*'))
                          if path.is_file() and 'source' not in path.relative_to(prepared).parts}
        plan = {'study': 'neat-simple-pilot-v1', 'fitness_version': VERSION,
            'scope': 'Warm-start development sensitivity only; not independent training or final assessment.',
            'settings': SETTINGS, 'evolution_seed': prior['evolution_seed'],
            'robot': prior['robot'], 'episode': prior['episode'], 'dependencies': dependencies(),
            'arms': {name: {'smoothness_weight': weight, 'parent': 'inputs/fast-parent.pkl',
                            'initial_population': 'initial/common.pkl.gz'} for name, weight in ARMS.items()},
            'source_files': source_files, 'prepared_files': prepared_files,
            'ancestor_protocol_sha256': digest(original_prepared / 'protocol.json'),
            'objective': {'incomplete_utility': 'mean ordered-route progress',
                          'completed_utility': 'mean((route_length/(completion_time*cap))/(1+lambda*sqrt(jitter)))',
                          'fitness': 'completed_count + utility/(1+utility)',
                          'quality_reference_used_by_fitness': False},
            'controls': 'Identical initial genomes/RNG, 20 six-course batches, stock NEAT operators and parameters for all arms.',
            'historical_diagnostics': 'Legacy five quality violations and 80-episode gates are logged by the reused trainer only; they never enter the new breeding fitness. best.pkl remains historical-gate selection, not a new winner.',
            'future_confirmation': 'Requires a separate from-scratch frozen curriculum, fresh independent populations, all five seeds and a new untouched shared test suite. No automatic confirmation launch.',
            'budget': {'nominal_training_per_arm': 12000, 'maximum_validation_slots_per_arm': 4800,
                       'arms': 3, 'automatic_extensions': False},
            'coefficient_rationale': 'lambda 0 is the no-smoothness control. At RMS command change 0.1, lambda 1 discounts utility by 9.1% and lambda 10 by 50%. This limited development range is not claimed optimal.',
            'reporting_rule': 'Report every arm and generation; no historical failed gate is relabeled. Compare fixed-suite completion, finish time, RMSE, command-change RMS, yaw and measured straight speed. New evolution is required to establish learning effects.'}
        write_json(prepared / 'protocol.json', plan)
        (prepared / 'protocol.sha256').write_text(digest(prepared / 'protocol.json') + '\n')
    verify(run)
    print('Prepared three bounded development arms; evolution has not started.')


def compatible_score(episodes, reference, *, smoothness_weight):
    # Keep the existing trainer's historical diagnostic fields without allowing
    # those old thresholds, references or reward weights into breeding fitness.
    from neat_quality_study.fitness import score_batch as historical_measurements
    from neat_simple_study.fitness import score_batch
    historical = historical_measurements(episodes, reference)
    result = score_batch(episodes, smoothness_weight=smoothness_weight)
    return {**historical, **result, 'historical_fitness_diagnostic': historical['fitness'],
            'historical_quality_gates_are_diagnostic_only': True}


def preflight(run, workers):
    """Replay the parent through both model and prepared-genome dispatch."""
    import math
    from neat_smooth_study.engine import evaluate_tasks
    run = Path(run).resolve()
    plan = verify(run)
    if workers not in (6, 7, 8):
        raise ValueError('Use 6, 7 or 8 workers.')
    prepared = run / 'prepared'
    directory = prepared / 'batches/generation-0000'
    expected = read(directory / 'reference.json')
    parent = prepared / 'inputs/fast-parent.pkl'
    with gzip.open(prepared / 'initial/common.pkl.gz', 'rb') as stream:
        population = pickle.load(stream)['population']
    genome = next(iter(population.population.values()))
    paths = [str(directory / f'{slot}.json') for slot in range(6)]
    tasks = [{**policy, 'paths': [path], 'cap': 1., 'detailed': True, 'branch': 'old'}
             for policy in ({'model': str(parent)}, {'genome': genome}) for path in paths]
    observed = [rows[0] for rows in evaluate_tasks(run, tasks, workers, parent)]
    fields = ('progress', 'elapsed_s', 'control_steps', 'completion_time_s', 'command_jitter',
              'geometry_lateral_rmse_m', 'average_forward_speed_m_s', 'peak_forward_speed_m_s')
    for index, row in enumerate(observed):
        baseline = expected[index % 6]
        for field in ('completed', 'termination', 'track_hash'):
            if row[field] != baseline[field]:
                raise ValueError(f'Parent replay differs: {field}')
        for field in fields:
            actual, recorded = row[field], baseline[field]
            if actual is None or recorded is None:
                equal = actual is recorded
            else:
                equal = math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-10)
            if not equal:
                raise ValueError(f'Parent replay differs: {field}: {actual} != {recorded}')
    write_json(run / 'preflight/physics-replay.json', {
        'protocol_sha256': digest(prepared / 'protocol.json'), 'workers': workers,
        'parent_sha256': digest(parent), 'expected_reference_sha256': digest(directory / 'reference.json'),
        'initial_population_sha256': digest(prepared / 'initial/common.pkl.gz'),
        'episodes': len(observed), 'all_match': True, 'observed': observed})
    print('All 12 real parent/model-dispatch episodes matched the preserved physics records.')


def status(run, arms):
    import csv
    for arm in arms:
        directory = Path(run) / 'development' / arm
        path = directory / 'generations.csv'
        rows = list(csv.DictReader(path.open(encoding='utf-8', newline=''))) if path.exists() else []
        print(json.dumps({'arm': arm, 'committed_generations': len(rows),
                          'latest': rows[-1] if rows else None,
                          'complete': read(directory / 'complete.json') if (directory / 'complete.json').exists() else None}))


def dispatch(args):
    run = args.run.resolve()
    if args.command == 'prepare':
        return prepare(run)
    plan = verify(run)
    arms = (args.arm,) if args.arm else tuple(ARMS)
    if args.command == 'verify':
        print('Simple pilot source, inputs, objective and dependencies verified.')
    elif args.command == 'status':
        status(run, arms)
    elif args.command == 'preflight':
        preflight(run, args.workers)
    elif args.command == 'report':
        from neat_simple_study.reporting import report
        report(run)
    elif args.command == 'pilot':
        if args.workers not in (6, 7, 8):
            raise ValueError('Use 6, 7 or 8 workers, as requested for this study.')
        if args.arm is None and not args.all:
            raise ValueError('Choose --arm or explicitly request all three with --all.')
        preflight_path = run / 'preflight/physics-replay.json'
        if (not preflight_path.exists() or not read(preflight_path)['all_match']
                or read(preflight_path)['protocol_sha256'] != digest(run / 'prepared/protocol.json')):
            raise ValueError('Run the real physics preflight before evolution.')
        from neat_quality_followup.training import train
        for arm in arms:
            directory = run / 'development' / arm
            checkpoint = directory / 'checkpoints/latest.pkl.gz'
            completed = directory / 'complete.json'
            if completed.exists():
                saved = read(completed)
                if (saved['signature']['protocol_sha256'] != digest(run / 'prepared/protocol.json')
                        or saved['generations'] != SETTINGS['generations']):
                    raise ValueError('Completed pilot arm differs from the frozen protocol.')
                print(f'{arm}: verified completed arm; no extra evolution.')
                continue
            scorer = partial(compatible_score, smoothness_weight=plan['arms'][arm]['smoothness_weight'])
            train(run, arm, args.workers, resume=args.resume and checkpoint.exists(),
                  verify_protocol=verify, fitness_function=scorer, fitness_version=VERSION,
                  condition='simple-reward-development')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'preflight', 'pilot', 'status', 'report'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--frozen-source', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == 'prepare' or args.frozen_source:
        return dispatch(args)
    source = args.run.resolve() / 'prepared/source'
    if source_inventory(source) != read(args.run / 'prepared/protocol.json')['source_files']:
        raise ValueError('Frozen simple pilot source changed.')
    command = [sys.executable, str(source / 'run_neat_simple_study.py'),
               *(argv if argv is not None else sys.argv[1:]), '--run', str(args.run.resolve()), '--frozen-source']
    result = subprocess.run(command, cwd=source)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
