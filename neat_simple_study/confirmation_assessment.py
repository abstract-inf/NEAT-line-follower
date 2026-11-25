"""Freeze all selected controllers before generating or evaluating final tracks."""
import multiprocessing as mp
from pathlib import Path
import shutil

from line_follower.controllers import read_model_artifact
from neat_quality_study.training import genotype_hash
from neat_smooth_study.common import digest, lock, read, write_json
from neat_smooth_study.courses import make_suite, protected_geometry
from neat_smooth_study.engine import collect, initialize
from .confirmation import SEEDS, SETTINGS, SPEED_SCHEDULE, TEST_RULE, verify


def freeze_controllers(run, plan):
    run = Path(run)
    protocol_hash = digest(run / 'prepared/protocol.json')
    frozen = run / 'assessment/frozen'
    manifest_path = frozen / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest['protocol_sha256'] != protocol_hash:
            raise ValueError('Frozen assessment belongs to another protocol.')
        if any(digest(frozen / name) != row['sha256'] for name, row in manifest['controllers'].items()):
            raise ValueError('An assessment controller changed after freezing.')
        return manifest
    if (run / 'assessment/suite/suite.json').exists() or any((run / 'assessment/episodes').glob('*.json')):
        raise ValueError('Cannot select/freeze controllers after loading final tracks.')
    missing = [seed for seed in SEEDS if not (run / f'seed-{seed:02d}/complete.json').exists()]
    if missing:
        raise ValueError(f'All five seeds must finish before assessment; incomplete seeds: {missing}')
    records, source_files = {}, {}
    for seed in SEEDS:
        directory = run / f'seed-{seed:02d}'
        if not (directory / 'complete.json').exists():
            raise ValueError(f'All five seeds must finish before assessment; seed {seed} is incomplete.')
        completed = read(directory / 'complete.json')
        signature = completed['signature']
        initial = run / 'prepared' / plan['seeds'][str(seed)]['initial_population']
        if (completed['generations'] != SETTINGS['generations']
                or signature['protocol_sha256'] != protocol_hash or signature['seed'] != seed
                or signature['initial_population_sha256'] != digest(initial)):
            raise ValueError('Seed completion/initialization differs from the frozen confirmation.')
        selection = read(directory / 'selection.json')
        model = directory / 'best.pkl'
        artifact = read_model_artifact(model)
        metadata = artifact['metadata']
        if (completed['selected'] != selection or digest(model) != completed['best_sha256']
                or not SPEED_SCHEDULE[-1][0] <= selection['generation'] < SETTINGS['generations']
                or selection['cap_m_s'] != 1. or metadata['condition'] != 'simple-from-scratch-v1'
                or metadata['seed'] != seed or metadata['initial_population_sha256'] != digest(initial)
                or metadata['evolution_seed'] != plan['seeds'][str(seed)]['evolution_seed']
                or metadata['smoothness_weight'] != plan['smoothness_weight']
                or metadata['protocol_sha256'] != protocol_hash
                or metadata['genotype_sha256'] != selection['genotype_sha256']
                or genotype_hash(artifact['genome']) != selection['genotype_sha256']):
            raise ValueError('Final-stage selection, model identity or recorded ancestry differs.')
        name = f'neat-seed-{seed:02d}.pkl'
        records[name] = {'kind': 'neat', 'seed': seed, 'sha256': digest(model), 'selection': selection}
        source_files[name] = model
    for name, source in {'old-prototype.pkl': run / 'prepared/inputs/accepted-prototype.pkl',
                         'pid-100.json': run / 'prepared/inputs/pid-100.json',
                         'pid-050.json': run / 'prepared/inputs/pid-050.json'}.items():
        if name.startswith('pid') and read(source)['base'] != 1.:
            raise ValueError('The PID baseline must permit full-speed cruise.')
        records[name] = {'kind': 'pid' if name.startswith('pid') else 'development-neat-control',
                         'sha256': digest(source)}
        source_files[name] = source
    frozen.mkdir(parents=True, exist_ok=True)
    for name, source in source_files.items():
        shutil.copy2(source, frozen / name)
    manifest = {'protocol_sha256': protocol_hash, 'controllers': records, 'test_rule': TEST_RULE,
                'scope': 'Five independent validation-selected controllers, one previously accepted development prototype, cap-specific full-base PID. No test-based controller selection.'}
    write_json(manifest_path, manifest)
    return manifest


def prepare_test(run, plan):
    """The manifest is committed before any new test geometry is generated."""
    run = Path(run)
    manifest = freeze_controllers(run, plan)
    root = run / 'assessment'
    suite_path = root / 'suite/suite.json'
    freeze_path = root / 'test-freeze.json'
    if freeze_path.exists():
        saved = read(freeze_path)
        if (saved['controllers_sha256'] != digest(root / 'frozen/manifest.json')
                or saved['suite_sha256'] != digest(suite_path)
                or any(digest(root / 'suite' / name) != value for name, value in saved['track_files'].items())):
            raise ValueError('Final-test geometry or controller freeze changed.')
        return manifest, read(suite_path), saved
    if suite_path.exists():
        raise ValueError('Preserve incomplete test preparation; do not regenerate silently.')
    used = set(read(run / 'prepared/inputs/all-development-hashes.json'))
    project = run.parent.parent if run.parent.name == 'runs' else run
    used.update(protected_geometry(project))
    previous = set(used)
    suite = make_suite(root / 'suite', TEST_RULE['namespace'], {'test': TEST_RULE['counts']}, used)
    entries = suite['splits']['test']
    if len(entries) != 180 or len({entry['geometry_hash'] for entry in entries}) != 180:
        raise ValueError('Expected 180 distinct final-test geometries.')
    if {entry['geometry_hash'] for entry in entries} & previous:
        raise ValueError('Final-test geometry overlaps protected development data.')
    saved = {'controllers_sha256': digest(root / 'frozen/manifest.json'),
             'suite_sha256': digest(suite_path), 'previous_protected_geometries': len(previous),
             'overlap_count': 0,
             'track_files': {entry['path']: digest(root / 'suite' / entry['path']) for entry in entries}}
    write_json(freeze_path, saved)
    return manifest, suite, saved


def assess(run, workers=6):
    if workers not in (6, 7, 8):
        raise ValueError('Use six to eight workers.')
    run = Path(run).resolve()
    plan = verify(run)
    root = run / 'assessment'
    with lock(root):
        manifest, suite, freeze = prepare_test(run, plan)
        entries = suite['splits']['test']
        expected = [entry['geometry_hash'] for entry in entries]
        (root / 'episodes').mkdir(exist_ok=True)
        names = [f'neat-seed-{seed:02d}' for seed in SEEDS] + ['old-prototype', 'tuned-pid']
        previous_complete = read(root / 'complete.json') if (root / 'complete.json').exists() else None
        if previous_complete and any(digest(root / 'episodes' / name) != value
                                     for name, value in previous_complete['result_files'].items()):
            raise ValueError('Previously completed assessment output changed.')
        if previous_complete:
            return root
        new_records = reused_records = 0
        with mp.get_context('spawn').Pool(workers, initialize, (str(run), None),
                maxtasksperchild=SETTINGS['worker_recycle_tasks']) as pool:
            for cap in TEST_RULE['caps']:
                for name in names:
                    path = root / 'episodes' / f'{name}-cap-{round(cap*100):03d}.json'
                    if path.exists():
                        saved = read(path)
                        if (saved['controllers_sha256'] != freeze['controllers_sha256']
                                or saved['suite_sha256'] != freeze['suite_sha256']):
                            raise ValueError('Assessment cache belongs to a different freeze.')
                        rows = saved['episodes']
                        reused_records += len(rows)
                    else:
                        policy = {'parameters': read(root / f'frozen/pid-{round(cap*100):03d}.json')} if name == 'tuned-pid' else {
                            'model': str(root / 'frozen' / (name + '.pkl'))}
                        tasks = [{**policy, 'paths': [str(root / 'suite' / entry['path']) for entry in entries[start:start+4]],
                                  'groups': {entry['geometry_hash']: entry['group'] for entry in entries[start:start+4]},
                                  'cap': cap, 'detailed': True, 'branch': 'old'}
                                 for start in range(0, len(entries), 4)]
                        rows = [row for batch in collect(pool, tasks, f'{name} final cap {cap:g}') for row in batch]
                        new_records += len(rows)
                    if ([row['track_hash'] for row in rows] != expected
                            or any(row['cap_m_s'] != cap or row['error'] for row in rows)):
                        raise ValueError('Assessment row geometry/order/cap or validity differs.')
                    if not path.exists():
                        write_json(path, {'controllers_sha256': freeze['controllers_sha256'],
                                          'suite_sha256': freeze['suite_sha256'], 'episodes': rows})
        write_json(root / 'complete.json', {'protocol_sha256': digest(run / 'prepared/protocol.json'),
            'controllers_sha256': freeze['controllers_sha256'], 'suite_sha256': freeze['suite_sha256'],
            'assessment_episode_records': len(names) * len(TEST_RULE['caps']) * len(entries),
            'new_episode_records_in_final_invocation': new_records,
            'reused_episode_records_in_final_invocation': reused_records,
            'cost_scope': 'Dataset size and successfully retained work; unknown discarded partial simulations are not inferred.',
            'result_files': {path.name: digest(path) for path in sorted((root / 'episodes').glob('*.json'))}})
    return root
