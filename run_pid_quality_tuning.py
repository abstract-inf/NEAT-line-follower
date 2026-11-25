"""Reproducible PID development search, including the previously inspected long course."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import time
import traceback

import numpy as np

from line_follower.pid import ResearchPIDConfig
from line_follower.suites import derived_seed
from line_follower.tracks import TrackSpec
from neat_smooth_study.common import dependencies, digest, lock, read, write_json
from neat_smooth_study.engine import Context
from neat_smooth_study.tuning import FIELDS, BOUNDS

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / 'runs/pid-quality-tuning-v1'
SEED = 2026112502
CAPS = (1., 1.5, .5)
SETTINGS = {'seed': SEED, 'caps': CAPS, 'coarse_candidates': 130, 'promoted': 12,
            'refinement_rounds': 2, 'refinements_per_round': 24, 'finalists': 10,
            'base': 1., 'gain_fields': FIELDS, 'bounds': BOUNDS,
            'minimum_validation_completion': .95, 'minimum_isolated_speed_fraction': .95,
            'fastest_time_allowance': 1.02, 'long_course_timeout': '180 / min(cap, 1)',
            'mask_cache_bytes_per_worker': 64 * 1024**2}


def normalized(value):
    return json.loads(json.dumps(value))


def source_inventory(root=ROOT):
    paths = [root / Path(__file__).name, root / 'requirements-lock.txt']
    for folder in ('line_follower', 'neat_smooth_study'):
        paths.extend(p for p in (root / folder).rglob('*')
                     if p.is_file() and p.suffix in {'.py', '.xml', '.json', '.txt'})
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def verify(run):
    prepared = Path(run) / 'prepared'
    plan = read(prepared / 'protocol.json')
    if (prepared / 'protocol.sha256').read_text().strip() != digest(prepared / 'protocol.json'):
        raise ValueError('PID protocol changed after preparation.')
    if plan['settings'] != normalized(SETTINGS) or plan['dependencies'] != dependencies():
        raise ValueError('PID settings or runtime dependencies changed.')
    if source_inventory() != plan['source_files'] or source_inventory(prepared / 'source') != plan['source_files']:
        raise ValueError('PID execution source differs from the frozen source.')
    for relative, expected in plan['prepared_files'].items():
        if digest(prepared / relative) != expected:
            raise ValueError(f'Frozen PID input changed: {relative}')
    return plan


def prepare(run):
    run = Path(run).resolve()
    if (run / 'prepared/protocol.json').exists():
        verify(run)
        print('Existing PID preparation verified.', flush=True)
        return
    old = ROOT / 'runs/neat-smooth-study-v1'
    parent_path = ROOT / 'runs/neat-quality-speed-v1/prepared/protocol.json'
    parent = read(parent_path)
    if parent['dependencies'] != dependencies():
        raise ValueError('Use the exact frozen quality-speed dependency environment.')
    for relative, expected in parent['source_files'].items():
        if relative.startswith(('line_follower/', 'neat_smooth_study/')) and digest(ROOT / relative) != expected:
            raise ValueError(f'Physics, controller or evaluator changed: {relative}')
    originals = {
        'quality-speed-protocol.json': parent_path,
        'pid-100.json': old / 'pid/cap-100/pid-fullspeed.json',
        'pid-100-selection.json': old / 'pid/cap-100/selection-freeze.json',
        'pid-050.json': old / 'pid/cap-050/pid-fullspeed.json',
        'pid-050-selection.json': old / 'pid/cap-050/selection-freeze.json',
        'pid-legacy-150.json': old / 'prepared/inputs/pid-legacy-150.json',
        'long-course.json': ROOT / 'runs/rhythm-changes-case-study-20261121/track.json',
    }
    for cap in ('100', '050'):
        selected = read(originals[f'pid-{cap}-selection.json'])
        if not selected['eligible'] or selected['sha256'] != digest(originals[f'pid-{cap}.json']):
            raise ValueError(f'Expected the eligible saved full-base {cap} PID.')
        if read(originals[f'pid-{cap}.json'])['base'] != 1.:
            raise ValueError('Saved full-speed PID must have base=1.')
    splits = read(old / 'prepared/pid/suite/suite.json')['splits']
    if len(splits['development']) != 50 or len(splits['validation']) != 60:
        raise ValueError('Expected the existing 50-development/60-selection PID suite.')
    with lock(run):
        prepared = run / 'prepared'
        if prepared.exists():
            raise FileExistsError('Partial preparation exists; preserve it and choose a new run.')
        (prepared / 'inputs').mkdir(parents=True)
        for name, source in originals.items():
            shutil.copy2(source, prepared / 'inputs' / name)
        shutil.copytree(old / 'prepared/pid/suite', prepared / 'suite')
        suite = read(prepared / 'suite/suite.json')['splits']
        inventory = source_inventory()
        for relative in inventory:
            target = prepared / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        track = TrackSpec.load(prepared / 'inputs/long-course.json')
        plan = {
            'study': 'pid-quality-tuning-v1', 'stage': 'development',
            'prepared_utc': datetime.now(timezone.utc).isoformat(),
            'settings': normalized(SETTINGS), 'dependencies': dependencies(),
            'source_files': inventory, 'robot': parent['robot'], 'episode': parent['episode'],
            'input_sources': {name: str(path) for name, path in originals.items()},
            'suite_source': str(old / 'prepared/pid/suite'),
            'course_counts': {name: len(entries) for name, entries in suite.items()},
            'maximum_episode_records_per_cap': (SETTINGS['coarse_candidates'] * 13 +
                (SETTINGS['promoted'] + SETTINGS['refinement_rounds'] * SETTINGS['refinements_per_round']) *
                (len(suite['development']) + 1) + SETTINGS['finalists'] * (len(suite['validation']) + 1)),
            'long_course': {'geometry_hash': track.geometry_hash, 'route_length_m': track.route.length,
                            'role': 'Previously inspected Rhythm Changes course, now explicit DEVELOPMENT/tuning data; never fresh final-test evidence.'},
            'controller': 'Stock ResearchPIDController; binary sensors, control time, recovery, physical robot and interface unchanged. Only the five declared fields vary; base=1 for every policy.',
            'search': 'Always run 130 coarse policies, including the current tune and tiny gain perturbations; promote 12 to every existing development course, two rounds of 24, up to 10 selection finalists. No cap-1 early return.',
            'reference_bounds': 'Saved references are evaluated exactly (legacy cap-1.5 gains may lie outside search bounds); all generated search points are clipped to the established five-field bounds.',
            'selection': 'Require every isolated window observed at >=.95*cap, >=95% completion on the 60 selection courses, and completion of the long case. Maximize total completion, keep within 2% of fastest failure-penalized mean time, then minimize straight yaw, jitter, geometric RMSE and time.',
            'data_policy': 'Development and selection reuse disclosed existing courses. Freeze eligible observed winners only; preserve every evaluated candidate, failure, rejection, runtime and paired denominator. No global-optimum or independent-final-test claim.',
        }
        plan['prepared_files'] = {p.relative_to(prepared).as_posix(): digest(p)
                                  for p in prepared.rglob('*') if p.is_file() and 'source' not in p.relative_to(prepared).parts}
        write_json(prepared / 'protocol.json', plan)
        (prepared / 'protocol.sha256').write_text(digest(prepared / 'protocol.json') + '\n', encoding='utf-8')
    verify(run)
    print(f'Frozen PID source, inputs, suites and {track.route.length:.2f} m DEVELOPMENT long case.', flush=True)


def references(run, cap):
    inputs = Path(run) / 'prepared/inputs'
    full = read(inputs / 'pid-100.json')
    saved = {'fullspeed-100': full, 'fullspeed-050': read(inputs / 'pid-050.json'),
             'legacy-150-base-one': {**read(inputs / 'pid-legacy-150.json'), 'base': 1.}}
    current = {1.: 'fullspeed-100', .5: 'fullspeed-050', 1.5: 'legacy-150-base-one'}[cap]
    ordered = {'current-tune': saved[current], **{name: p for name, p in saved.items() if name != current}}
    for parameters in ordered.values():
        ResearchPIDConfig(**parameters)
        if parameters['base'] != 1. or any(parameters[k] != full[k] for k in full if k not in FIELDS):
            raise ValueError('References differ beyond the five gain fields or base=1.')
    return ordered


def initial_candidates(controls, cap):
    policies = dict(controls)
    seen = {tuple(p[k] for k in FIELDS) for p in policies.values()}
    base = controls['current-tune']

    def add(values, prefix):
        point = tuple(float(np.clip(v, *bounds)) for v, bounds in zip(values, BOUNDS, strict=True))
        if point in seen:
            return None
        seen.add(point)
        name = f'{prefix}-{len(policies):03d}'
        parameters = {**base, **dict(zip(FIELDS, point, strict=True))}
        ResearchPIDConfig(**parameters)
        policies[name] = parameters
        return name

    center = np.array([base[k] for k in FIELDS])
    # Explicit small perturbations test gains that an earlier feasibility shortcut never searched.
    for scale, prefix in ((.03, 'tiny'), (.20, 'axis')):
        for axis in range(len(FIELDS)):
            for direction in (-1, 1):
                point = center.copy()
                point[axis] += direction * max(abs(center[axis]) * scale, .002 if axis == 1 else .0001)
                add(point, prefix)
    for kp in (.1, .3, .7, 1.5, 3.):
        for kd in (.005, .015, .04, .09, .2):
            for ki, filtering, slowdown in ((0., .06, 1.), (.08, .14, 4.)):
                add((kp, ki, kd, filtering, slowdown), 'grid')
    rng = np.random.default_rng(derived_seed(SEED, 'coarse', round(cap * 100)))
    while len(policies) < SETTINGS['coarse_candidates']:
        add((np.exp(rng.uniform(np.log(.05), np.log(3.))),
             0. if len(policies) % 3 == 0 else rng.uniform(0., .25),
             np.exp(rng.uniform(np.log(.005), np.log(.2))), rng.uniform(.02, .2),
             np.exp(rng.uniform(np.log(.3), np.log(10.)))), 'random')
    return policies, add


def long_episode(episode, cap):
    limit = 180. / min(cap, 1.)
    return {**episode, 'minimum_time_limit': limit, 'maximum_time_limit': limit}


def evaluate_policy(context, task):
    """Local adapter supplies an explicit case timeout without changing the frozen engine."""
    entries = task['entries']
    regular = [e for e in entries if e['kind'] != 'long_case']
    case = [e for e in entries if e['kind'] == 'long_case']
    policy = {'parameters': task['parameters'], 'cap': task['cap'], 'detailed': True}
    rows = []
    for selected in (regular, case):
        if not selected:
            continue
        old = context.plan
        if selected is case:
            context.plan = {**old, 'episode': long_episode(old['episode'], task['cap'])}
        try:
            records = context.evaluate({**policy, 'paths': [e['absolute_path'] for e in selected],
                                        'groups': {e['geometry_hash']: e['group'] for e in selected}})
        finally:
            context.plan = old
        if len(records) != len(selected):
            raise ValueError('Incomplete candidate evaluation.')
        for record, entry in zip(records, selected, strict=True):
            if record['track_hash'] != entry['geometry_hash']:
                raise ValueError('Evaluator returned different geometry.')
            rows.append({**record, 'kind': entry['kind'], 'analysis_role': 'development-tuning',
                         'suite_entry': entry['path']})
    return rows


_CONTEXT = None


def initialize_worker(run):
    global _CONTEXT
    _CONTEXT = Context(run)


def worker(task):
    started = time.perf_counter()
    try:
        result = {'rows': evaluate_policy(_CONTEXT, task), 'error': None}
    except Exception:
        result = {'rows': [], 'error': traceback.format_exc()}
    return {**result, 'wall_seconds': time.perf_counter() - started}


def summarize(result, cap):
    rows = result['rows']
    ordinary = [r for r in rows if r['kind'] != 'long_case']
    cases = [r for r in rows if r['kind'] == 'long_case']
    dedicated = [r for r in ordinary if r['kind'] == 'isolated']
    windows = [w for r in dedicated for w in r['straight_windows']]
    observed = [w for r in rows for w in r['straight_windows'] if w['observed']]
    feasible = bool(dedicated) and all(r['completed'] for r in dedicated) and bool(windows) and all(
        w['observed'] and w['speed_m_s'] is not None and w['speed_m_s'] >= .95 * cap - 1e-12 for w in windows)
    completion = sum(r['completed'] for r in ordinary)
    reasons = []
    if result['error']:
        reasons.append('evaluation_error')
    if len(rows) != result.get('expected_episodes', len(rows)):
        reasons.append('incomplete_episode_set')
    if not feasible:
        reasons.append('isolated_straight_speed')
    if not ordinary or completion < .95 * len(ordinary):
        reasons.append('suite_completion_below_95_percent')
    if len(cases) != 1 or not cases[0]['completed']:
        reasons.append('long_case_incomplete')
    return {'eligible': not reasons, 'rejection_reasons': reasons, 'error': result['error'],
            'completed': sum(r['completed'] for r in rows), 'tracks': len(rows),
            'suite_completed': completion, 'suite_tracks': len(ordinary),
            'long_case_completed': len(cases) == 1 and bool(cases[0]['completed']),
            'speed_feasible': feasible, 'isolated_tracks': len(dedicated),
            'required_isolated_windows': len(windows), 'observed_straight_windows': len(observed),
            'minimum_isolated_speed_m_s': min((w['speed_m_s'] for w in windows if w['observed']), default=None),
            'failure_penalized_mean_time_s': mean(r['completion_time_s'] if r['completed'] else r['time_limit_s'] for r in rows) if rows else 1e9,
            'steady_yaw_rms': mean(w['yaw_rms_rad_s'] for w in observed) if observed else 1e9,
            'command_jitter': mean(r['command_jitter'] for r in rows) if rows else 1e9,
            'geometric_rmse_m': mean(r['geometry_lateral_rmse_m'] for r in rows) if rows else 1e9,
            'failure_terminations': dict(Counter(r['termination'] for r in rows if not r['completed']))}


def rank(summary):
    return (summary['speed_feasible'], summary['completed'],
            -summary['failure_penalized_mean_time_s'], -summary['steady_yaw_rms'],
            -summary['command_jitter'], -summary['geometric_rmse_m'])


def promote(summaries, limit, required=()):
    usable = [n for n, s in summaries.items() if not s['error']]
    feasible = [n for n in usable if summaries[n]['speed_feasible']]
    candidates = feasible or usable  # Continue a failed search; this never weakens the final gate.
    if not candidates:
        raise ValueError('All candidates raised evaluation errors; raw errors were preserved.')
    fastest = sorted(candidates, key=lambda n: rank(summaries[n]), reverse=True)
    maximum = max(summaries[n]['completed'] for n in candidates)
    smooth = sorted((n for n in candidates if summaries[n]['completed'] == maximum),
                    key=lambda n: (summaries[n]['steady_yaw_rms'], summaries[n]['command_jitter']))
    required = [n for n in required if n in summaries]
    return list(dict.fromkeys([*required, *fastest[:8], *smooth[:4], *fastest]))[:limit]


def select(summaries):
    eligible = [n for n, s in summaries.items() if s['eligible']]
    if not eligible:
        return None, []
    maximum = max(summaries[n]['completed'] for n in eligible)
    eligible = [n for n in eligible if summaries[n]['completed'] == maximum]
    fastest = min(summaries[n]['failure_penalized_mean_time_s'] for n in eligible)
    shortlist = [n for n in eligible if summaries[n]['failure_penalized_mean_time_s'] <= 1.02 * fastest + 1e-12]
    winner = min(shortlist, key=lambda n: (summaries[n]['steady_yaw_rms'], summaries[n]['command_jitter'],
                                        summaries[n]['geometric_rmse_m'], summaries[n]['failure_penalized_mean_time_s'], n))
    return winner, shortlist


def paired(candidate, control):
    left = {r['track_hash']: r for r in candidate['rows']}
    right = {r['track_hash']: r for r in control['rows']}
    if len(left) != len(candidate['rows']) or len(right) != len(control['rows']):
        raise ValueError('Duplicate geometry in paired records.')
    keys = sorted(left.keys() & right.keys())
    shared = [(left[k], right[k]) for k in keys if left[k]['completed'] and right[k]['completed']]
    result = {'candidate_tracks': len(left), 'control_tracks': len(right), 'paired_tracks': len(keys),
              'identical_track_sets': left.keys() == right.keys(), 'shared_finishes': len(shared),
              'candidate_only_finishes': sum(left[k]['completed'] and not right[k]['completed'] for k in keys),
              'control_only_finishes': sum(right[k]['completed'] and not left[k]['completed'] for k in keys)}
    for field in ('completion_time_s', 'command_jitter', 'geometry_lateral_rmse_m'):
        result[field] = {'denominator': len(shared),
                         'candidate_mean': mean(a[field] for a, _ in shared) if shared else None,
                         'control_mean': mean(b[field] for _, b in shared) if shared else None}
    return result


def phase_entries(run, split, coarse=False):
    run = Path(run)
    entries = read(run / 'prepared/suite/suite.json')['splits'][split]
    if coarse:
        entries = ([e for e in entries if e['kind'] == 'isolated'][:4] +
                   [e for e in entries if e['kind'] == 'post_turn'][:4] +
                   [e for e in entries if e['group'] == 'mixed'][:2] +
                   [e for e in entries if e['group'] == 'corner'][:2])
        if len(entries) != 12:
            raise ValueError('Expected 12 stratified coarse development courses.')
    output = [{**e, 'absolute_path': str(run / 'prepared/suite' / e['path'])} for e in entries]
    track = TrackSpec.load(run / 'prepared/inputs/long-course.json')
    output.append({'path': 'inputs/long-course.json', 'absolute_path': str(run / 'prepared/inputs/long-course.json'),
                   'geometry_hash': track.geometry_hash, 'kind': 'long_case', 'group': 'long_case'})
    return output


def batch(pool, directory, signature, policies, entries, cap, label):
    records, tasks, names = {}, [], []
    phase_signature = {**signature, 'phase': label,
                       'entries': [{k: v for k, v in e.items() if k != 'absolute_path'} for e in entries]}
    for name, parameters in policies.items():
        path = directory / 'episodes' / label / f'{name}.json'
        if path.exists():
            record = read(path)
            if record['parameters'] != parameters or record['signature'] != phase_signature:
                raise ValueError(f'Stale candidate cache: {path}')
            records[name] = record
        else:
            tasks.append({'parameters': parameters, 'entries': entries, 'cap': cap})
            names.append(name)
    completed = 0
    for start in range(0, len(tasks), max(1, pool._processes)):
        block = tasks[start:start + pool._processes]
        for name, result in zip(names[start:start + len(block)], pool.imap(worker, block), strict=True):
            record = {**result, 'parameters': policies[name], 'signature': phase_signature,
                      'expected_episodes': len(entries), 'committed_utc': datetime.now(timezone.utc).isoformat()}
            write_json(directory / 'episodes' / label / f'{name}.json', record)
            records[name] = record
            completed += 1
            print(f'PID {cap:.2f} {label}: {completed}/{len(tasks)} new candidates committed; '
                  f'completed={sum(r["completed"] for r in record["rows"])}/{len(entries)}', flush=True)
    summaries = {n: summarize(r, cap) for n, r in records.items()}
    write_json(directory / f'{label}-summary.json', summaries)
    return records


def tune_cap(run, cap, workers):
    run = Path(run)
    directory = run / f'cap-{round(cap * 100):03d}'
    signature = {'protocol_sha256': digest(run / 'prepared/protocol.json'), 'cap': cap}
    with lock(directory):
        if (directory / 'complete.json').exists():
            complete = read(directory / 'complete.json')
            if complete['signature'] != signature:
                raise ValueError('Completed search has different inputs.')
            if complete['eligible']:
                frozen = read(directory / 'selection-freeze.json')
                if digest(directory / frozen['file']) != frozen['sha256']:
                    raise ValueError('Selected PID changed.')
            print(f'Completed PID cap {cap:.2f} search verified.', flush=True)
            return
        started = time.perf_counter()
        controls = references(run, cap)
        policies, add = initial_candidates(controls, cap)
        write_json(directory / 'candidates.json', policies)
        with mp.get_context('spawn').Pool(workers, initialize_worker, (str(run),), maxtasksperchild=128) as pool:
            coarse = batch(pool, directory, signature, policies, phase_entries(run, 'development', True), cap, 'coarse')
            promoted = promote({n: summarize(r, cap) for n, r in coarse.items()}, 12, ('current-tune',))
            results = batch(pool, directory, signature, {n: policies[n] for n in promoted},
                            phase_entries(run, 'development'), cap, 'expanded')
            for generation in (1, 2):
                parents = promote({n: summarize(r, cap) for n, r in results.items()}, 6)
                rng = np.random.default_rng(derived_seed(SEED, 'refine', round(cap * 100), generation))
                children = []
                for index in range(24):
                    center = np.array([policies[parents[index % len(parents)]][k] for k in FIELDS])
                    factor = 1. if generation == 1 else .35
                    scales = factor * np.array([max(.04, center[0] * .15), .025,
                                                max(.005, center[2] * .15), .012, max(.3, center[4] * .2)])
                    for _ in range(100):
                        name = add(rng.normal(center, scales), f'refine{generation}')
                        if name:
                            children.append(name)
                            break
                    else:
                        raise ValueError('Could not produce a distinct refinement.')
                write_json(directory / 'candidates.json', policies)
                results.update(batch(pool, directory, signature, {n: policies[n] for n in children},
                                     phase_entries(run, 'development'), cap, f'refine-{generation}'))
            finalists = promote({n: summarize(r, cap) for n, r in results.items()}, 10, ('current-tune',))
            validation = batch(pool, directory, signature, {n: policies[n] for n in finalists},
                               phase_entries(run, 'validation'), cap, 'selection')
        summaries = {n: summarize(r, cap) for n, r in validation.items()}
        winner, shortlist = select(summaries)
        best = winner or max(summaries, key=lambda n: rank(summaries[n]))
        report = {'signature': signature, 'candidate': winner, 'best_observed': best,
                  'eligible': winner is not None, 'shortlist': shortlist, 'summaries': summaries,
                  'paired_against_current': {n: paired(r, validation['current-tune']) for n, r in validation.items()},
                  'candidates': len(policies), 'committed_episode_records': sum(len(read(p)['rows']) for p in (directory / 'episodes').rglob('*.json')),
                  'claim_scope': 'Best observed deterministic development search; consulted selection and long-case data, no global optimum or fresh-final-test claim.'}
        write_json(directory / 'report.json', report)
        if winner:
            if policies[winner]['base'] != 1. or not summaries[winner]['eligible']:
                raise ValueError('Refusing to freeze an ineligible PID.')
            target = directory / 'pid-fullspeed.json'
            write_json(target, policies[winner])
            write_json(directory / 'selection-freeze.json', {'file': target.name, 'sha256': digest(target),
                       'candidate': winner, 'eligible': True, 'summary': summaries[winner], 'signature': signature})
        else:
            write_json(directory / 'failed.json', {'reason': 'No candidate met all original full-speed, completion and long-case gates.',
                       'best_observed': best, 'summary': summaries[best], 'frozen': False})
        write_json(directory / 'complete.json', {'signature': signature, 'eligible': winner is not None,
                   'selected': winner, 'best_observed': best, 'wall_seconds_this_invocation': time.perf_counter() - started,
                   'candidates': len(policies), 'episode_records': report['committed_episode_records']})
        print(json.dumps({'cap': cap, 'selected': winner, 'best_observed': best, 'summary': summaries[best]}, indent=2), flush=True)


def dispatch(args):
    run = args.run.resolve()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if args.command == 'prepare':
        return prepare(run)
    verify(run)
    if args.command == 'verify':
        print('PID frozen source, inputs, settings and dependencies verified.', flush=True)
    elif args.command == 'tune':
        with lock(run):
            runtime = run / 'runtime' / f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.json'
            runtime_record = {'started_utc': datetime.now(timezone.utc).isoformat(),
                              'workers': args.workers, 'caps': args.caps, 'source': 'prepared/source',
                              'committed_episode_counts_exclude_unknown_partial_work': True, 'status': 'running'}
            write_json(runtime, runtime_record)
            try:
                for cap in args.caps:
                    tune_cap(run, cap, args.workers)
                runtime_record['status'] = 'completed'
            except BaseException:
                runtime_record.update(status='interrupted_or_failed', error=traceback.format_exc())
                raise
            finally:
                runtime_record['ended_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(runtime, runtime_record)
    else:
        for cap in CAPS:
            directory = run / f'cap-{round(cap * 100):03d}'
            completed = directory / 'complete.json'
            print(json.dumps({'cap': cap, 'complete': read(completed) if completed.exists() else None,
                              'candidate_records': len(list((directory / 'episodes').rglob('*.json')))}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'tune', 'status'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--caps', type=float, nargs='+', choices=CAPS, default=CAPS)
    parser.add_argument('--frozen-source', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == 'prepare' or args.frozen_source:
        return dispatch(args)
    run = args.run.resolve()
    source = run / 'prepared/source'
    if source_inventory(source) != read(run / 'prepared/protocol.json')['source_files']:
        raise ValueError('Frozen PID launcher source changed.')
    command = [sys.executable, str(source / Path(__file__).name),
               *(argv if argv is not None else sys.argv[1:]), '--run', str(run), '--frozen-source']
    result = subprocess.run(command, cwd=source)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    mp.freeze_support()
    main()
