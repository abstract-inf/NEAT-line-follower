"""Offline counterfactual fitness ranking; never imports or runs the experiment."""
import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
FORMULA = 'Keep incomplete fitness; completed: 10*courses + 9*speed_utility/(1+quality_violation)'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def csv_rows(data):
    # A concurrent append can leave a partial final line. Preserve its bytes in
    # the snapshot but do not treat it as a complete logged record.
    text = data.decode('utf-8-sig')
    if text and not text.endswith(('\r', '\n')):
        text = text[:text.rfind('\n') + 1]
    return list(csv.DictReader(io.StringIO(text, newline='')))


def rescore(row):
    completed, courses = int(row['completed']), int(row['courses'])
    speed, violation, original = (float(row[key]) for key in
                                  ('speed_utility', 'quality_violation', 'fitness'))
    if not (0 <= completed <= courses and courses > 0 and
            all(math.isfinite(x) for x in (speed, violation, original)) and
            0 <= speed <= 1 and violation >= 0):
        raise ValueError('Invalid saved fitness components.')
    if completed < courses:
        if not 10*completed <= original <= 10*completed + 9 + 1e-10:
            raise ValueError('Incomplete fitness is outside the frozen V1 band.')
        return original
    expected = 10*courses + (10 + 9*speed if violation == 0 else 9/(1+violation))
    if not math.isclose(original, expected, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError('Saved complete-batch fitness does not match V1.')
    return 10*courses + 9*speed/(1+violation)


def synthetic_checks():
    def full(speed, violation):
        original = 60 + (10 + 9*speed if violation == 0 else 9/(1+violation))
        return rescore(dict(completed=6, courses=6, speed_utility=speed,
                            quality_violation=violation, fitness=original))

    incomplete = dict(completed=5, courses=6, speed_utility=1,
                      quality_violation=0, fitness=58.25)
    checks = {
        'incomplete_exactly_preserved': rescore(incomplete) == 58.25,
        'completion_dominates_even_zero_completed_speed': full(0, 1000) > 59,
        'every_adjacent_completion_band_separated': all(10*c + 9 < 10*(c+1) for c in range(6)),
        'sub_limit_quality_plateau': full(.75, 0) == 66.75,
        'speed_visible_with_positive_excess': full(.95, .05) > full(.85, .05),
        'faster_small_excess_can_beat_slower_feasible': full(.95, .05) > full(.75, 0),
        'violation_still_costly_at_fixed_speed': full(.95, .5) < full(.95, .05),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {'checks': checks, 'examples': [
        {'case': 'slower feasible', 'S': .75, 'V': 0, 'proposal': full(.75, 0), 'original': 76.75},
        {'case': 'faster small excess', 'S': .95, 'V': .05, 'proposal': full(.95, .05),
         'original': 60 + 9/1.05},
        {'case': 'faster larger excess', 'S': .95, 'V': .5, 'proposal': full(.95, .5),
         'original': 60 + 9/1.5},
    ]}


def capture_inputs(run):
    directory = run / 'development/seed-01'
    sources = {'generations.csv': directory / 'generations.csv',
               'genome-fitness.csv': directory / 'genome-fitness.csv',
               'protocol.json': run / 'prepared/protocol.json',
               'v1-fitness.py': run / 'prepared/source/neat_quality_study/fitness.py'}
    # Read the generation log first: later genome rows are not automatically in scope.
    payload = {name: path.read_bytes() for name, path in sources.items()}
    generation_rows = csv_rows(payload['generations.csv'])
    if not generation_rows:
        raise ValueError('No complete logged generation rows.')
    for row in generation_rows:
        name = f"validation/generation-{int(row['generation']):04d}.json"
        sources[name] = directory / name
        payload[name] = sources[name].read_bytes()
    calibration = ROOT / 'results/neat-quality-speed-v1/fitness-calibration.json'
    if calibration.exists():
        sources['fitness-calibration.json'] = calibration
        payload['fitness-calibration.json'] = calibration.read_bytes()
    complete = directory / 'complete.json'
    if complete.exists():
        sources['complete.json'] = complete
        payload['complete.json'] = complete.read_bytes()
    manifest = {'formula': FORMULA, 'sources': {
        name: {'original_path': str(sources[name]), 'sha256': sha256(data), 'bytes': len(data)}
        for name, data in sorted(payload.items())}}
    return payload, manifest


def load_snapshot(snapshot):
    manifest = json.loads((snapshot / 'input-manifest.json').read_bytes())
    payload = {name: (snapshot / 'inputs' / name).read_bytes() for name in manifest['sources']}
    for name, data in payload.items():
        if sha256(data) != manifest['sources'][name]['sha256']:
            raise ValueError(f'Snapshot hash mismatch: {name}')
    if sha256(Path(__file__).read_bytes()) != manifest['diagnostic_source_sha256']:
        raise ValueError('Use the diagnostic.py preserved with this snapshot to replay its version.')
    return payload, manifest


def save_snapshot(output, payload, manifest):
    source = Path(__file__).read_bytes()
    manifest['diagnostic_source_sha256'] = sha256(source)
    identity = sha256(json.dumps(manifest, sort_keys=True).encode())[:16]
    last = max(int(row['generation']) for row in csv_rows(payload['generations.csv']))
    snapshot = output / 'snapshots' / f'g{last:04d}-{identity}'
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        target = snapshot / 'inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Existing immutable snapshot differs.')
        target.write_bytes(data)
    (snapshot / 'diagnostic.py').write_bytes(source)
    (snapshot / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return snapshot


def rank_rows(rows, identity_key):
    ranked = sorted(rows, key=lambda row: (-row['fitness'], row[identity_key]))
    old_ranks = {row[identity_key]: rank for rank, row in enumerate(ranked, 1)}
    proposed = sorted(rows, key=lambda row: (-row['proposal_fitness'], row[identity_key]))
    new_ranks = {row[identity_key]: rank for rank, row in enumerate(proposed, 1)}
    for row in rows:
        row['original_rank'] = old_ranks[row[identity_key]]
        row['proposal_rank'] = new_ranks[row[identity_key]]
        row['rank_gain'] = row['original_rank'] - row['proposal_rank']
    return ranked, proposed


def analyze(payload, manifest):
    generation_rows = csv_rows(payload['generations.csv'])
    generation_log = {int(row['generation']): row for row in generation_rows}
    generations = sorted(generation_log)
    if generations != list(range(generations[-1] + 1)) or len(generation_log) != len(generation_rows):
        raise ValueError('Expected one complete logged row for every generation from zero.')
    plan = json.loads(payload['protocol.json'])
    if sha256(payload['v1-fitness.py']) != plan['source_files']['neat_quality_study/fitness.py']:
        raise ValueError('Captured frozen fitness source differs from protocol.')
    grouped = defaultdict(list)
    all_saved = csv_rows(payload['genome-fitness.csv'])
    for saved in all_saved:
        if int(saved['generation']) not in generation_log:
            continue
        row = {key: int(saved[key]) for key in ('generation', 'genome_id', 'completed', 'courses')}
        row.update({key: float(saved[key]) for key in ('fitness', 'speed_utility', 'quality_violation')})
        row['proposal_fitness'] = rescore(row)
        row['completed_quality_feasible'] = row['completed'] == row['courses'] and row['quality_violation'] == 0
        grouped[row['generation']].append(row)
    details, per_generation = [], []
    for generation in generations:
        rows = grouped[generation]
        if not rows or len({row['genome_id'] for row in rows}) != len(rows):
            raise ValueError('Missing or duplicate genome rows in a logged generation.')
        if any(row['courses'] != plan['settings']['courses_per_genome'] for row in rows):
            raise ValueError('Training course count differs from frozen protocol.')
        if len(rows)*rows[0]['courses'] != int(generation_log[generation]['training_episodes']):
            raise ValueError('Genome rows do not match the logged generation episode total.')
        saved = json.loads(payload[f'validation/generation-{generation:04d}.json'])['candidates']
        validated = {int(row['genome_id']): row for row in saved}
        for row in rows:
            row['archived_validation_this_generation'] = row['genome_id'] in validated
        original, proposed = rank_rows(rows, 'genome_id')
        old_top, new_top = original[:3], proposed[:3]
        per_generation.append({
            'generation': generation, 'population': len(rows),
            'fully_completed': sum(row['completed'] == row['courses'] for row in rows),
            'completed_quality_feasible': sum(row['completed_quality_feasible'] for row in rows),
            'original_top1': original[0]['genome_id'], 'proposal_top1': proposed[0]['genome_id'],
            'top1_changed': original[0]['genome_id'] != proposed[0]['genome_id'],
            'original_top1_completed_quality_feasible': original[0]['completed_quality_feasible'],
            'proposal_top1_completed_quality_feasible': proposed[0]['completed_quality_feasible'],
            'original_top1_speed_utility': original[0]['speed_utility'],
            'proposal_top1_speed_utility': proposed[0]['speed_utility'],
            'original_top1_quality_violation': original[0]['quality_violation'],
            'proposal_top1_quality_violation': proposed[0]['quality_violation'],
            'original_top3_ids': ';'.join(str(row['genome_id']) for row in old_top),
            'proposal_top3_ids': ';'.join(str(row['genome_id']) for row in new_top),
            'top3_overlap': len({row['genome_id'] for row in old_top} & {row['genome_id'] for row in new_top}),
            'original_top3_completed_quality_feasible': sum(row['completed_quality_feasible'] for row in old_top),
            'proposal_top3_completed_quality_feasible': sum(row['completed_quality_feasible'] for row in new_top),
            'proposal_top3_archived_this_generation': sum(row['genome_id'] in validated for row in new_top),
            'proposal_top1_archived_this_generation': proposed[0]['genome_id'] in validated,
            'genomes_with_changed_rank': sum(row['rank_gain'] != 0 for row in rows),
            'mean_absolute_rank_change': mean(abs(row['rank_gain']) for row in rows),
            'maximum_absolute_rank_change': max(abs(row['rank_gain']) for row in rows),
        })
        details.extend(rows)
    calibration = []
    if 'fitness-calibration.json' in payload:
        records = json.loads(payload['fitness-calibration.json'])['records']
        for saved in records:
            row = {key: saved[key] for key in ('branch', 'generation', 'completed', 'courses', 'fitness',
                                              'speed_utility', 'quality_violation')}
            row['identity'] = f"{row['branch']}:{row['generation']:04d}"
            row['proposal_fitness'] = rescore(row)
            row['completed_quality_feasible'] = row['completed'] == row['courses'] and row['quality_violation'] == 0
            calibration.append(row)
        rank_rows(calibration, 'identity')
        calibration.sort(key=lambda row: row['proposal_rank'])
    complete = json.loads(payload.get('complete.json', b'{}'))
    finished = len(generations) == plan['settings']['generations'] == complete.get('generations')
    summary = {
        'scope': 'Counterfactual ranking of saved development measurements; no re-evolution, simulation, or new controller selection.',
        'formula': FORMULA, 'registered_or_activated': False,
        'logged_generation_count': len(generations), 'first_generation': generations[0],
        'last_generation': generations[-1], 'planned_generations': plan['settings']['generations'],
        'complete_record_matches_captured_horizon': finished,
        'genome_rows_rescored': len(details), 'rows_excluded_outside_logged_horizon': len(all_saved)-len(details),
        'incomplete_rows_preserved': sum(row['completed'] < row['courses'] for row in details),
        'generations_with_top1_change': sum(row['top1_changed'] for row in per_generation),
        'generations_with_top3_genome_set_change': sum(row['top3_overlap'] < 3 for row in per_generation),
        'original_top1_quality_feasible_generations': sum(row['original_top1_completed_quality_feasible'] for row in per_generation),
        'proposal_top1_quality_feasible_generations': sum(row['proposal_top1_completed_quality_feasible'] for row in per_generation),
        'proposal_top1_without_archived_validation_this_generation': sum(not row['proposal_top1_archived_this_generation'] for row in per_generation),
        'calibration_records': len(calibration),
        'calibration_top10': calibration[:10], 'input_manifest': manifest,
    }
    return summary, per_generation, details, calibration


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(summary, per_generation, checks, snapshot):
    count = summary['logged_generation_count']
    status = 'Completed-horizon snapshot' if summary['complete_record_matches_captured_horizon'] else 'Interim logged-horizon snapshot'
    lines = ['# Offline quality-speed objective diagnostic', '',
        f"**{status}: generations {summary['first_generation']}–{summary['last_generation']} ({count}/{summary['planned_generations']} logged).** Later progress is outside this captured evidence.", '',
        'Conditional project proposal only; no new experiment is registered or activated. The frozen V1 run, thresholds, policy, training data and final acceptance rules are unchanged.', '',
        '`Completed batch: F = 10*m + 9*S/(1+V)`. Incomplete rows retain their exact saved V1 fitness. Existing S and V are reused without coefficients or threshold changes.', '',
        f"Rescored **{summary['genome_rows_rescored']}** genome rows; preserved {summary['incomplete_rows_preserved']} incomplete scores. Excluded {summary['rows_excluded_outside_logged_horizon']} captured rows outside the generation-log horizon. Top-one genome changed in **{summary['generations_with_top1_change']}/{count}** generations; the top-three genome set changed in **{summary['generations_with_top3_genome_set_change']}/{count}**.", '',
        f"The original versus proposed top-one completed-quality-feasible counts are **{summary['original_top1_quality_feasible_generations']} versus {summary['proposal_top1_quality_feasible_generations']}**. Proposed top-one genome IDs lack archived validation for that generation in {summary['proposal_top1_without_archived_validation_this_generation']} cases. Feasibility of the stored measurements themselves never changes.", '',
        '| Generation | Population | Completed quality-feasible | Top one, original → proposal | Top-three overlap | Feasible top one, original → proposal |',
        '|---:|---:|---:|---|---:|---|']
    for row in per_generation:
        lines.append(f"| {row['generation']} | {row['population']} | {row['completed_quality_feasible']} | {row['original_top1']} → {row['proposal_top1']} | {row['top3_overlap']}/3 | {row['original_top1_completed_quality_feasible']} → {row['proposal_top1_completed_quality_feasible']} |")
    lines += ['', '## Synthetic ordering', '',
        'All seven executable checks pass. Completion stays dominant: every incomplete score is at most `10*C+9`, whereas a completed batch scores at least `10*m`, even with zero speed utility. At V=0, F depends only on S: improving already acceptable quality has no reward. At fixed positive V, increasing S still increases fitness. At fixed S, violation still reduces fitness. S remains the existing capped utility, so its existing saturation is retained.', '',
        '| Synthetic completed batch | S | V | V1 score | Proposed score |', '|---|---:|---:|---:|---:|']
    for row in checks['examples']:
        lines.append(f"| {row['case']} | {row['S']} | {row['V']} | {row['original']:.6f} | {row['proposal']:.6f} |")
    lines += ['', 'The faster small-excess example can outrank the slower feasible example. That is the intended trade-off; strict feasible-first ordering is lost. Passing final quality and speed gates is still required.', '',
        '## Archived 60-course calibration', '']
    if summary['calibration_records']:
        lines += [f"Also rescored {summary['calibration_records']} stored calibration records without recomputing physical metrics.", '',
            '| Proposal rank | Archived branch / generation | Original rank | Original score | Proposal score | Quality-feasible completed |',
            '|---:|---|---:|---:|---:|---|']
        for row in summary['calibration_top10']:
            lines.append(f"| {row['proposal_rank']} | {row['branch']} / {row['generation']} | {row['original_rank']} | {row['fitness']:.6f} | {row['proposal_fitness']:.6f} | {row['completed_quality_feasible']} |")
    else:
        lines.append('No stored calibration input was available in this snapshot.')
    lines += ['', '## Limits and reproduction', '',
        '- These are counterfactual rankings of populations generated by V1. Different fitness would change future parents, species allocations and populations; no learning or final-performance improvement is demonstrated.',
        '- Only the archived distinct-genotype top-three candidates were validated. Unsaved alternatives cannot acquire validation outcomes through rescoring. The CSV lacks genotype identities for all genomes, so reported top-three genome ranks are not reconstructed distinct-genotype validation selections. Same-generation archival coverage is conservative; clones or prior-generation cache matches are not inferred.',
        '- Training quality-feasible means completed with V=0 on that six-course batch, not passing the 60-primary/20-retention final validation gates. Changing batch difficulty also prevents interpreting raw scores across generations as a fixed-suite learning curve.',
        '- Calibration aggregates 60 courses, while breeding aggregates six. Absolute bases/bands differ (`600` versus `60`); never compare their numeric fitness directly or treat the calibration ranking as re-evolution evidence.',
        '- Small excess can be rewarded for speed; substantial violation or missing-window penalties may still produce attractive counterfactual scores. Existing final gates are essential. No coefficient or threshold has been tuned by this diagnostic.', '',
        f"Exact captured input bytes, hashes and this executable source are preserved in `{snapshot}`. The root summary includes their SHA-256 values. Refreshing after the run finishes creates a separate immutable snapshot and updates this report.", '',
        'Refresh from current saved logs:', '', '```powershell',
        '& .\\.venv\\Scripts\\python.exe research/diagnose_quality_objective.py', '```', '',
        'Replay the exact snapshot into a separate output directory:', '', '```powershell',
        f'& .\\.venv\\Scripts\\python.exe "{snapshot / "diagnostic.py"}" --replay-snapshot "{snapshot}" --out ".cache/objective-diagnostic-replay"',
        '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/neat-quality-speed-v1')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/neat-quality-speed-v1/objective-diagnostic')
    parser.add_argument('--replay-snapshot', type=Path)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = synthetic_checks()
    if args.replay_snapshot:
        snapshot = args.replay_snapshot.resolve()
        payload, manifest = load_snapshot(snapshot)
    else:
        payload, manifest = capture_inputs(args.run.resolve())
        snapshot = save_snapshot(output, payload, manifest)
    summary, generations, details, calibration = analyze(payload, manifest)
    summary['snapshot_directory'] = str(snapshot)
    summary['synthetic_checks_passed'] = len(checks['checks'])
    outputs = {'summary.json': json.dumps(summary, indent=2) + '\n',
               'synthetic-checks.json': json.dumps(checks, indent=2) + '\n',
               'report.md': report(summary, generations, checks, snapshot)}
    for name, text in outputs.items():
        (output / name).write_text(text, encoding='utf-8')
    write_csv(output / 'per-generation.csv', generations)
    write_csv(output / 'genome-ranks.csv', sorted(details, key=lambda row: (row['generation'], row['proposal_rank'])))
    write_csv(output / 'calibration-ranks.csv', calibration)
    if not args.replay_snapshot:
        for name in (*outputs, 'per-generation.csv', 'genome-ranks.csv', 'calibration-ranks.csv'):
            if (output / name).exists():
                (snapshot / name).write_bytes((output / name).read_bytes())
    print(json.dumps({key: summary[key] for key in ('logged_generation_count', 'last_generation',
        'genome_rows_rescored', 'generations_with_top1_change', 'generations_with_top3_genome_set_change',
        'synthetic_checks_passed', 'snapshot_directory')}, indent=2))


if __name__ == '__main__':
    main()
