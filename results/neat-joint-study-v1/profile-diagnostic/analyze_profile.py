"""Saved-data profile comparison only; no simulator, models, or fitness imports."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean

FIELDS = ('geometry_lateral_rmse_m', 'command_jitter')
GROUPS = ('speed', 'mixed', 'corner')
WEIGHTS = {'speed': 1/3, 'mixed': .5, 'corner': 1/6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(); output = Path(__file__).resolve().parent
    directory = run / 'development/joint'
    inputs, blobs = {}, {}
    def load(path):
        path = path.resolve()
        if path not in blobs:
            blobs[path] = path.read_bytes()
            inputs[str(path)] = hashlib.sha256(blobs[path]).hexdigest()
        return blobs[path]
    def read(path): return json.loads(load(path))
    fitness = list(csv.DictReader(io.StringIO(load(directory/'genome-fitness.csv').decode(), newline='')))
    saved = {(row['generation'], row['genome_id']): row
             for path in sorted((directory/'validation').glob('generation-*.json')) for row in read(path)['candidates']}
    reference = read(run/'prepared/inputs/reference-validation.json')['primary']
    eligible, identities = [], set()
    for row in fitness:
        key = int(row['generation']), int(row['genome_id'])
        if row['joint_feasible'] == 'True' and key in saved and saved[key]['genotype_sha256'] not in identities:
            candidate = saved[key]
            cache = read(directory/'validation-cache'/f"{candidate['genotype_sha256']}.json")
            eligible.append((key, cache, sum(item['completed'] for item in cache['primary']) == 60))
            identities.add(candidate['genotype_sha256'])
    eligible.sort(key=lambda item: (not item[2], item[0]))
    selected = [(49, 4739), *(item[0] for item in eligible[:3] if item[0] != (49, 4739))]
    cases = []
    def summarize(rows, refs, groups, weights=None):
        if [row['track_hash'] for row in rows] != [row['track_hash'] for row in refs]:
            by_hash = {row['track_hash']: row for row in refs}
            refs = [by_hash[row['track_hash']] for row in rows]
        result = {'courses': len(rows), 'completed': sum(row['completed'] for row in rows), 'groups': {}}
        for group in GROUPS:
            pairs = [(row, baseline) for row, baseline, label in zip(rows, refs, groups) if label == group]
            result['groups'][group] = {'courses': len(pairs), 'completed': sum(row['completed'] for row, _ in pairs),
                **{field: {'candidate': mean(row[field] for row, _ in pairs),
                           'reference': mean(baseline[field] for _, baseline in pairs)} for field in FIELDS}}
            for field in FIELDS:
                metrics = result['groups'][group][field]
                metrics['ratio'] = metrics['candidate']/metrics['reference']
        for field in FIELDS:
            actual_weights = weights or {group: result['groups'][group]['courses']/len(rows) for group in GROUPS}
            candidate_mean = sum(actual_weights[group]*result['groups'][group][field]['candidate'] for group in GROUPS)
            reference_mean = sum(actual_weights[group]*result['groups'][group][field]['reference'] for group in GROUPS)
            result[field] = {'candidate': candidate_mean, 'reference': reference_mean, 'ratio': candidate_mean/reference_mean,
                             'difference': candidate_mean-reference_mean}
        return result
    for generation, genome_id in selected:
        gate = saved[(generation, genome_id)]
        score = next(row for row in fitness if int(row['generation']) == generation and int(row['genome_id']) == genome_id)
        raw = read(directory/'training-raw'/f'generation-{generation:04d}.json')
        rows = next(row['episodes'] for row in raw['genomes'] if row['genome_id'] == genome_id)
        refs = read(run/'prepared/batches'/f'generation-{generation:04d}'/'reference.json')
        hashes = read(run/'prepared/batches'/f'generation-{generation:04d}'/'batch.json')['hashes']
        if [row['track_hash'] for row in rows] != hashes or [row['track_hash'] for row in refs] != hashes:
            raise ValueError('Training slot geometry/order mismatch.')
        cached = read(directory/'validation-cache'/f"{gate['genotype_sha256']}.json")
        labels = ['speed', 'speed', 'mixed', 'mixed', 'corner', 'corner']
        equal = summarize(rows, refs, labels)
        weighted = summarize(rows, refs, labels, WEIGHTS)
        validation = summarize(cached['primary'], reference, [row['group'] for row in cached['primary']])
        cases.append({'generation': generation, 'genome_id': genome_id, 'candidate_rank': gate['candidate_rank'],
            'genotype_sha256': gate['genotype_sha256'], 'training_joint_feasible': score['joint_feasible'] == 'True',
            'training_joint_violation': float(score['joint_violation']), 'validation_failed_gates': gate['reasons'],
            'retention_completed': sum(row['completed'] for row in cached['retention']),
            'training_equal': equal, 'training_validation_weights': weighted, 'validation_actual': validation,
            'weighting_shift_in_candidate_minus_reference': {field: weighted[field]['difference']-equal[field]['difference'] for field in FIELDS},
            'six_training_slots': [{'slot': index, 'group': labels[index], 'track_hash': row['track_hash'],
                'completed': row['completed'], **{field: {'candidate': row[field], 'reference': refs[index][field]} for field in FIELDS}}
                for index, row in enumerate(rows)]})
    record = {'scope': 'Data-only counterfactual group weighting; no evolution, physics, selection or source changes.',
        'training_weights': dict.fromkeys(GROUPS, 1/3), 'validation_weights': WEIGHTS,
        'joint_feasible_training_rows': sum(row['joint_feasible'] == 'True' for row in fitness),
        'saved_validated_distinct_joint_feasible_genotypes_available': len(eligible), 'cases': cases,
        'input_sha256': inputs, 'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'diagnostic.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    focus = cases[0]
    lines = ['# Training/validation profile diagnostic', '',
        'Saved data only. No controller, score, threshold, source or prepared input changed; no physics or evolution ran.', '',
        f"Focus: generation 49, rank {focus['candidate_rank']}, genome 4739, genotype `{focus['genotype_sha256']}`. It completes 60/60 primary and 20/20 retention; only geometric error and jitter fail validation. **It was not joint-feasible on its own training batch** (joint violation {focus['training_joint_violation']:.8f}).", '',
        'Training assigns equal episode weight: two speed, two mixed, two corner courses (1/3 each). Validation uses 20 speed, 30 mixed, 10 corner courses (1/3, 1/2, 1/6). Reweighting below changes neither episodes nor measurements.', '',
        '| Focus profile | Geometric RMSE candidate / ref (mm) | Ratio | Jitter candidate / ref | Ratio |',
        '|---|---:|---:|---:|---:|']
    for key, label in (('training_equal','Actual equal training means'), ('training_validation_weights','Same six courses, validation weights'), ('validation_actual','Actual 60-course validation')):
        profile = focus[key]; error, jitter = (profile[field] for field in FIELDS)
        lines.append(f"| {label} | {error['candidate']*1000:.4f} / {error['reference']*1000:.4f} | {error['ratio']:.5f} | {jitter['candidate']:.8f} / {jitter['reference']:.8f} | {jitter['ratio']:.5f} |")
    lines += ['', '| Focus group | Training RMSE candidate / ref (mm; ratio) | Validation RMSE candidate / ref (mm; ratio) | Training jitter candidate / ref (ratio) | Validation jitter candidate / ref (ratio) |', '|---|---|---|---|---|']
    for group in GROUPS:
        train, valid = focus['training_equal']['groups'][group], focus['validation_actual']['groups'][group]
        cells = []
        for profile, field, scale in ((train,FIELDS[0],1000),(valid,FIELDS[0],1000),(train,FIELDS[1],1),(valid,FIELDS[1],1)):
            metric = profile[field]
            cells.append(f"{metric['candidate']*scale:.8f} / {metric['reference']*scale:.8f} ({metric['ratio']:.5f})")
        lines.append('| '+group+' | '+' | '.join(cells)+' |')
    lines += ['', 'The weighting change in candidate-minus-reference difference equals `(mixed difference - corner difference)/6`; speed weighting is identical. Positive shifts mean that extra corner weight made training look more favorable.', '',
        f"For the focus candidate the RMSE shift is **{focus['weighting_shift_in_candidate_minus_reference'][FIELDS[0]]*1000:+.4f} mm** and jitter shift **{focus['weighting_shift_in_candidate_minus_reference'][FIELDS[1]]:+.8f}**.", '',
        '| Other saved training-joint-feasible candidate | Primary / retention finishes | Equal-training RMSE / jitter ratios | Reweighted-training ratios | Actual-validation ratios |', '|---|---|---|---|---|']
    for case in cases[1:]:
        def ratios(profile): return ' / '.join(f"{case[profile][field]['ratio']:.5f}" for field in FIELDS)
        lines.append(f"| g{case['generation']} r{case['candidate_rank']} key{case['genome_id']} | {case['validation_actual']['completed']}/60 + {case['retention_completed']}/20 | {ratios('training_equal')} | {ratios('training_validation_weights')} | {ratios('validation_actual')} |")
    lines += ['', f"The archive contains {record['joint_feasible_training_rows']} training-joint-feasible rows and {len(eligible)} distinct such genotypes with same-generation saved validation. This note shows up to three, preferring complete primary validations. Exact six-slot values, per-group counts, all selected input hashes and helper hash are in diagnostic.json.", '',
        'Interpretation: equal group weighting can hide group-specific deficits when corner gains offset mixed-course losses. The numerical tables test that mechanism for these saved cases. Reweighting two examples per group cannot predict a 60-course result: geometry, difficulty, reference errors and failure exposure differ as well. Ratios use paired candidate/reference means; they are not averages of per-track ratios. Training floors (5 mm and .0005 jitter) are not applied to these descriptive raw-reference ratios. Any incomplete comparison is observed-exposure only, not whole-lap equivalence. No unsaved alternative receives invented validation evidence.', '',
        'Reproduce from the project root:', '', '```powershell',
        '& .\\.venv\\Scripts\\python.exe results/neat-joint-study-v1/profile-diagnostic/analyze_profile.py --run runs/neat-joint-study-v1', '```', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'focus_training_equal':focus['training_equal'][FIELDS[0]],
        'focus_training_reweighted':focus['training_validation_weights'][FIELDS[0]],
        'focus_validation':focus['validation_actual'][FIELDS[0]], 'cases':len(cases), 'source_files_hashed':len(inputs)},indent=2))


if __name__ == '__main__':
    main()
