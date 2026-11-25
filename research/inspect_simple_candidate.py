"""Inspect the exploratory g27 candidate from sealed development episodes only."""
import hashlib
import json
from pathlib import Path
from statistics import mean

from research.report_simple_pilot import analyze, encoded, paired, replay, summary


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/neat-simple-pilot-50-v1'


def main():
    report_bytes = (OUTPUT / 'report.json').read_bytes()
    address = json.loads(report_bytes)['snapshot_sha256']
    snapshot = OUTPUT / 'snapshots' / address
    blobs, _ = replay(snapshot)
    report = analyze(blobs, snapshot)
    if encoded(report) != report_bytes or not report['complete']:
        raise ValueError('A completed, reproducible development report is required.')
    candidate = next(row for row in report['arms']['lambda-1']['candidates']
                     if row['generation'] == 27 and row['rank'] == 0)
    identity = candidate['genotype_sha256']
    cache = json.loads(blobs['lambda-1/cache/' + identity])
    rows = {row['track_hash']: row for row in cache['primary']}
    prototype = {row['track_hash']: row for row in json.loads(blobs['prototype'])['primary']}
    pid = {row['track_hash']: row for row in json.loads(blobs['pid'])['primary']}
    groups = {}
    for group in ('mixed', 'speed', 'corner'):
        current = {key: row for key, row in rows.items() if row['group'] == group}
        reference = {key: prototype[key] for key in current}
        groups[group] = {'candidate': summary(current), 'prototype': summary(reference),
                         'paired_vs_prototype': paired(current, reference),
                         'paired_vs_pid': paired(current, {key: pid[key] for key in current})}
    differences = sorted(rows, key=lambda key: rows[key]['geometry_lateral_rmse_m'] -
                         prototype[key]['geometry_lateral_rmse_m'], reverse=True)
    result = {'scope': 'Post-hoc inspection of a non-selected related candidate on consulted data; no new simulation.',
              'snapshot_sha256': address, 'report_sha256': hashlib.sha256(report_bytes).hexdigest(),
              'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'new_simulation_episodes': 0, 'selected_by_declared_rule': False,
              'model_file': 'runs/neat-simple-pilot-50-v1/development/lambda-1/' + candidate['model_file'].replace('\\', '/'),
              'model_sha256': hashlib.sha256(blobs['lambda-1/model/27-0']).hexdigest(),
              'candidate': candidate, 'groups': groups,
              'worst_error_increases': [{'track_hash': key, 'track_name': rows[key]['track_name'],
                  'group': rows[key]['group'], 'candidate_rmse_m': rows[key]['geometry_lateral_rmse_m'],
                  'prototype_rmse_m': prototype[key]['geometry_lateral_rmse_m']} for key in differences[:5]]}
    destination = OUTPUT / 'candidate-g27'
    destination.mkdir(exist_ok=True)
    (destination / 'inspection.json').write_bytes(encoded(result))
    lines = ['# Exploratory generation-27 inspection', '', result['scope'], '',
             '| Group | Courses | Candidate / prototype RMSE mm | Candidate / prototype time s |',
             '|---|---:|---:|---:|']
    for group, record in groups.items():
        pair = record['paired_vs_prototype']
        error, elapsed = pair['geometry_lateral_rmse_m'], pair['completion_time_s']
        lines.append(f'| {group} | {pair["shared_successes"]} | {error["left"]*1000:.3f} / {error["right"]*1000:.3f} | '
                     f'{elapsed["left"]:.4f} / {elapsed["right"]:.4f} |')
    pair = candidate['paired_vs_prototype']
    error, elapsed = pair['geometry_lateral_rmse_m'], pair['completion_time_s']
    lines += ['', f'Across 60 shared finishes, time improves only {100*(1-elapsed["left"]/elapsed["right"]):.2f}%, '
              f'while mean geometric RMSE increases {100*(error["left"]/error["right"]-1):.2f}%.', '',
              'The candidate is faster on the speed-course group, slower on mixed courses, and has the same mean '
              'corner completion time. Its higher tracking error appears in all three groups; episode summaries '
              'do not establish where within an individual course the error occurs. Localizing overshoot or corner '
              'cutting would require telemetry or replay. Near-cap straight speed alone is insufficient evidence '
              'that this is a better controller overall.', '',
              'Both models finish 60/60 primary courses. Candidate retention is 20/20. Against PID the candidate '
              'has 59 shared finishes; the full paired metrics, physical straight measurements, five largest '
              'error increases and exact model/source hashes are in inspection.json. This inspection does not '
              'change model selection or replace the accepted prototype.', '',
              'Reproduce without evolution: `& .\\.venv\\Scripts\\python.exe -m research.inspect_simple_candidate`.', '',
              f'Captured pilot snapshot: `{address}`.', '']
    (destination / 'inspection.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(destination), 'new_simulation_episodes': 0,
                      'shared_prototype_finishes': pair['shared_successes']}))


if __name__ == '__main__':
    main()
