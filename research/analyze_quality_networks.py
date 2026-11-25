"""Inspect saved NEAT architecture and deterministic sensor probes without simulation."""
import csv
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.controllers import NEATController, read_model_artifact


def verify_neat_runtime():
    """Record installed source hashes and check the wheel's installation manifest."""
    distribution = importlib.metadata.distribution('neat-python')
    records = {}
    for entry in distribution.files:
        if not str(entry).replace('\\', '/').startswith('neat/') or not str(entry).endswith('.py'):
            continue
        data = distribution.locate_file(entry).read_bytes()
        expected = entry.hash
        matches = (expected is not None and
            base64.urlsafe_b64encode(hashlib.new(expected.mode, data).digest()).decode().rstrip('=') == expected.value)
        records[str(entry)] = {'sha256': hashlib.sha256(data).hexdigest(),
                               'installed_distribution_record_matches': matches}
    result = {'scope': 'Read-only source check against installed distribution RECORD; complements version pinning, not independent package authentication.',
              'package_version': distribution.version, 'python': sys.version, 'files': records,
              'all_recorded_sources_match': bool(records) and all(r['installed_distribution_record_matches'] for r in records.values())}
    path = ROOT / 'results/neat-quality-speed-v1/neat-runtime-source-integrity.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if distribution.version != '0.92' or not result['all_recorded_sources_match']:
        raise ValueError('Installed NEAT source differs from the recorded distribution.')


def inspect(path):
    artifact = read_model_artifact(path)
    genome, config = artifact['genome'], artifact['config']
    controller = NEATController(genome, config)
    active = {node[0] for node in controller.network.node_evals}
    inputs = set(config.genome_config.input_keys)
    edges = [edge.key for edge in genome.connections.values() if edge.enabled]
    reachable = set(inputs)
    while True:
        updated = reachable | {target for source, target in edges if source in reachable}
        if updated == reachable:
            break
        reachable = updated
    probes = {}
    for label, index in (('center', 7), ('left', 5), ('right', 9)):
        observation = np.zeros(15); observation[index] = 1.
        controller.reset()
        action = None
        for _ in range(180):
            action = controller.act(observation, 1 / 60)
        probes[label] = {'left_command': float(action[0]), 'right_command': float(action[1]),
                         'mean_command': float(action.mean()), 'steering': float((action[0] - action[1]) / 2)}
    return {'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'generation': artifact['metadata'].get('generation'),
            'nodes_in_genotype': len(genome.nodes), 'hidden_nodes_in_genotype': len(set(genome.nodes) - set(config.genome_config.output_keys)),
            'enabled_connections': len(edges), 'evaluated_nodes': len(active),
            'drive_evaluated': config.genome_config.output_keys[0] in active,
            'steering_evaluated': config.genome_config.output_keys[1] in active,
            'drive_graph_reachable_from_inputs': config.genome_config.output_keys[0] in reachable,
            'steering_graph_reachable_from_inputs': config.genome_config.output_keys[1] in reachable,
            'synthetic_constant_sensor_probes': probes}


def main():
    verify_neat_runtime()
    run = ROOT / 'runs/neat-quality-speed-v1'
    directory = run / 'development/seed-01'
    with (directory / 'generations.csv').open(newline='', encoding='utf-8') as stream:
        generations = [int(row['generation']) for row in csv.DictReader(stream)]
    paths = [run / 'prepared/inputs/warm-start-g0083.pkl', run / 'prepared/inputs/reference-g0134.pkl']
    for generation in generations:
        saved = json.loads((directory / 'validation' / f'generation-{generation:04d}.json').read_text(encoding='utf-8'))
        paths.extend(directory / row['model_file'] for row in saved['candidates'])
    records = [inspect(path) for path in paths]
    output = ROOT / 'results/neat-quality-speed-v1/training'
    output.mkdir(parents=True, exist_ok=True)
    summary = {'scope': 'Offline architecture and synthetic sensor probes only; commands are not measured chassis speed.',
               'stock_behavior': 'NEAT-Python 0.92 FeedForwardNetwork.create and existing NEATController.act, without altering disconnected nodes.',
               'probe_protocol': 'Reset history, then repeat one active sensor (center 7, left 5, right 9) for 180 control steps at 60 Hz; record final commands. No physics, route or real replay.',
               'logged_generations': len(generations), 'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'records': records}
    (output / 'network-diagnostics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Inspected {len(records)} saved networks without evolution or physical evaluation.')


if __name__ == '__main__':
    main()
