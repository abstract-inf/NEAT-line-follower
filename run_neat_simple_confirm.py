"""Prospective from-scratch confirmation; requires a completed development decision."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from neat_simple_study.confirmation import DEFAULT_RUN, SEEDS, digest, preflight, prepare, read, source_inventory, verify


def dispatch(args):
    run = args.run.resolve()
    if args.command == 'prepare':
        if args.decision is None:
            raise ValueError('Provide the completed pilot decision JSON with --decision.')
        return prepare(run, args.decision)
    plan = verify(run)
    if args.command == 'verify':
        print('Independent-seed protocol, source, inputs and stock NEAT source verified.')
    elif args.command == 'preflight':
        preflight(run, args.workers)
    elif args.command == 'assess':
        from neat_simple_study.confirmation_assessment import assess
        assess(run, args.workers)
    elif args.command == 'plot':
        from neat_simple_study.confirmation_reporting import report
        report(run, args.output)
    elif args.command == 'train':
        if args.seed is None or args.workers not in (6, 7, 8):
            raise ValueError('Choose one --seed 1..5 and --workers 6..8.')
        path = run / 'preflight/instrumentation-invariance.json'
        if (not path.exists() or not read(path)['all_match']
                or read(path)['protocol_sha256'] != digest(run / 'prepared/protocol.json')):
            raise ValueError('Complete the real instrumentation preflight before evolution.')
        from neat_simple_study.confirmation_training import train
        train(run, args.seed, args.workers, args.resume)
    else:
        for seed in (args.seed,) if args.seed else SEEDS:
            directory = run / f'seed-{seed:02d}'
            path = directory / 'generations.csv'
            rows = list(csv.DictReader(path.open(encoding='utf-8', newline=''))) if path.exists() else []
            print(json.dumps({'seed': seed, 'logged_generations': len(rows),
                              'complete': (directory / 'complete.json').exists(),
                              'latest': rows[-1] if rows else None}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'preflight', 'train', 'status', 'assess', 'plot'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--decision', type=Path)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frozen-source', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == 'prepare' or args.frozen_source:
        return dispatch(args)
    source = args.run.resolve() / 'prepared/source'
    if source_inventory(source) != read(args.run / 'prepared/protocol.json')['source_files']:
        raise ValueError('Frozen confirmation source changed.')
    command = [sys.executable, str(source / 'run_neat_simple_confirm.py'),
               *(argv if argv is not None else sys.argv[1:]), '--run', str(args.run.resolve()), '--frozen-source']
    result = subprocess.run(command, cwd=source)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
