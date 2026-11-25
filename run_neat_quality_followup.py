"""Prepare, verify or explicitly run one frozen quality-speed follow-up arm."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from neat_quality_followup.protocol import DEFAULT_RUN, ARMS, read, source_inventory


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'verify', 'train', 'status'))
    result.add_argument('--run', type=Path, default=DEFAULT_RUN)
    result.add_argument('--arm', choices=tuple(ARMS))
    result.add_argument('--workers', type=int, default=3)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--frozen-source', action='store_true', help=argparse.SUPPRESS)
    return result


def dispatch(args):
    from neat_quality_followup.protocol import prepare, verify
    run = args.run.resolve()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if args.command == 'prepare':
        return prepare(run)
    verify(run)
    if args.command == 'verify':
        print('Follow-up source, inputs, dependencies and arm signatures verified.'); return
    if args.command == 'train':
        if args.arm is None:
            raise ValueError('Choose exactly one --arm smooth|fast; no automatic experiment-wide launch.')
        from neat_quality_followup.training import train
        return train(run, args.arm, args.workers, args.resume)
    for arm in (args.arm,) if args.arm else ARMS:
        directory = run / 'development' / arm
        path = directory / 'generations.csv'
        rows = []
        if path.exists():
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
        print(json.dumps({'arm': arm, 'logged_generations': len(rows), 'latest': rows[-1] if rows else None,
                          'complete': read(directory / 'complete.json') if (directory / 'complete.json').exists() else None}, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'prepare' or args.frozen_source:
        return dispatch(args)
    run = args.run.resolve()
    source = run / 'prepared/source'
    if source_inventory(source) != read(run / 'prepared/protocol.json')['source_files']:
        raise ValueError('Frozen follow-up source changed.')
    command = [sys.executable, str(source / 'run_neat_quality_followup.py'),
               *(argv if argv is not None else sys.argv[1:]), '--run', str(run), '--frozen-source']
    result = subprocess.run(command, cwd=source)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
