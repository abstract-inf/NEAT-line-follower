"""Prepare, verify, inspect or explicitly launch the single frozen joint arm."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from neat_joint_study.protocol import DEFAULT_RUN, read, source_inventory


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'verify', 'train', 'status'))
    result.add_argument('--run', type=Path, default=DEFAULT_RUN)
    result.add_argument('--workers', type=int, default=6)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--frozen-source', action='store_true', help=argparse.SUPPRESS)
    return result


def dispatch(args):
    from neat_joint_study.protocol import prepare, verify
    run = args.run.resolve()
    if args.workers < 1:
        raise ValueError('Workers must be positive.')
    if args.command == 'prepare':
        return prepare(run)
    verify(run)
    if args.command == 'verify':
        print('Joint source, inputs, mutation powers and signatures verified.'); return
    if args.command == 'train':
        from neat_joint_study.fitness import VERSION, score_batch
        from neat_quality_followup.training import train
        return train(run, 'joint', workers=args.workers, resume=args.resume,
                     verify_protocol=verify, fitness_function=score_batch, fitness_version=VERSION)
    directory = run / 'development/joint'
    rows = []
    if (directory / 'generations.csv').exists():
        with (directory / 'generations.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
    print(json.dumps({'arm': 'joint', 'logged_generations': len(rows), 'latest': rows[-1] if rows else None,
                      'complete': read(directory / 'complete.json') if (directory / 'complete.json').exists() else None}, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'prepare' or args.frozen_source:
        return dispatch(args)
    run = args.run.resolve()
    source = run / 'prepared/source'
    if source_inventory(source) != read(run / 'prepared/protocol.json')['source_files']:
        raise ValueError('Joint frozen source changed.')
    command = [sys.executable, str(source / 'run_neat_joint_study.py'),
               *(argv if argv is not None else sys.argv[1:]), '--run', str(run), '--frozen-source']
    completed = subprocess.run(command, cwd=source)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
