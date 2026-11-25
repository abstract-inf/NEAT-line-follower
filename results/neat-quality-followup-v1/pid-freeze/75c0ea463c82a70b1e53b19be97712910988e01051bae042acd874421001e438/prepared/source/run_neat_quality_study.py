"""Calibrate, freeze and run the separate quality-speed development experiment."""
import argparse
from pathlib import Path
import subprocess
import sys

from neat_quality_study.protocol import DEFAULT_RUN, read, source_inventory


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('calibrate', 'prepare', 'verify', 'baseline', 'train', 'status'))
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--frozen-source', action='store_true', help=argparse.SUPPRESS)
    return p


def dispatch(args):
    from neat_quality_study.protocol import prepare, verify
    from neat_quality_study.analysis import calibrate, baseline
    from neat_quality_study.training import train
    run = args.run.resolve()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if args.command == 'calibrate':
        if (run / 'prepared/protocol.json').exists():
            raise ValueError('Calibration is already committed in the prepared protocol; use a new version for changes.')
        return calibrate()
    if args.command == 'prepare':
        return prepare(run)
    verify(run)
    if args.command == 'verify':
        print('Quality-speed source, dependencies, settings and frozen inputs verified.'); return
    if args.command == 'baseline':
        return baseline(run, args.workers)
    if args.command == 'train':
        return train(run, args.workers, args.resume)
    import csv, json
    directory = run / 'development/seed-01'
    path = directory / 'generations.csv'
    if path.exists():
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        print(json.dumps({'completed_generations': len(rows), 'latest': rows[-1] if rows else None}, indent=2))
    if (directory / 'complete.json').exists():
        print(json.dumps(read(directory / 'complete.json'), indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ('calibrate', 'prepare') or args.frozen_source:
        return dispatch(args)
    run = args.run.resolve()
    source = run / 'prepared/source'
    if source_inventory(source) != read(run / 'prepared/protocol.json')['source_files']:
        raise ValueError('Frozen quality-speed source changed.')
    command = [sys.executable, str(source / 'run_neat_quality_study.py'),
               *(argv if argv is not None else sys.argv[1:]), '--run', str(run), '--frozen-source']
    result = subprocess.run(command, cwd=source)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
