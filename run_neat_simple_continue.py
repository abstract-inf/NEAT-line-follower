"""Continue the matched development pilot to exactly 50 total generations."""
import argparse
from functools import partial
from pathlib import Path
import subprocess
import sys

from neat_simple_study.continuation import (
    ARMS, CONDITION, DEFAULT_RUN, SETTINGS, VERSION, prepare, source_inventory, verify)
from neat_smooth_study.common import read
from run_neat_simple_study import compatible_score, status


def dispatch(args):
    run = args.run.resolve()
    if args.command == 'prepare':
        return prepare(run)
    plan = verify(run)
    if args.command == 'verify':
        print('Matched continuation source, inherited histories and hard 50-total budget verified.')
    elif args.command == 'status':
        status(run, (args.arm,) if args.arm else tuple(ARMS))
    elif args.command == 'train':
        if args.workers not in (6, 7, 8) or (args.arm is None and not args.all):
            raise ValueError('Choose --arm or --all and use six to eight workers.')
        from neat_quality_followup.training import train, training_signature
        for arm in (args.arm,) if args.arm else tuple(ARMS):
            directory = run / 'development' / arm
            expected = training_signature(run, plan, arm, VERSION, condition=CONDITION)
            complete = directory / 'complete.json'
            if complete.exists():
                saved = read(complete)
                if saved['generations'] != SETTINGS['generations'] or saved['signature'] != expected:
                    raise ValueError('Completed arm differs from its frozen 50-total protocol.')
                print(f'{arm}: already complete at 50 total generations; no extra evolution.')
                continue
            if not (directory / 'checkpoints/latest.pkl.gz').exists():
                raise ValueError('Prepare the exact inherited boundary checkpoint before training.')
            train(run, arm, args.workers, resume=True, verify_protocol=verify,
                  fitness_function=partial(compatible_score, smoothness_weight=plan['arms'][arm]['smoothness_weight']),
                  fitness_version=VERSION, condition=CONDITION)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'train', 'status'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--frozen-source', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == 'prepare' or args.frozen_source:
        return dispatch(args)
    source = args.run.resolve() / 'prepared/source'
    if source_inventory(source) != read(args.run / 'prepared/protocol.json')['source_files']:
        raise ValueError('Frozen continuation source changed.')
    command = [sys.executable, str(source / 'run_neat_simple_continue.py'),
               *(argv if argv is not None else sys.argv[1:]), '--run', str(args.run.resolve()), '--frozen-source']
    result = subprocess.run(command, cwd=source)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
