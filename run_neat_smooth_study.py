"""Prepare and manually run the isolated smooth-driving study. No implicit evolution."""
import argparse
from pathlib import Path
import subprocess
import sys

from neat_smooth_study.common import DEFAULT_RUN, read, sources, digest


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preflight','prepare','verify','audit','tune-pid','pilot','review-pilot','confirm','assess','plot','record'))
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--branch',choices=('old','smooth'))
    p.add_argument('--seed',type=int,choices=range(1,6))
    p.add_argument('--cap',type=float,choices=(.5,1.,1.5))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--force-pilot',action='store_true',help='Explicitly evolve even if the archive audit already found a qualifying controller')
    p.add_argument('--frozen-source',action='store_true',help=argparse.SUPPRESS)
    return p


def dispatch(args):
    from neat_smooth_study.common import verify
    from neat_smooth_study.workflow import preflight,prepare,audit,review_pilot,confirmation_branch,assess
    run=args.run.resolve()
    if args.workers<1:raise ValueError('workers must be positive')
    if args.command=='preflight':return preflight()
    if args.command=='prepare':return prepare(run)
    plan=verify(run)
    if args.command=='verify':
        print('Verified frozen source, dependencies, settings, parents and all prepared courses.');return
    if args.command=='audit':return audit(run,args.workers)
    if args.command=='tune-pid':
        from neat_smooth_study.tuning import tune
        if args.cap is None:raise ValueError('tune-pid requires --cap 0.5, 1.0 or 1.5')
        return tune(run,args.cap,args.workers)
    if args.command=='pilot':
        from neat_smooth_study.engine import train
        if args.branch is None:raise ValueError('pilot requires --branch old or smooth')
        review=read(run/'audit/complete.json')
        if review['protocol_sha256']!=digest(run/'prepared/protocol.json'):
            raise ValueError('Audit belongs to another protocol')
        if review['qualified_existing'] and not args.force_pilot:
            raise ValueError('The audit already found a qualifying controller. Review it; --force-pilot explicitly overrides this stop.')
        return train(run,'pilot',1,args.branch,args.workers,args.resume)
    if args.command=='review-pilot':return review_pilot(run)
    if args.command=='confirm':
        from neat_smooth_study.engine import train
        if args.seed is None:raise ValueError('confirm requires --seed 1..5')
        return train(run,'confirmation',args.seed,confirmation_branch(run),args.workers,args.resume)
    if args.command=='assess':return assess(run,args.workers)
    if args.command=='plot':
        from neat_smooth_study.reporting import report
        return report(run)
    if args.command=='record':
        from neat_smooth_study.reporting import record
        return record(run,args.cap)


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.command in ('preflight','prepare') or args.frozen_source:return dispatch(args)
    # Freeze legacy imports too: the child process imports only the prepared copy.
    source=args.run.resolve()/'prepared/source'
    protocol=read(args.run.resolve()/'prepared/protocol.json')
    if sources(source)!=protocol['source_files']:raise ValueError('Frozen source changed')
    command=[sys.executable,str(source/'run_neat_smooth_study.py'),*(argv if argv is not None else sys.argv[1:])]
    command.extend(['--run',str(args.run.resolve()),'--frozen-source'])
    result=subprocess.run(command,cwd=source)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
