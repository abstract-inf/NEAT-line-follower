"""Repeat the frozen fast-NEAT protocol; preview by default, evolve with --start."""

import argparse
from contextlib import nullcontext
import multiprocessing as mp

import run_neat_fast_study as study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, choices=study.SEEDS, default=[2, 3, 4, 5])
    parser.add_argument("--start", action="store_true", help="Start evolution sequentially for the listed seeds")
    parser.add_argument("--resume", action="store_true", help="Resume started seeds; start untouched seeds normally")
    parser.add_argument("--memory-safe", action="store_true", help="Limit track-image caches and periodically recycle CPU workers")
    parser.add_argument("--evaluation-workers", type=int, choices=range(1, 9),
                        help="With --memory-safe, override runtime parallelism only; record the deviation from the eight-worker plan")
    args = parser.parse_args(argv)
    if args.evaluation_workers is not None and not args.memory_safe:
        parser.error("--evaluation-workers requires --memory-safe")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("List each seed only once")
    plan = study.verify(study.OUTPUT / "prepared")
    pending = []
    for seed in args.seeds:
        directory = study.OUTPUT / f"seed-{seed:02d}"
        status = directory / "training/status.json"
        if status.exists() and study.read_json(status).get("complete"):
            study.finalize(directory / "training", study.OUTPUT / "prepared", plan, seed)
            print(f"Seed {seed}: already complete; skipping.", flush=True)
            continue
        started = (directory / "start.json").exists()
        if started and not args.resume:
            parser.error(f"Seed {seed} was already started; add --resume")
        if started and not (directory / "training/checkpoints/latest.pkl.gz").is_file():
            parser.error(f"Seed {seed} has no committed checkpoint; see docs/fast-neat-repeats.md")
        if not started and (directory / "training").exists() and any((directory / "training").iterdir()):
            parser.error(f"Seed {seed} has training files without start.json; inspect the seed directory")
        pending.append((seed, started))
        print(f"Seed {seed}: {'resume' if started else 'start'} from its own frozen low-speed winner; "
              "300 generations, target population 100, 6 training courses + 60 champion validation courses per generation.", flush=True)
    print("Caps: g0-49=0.65, g50-99=0.80, g100-149=1.00, g150-199=1.25, g200-299=1.50 m/s.", flush=True)
    if args.evaluation_workers is not None:
        print(f"Recorded runtime override: {args.evaluation_workers} evaluation workers instead of 8; evolution settings retained.", flush=True)
    if not args.start:
        print("Preview only. Add --start to begin evolution. See docs/fast-neat-repeats.md.", flush=True)
        return
    for seed, started in pending:
        command = ["train", "--seed", str(seed)]
        if started:
            command.append("--resume")
        execution = nullcontext()
        if args.memory_safe:
            from fast_study_memory import memory_safe_execution
            execution = memory_safe_execution(study.OUTPUT / f"seed-{seed:02d}", args.evaluation_workers)
        with execution:
            study.main(command)


if __name__ == "__main__":
    mp.freeze_support()
    main()
