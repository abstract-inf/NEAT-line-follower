"""User-operated fresh 100-generation NEAT run and frozen paired assessment."""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

from line_follower.config import EpisodeConfig, RobotConfig, verify_geometry
from line_follower.controllers import TEMPORAL_INPUT_COUNT, load_model, load_neat_config
from line_follower.curriculum import CurriculumConfig
from line_follower.experiments import TrainConfig, append_csv, environment_record, train, write_json
from line_follower.pid_experiments import evaluate_candidate
from line_follower.suites import create_suite, load_split, read_suite

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUTPUT = RUNS / "neat-corner-100-20261104"
REFERENCE_NEAT = RUNS / "28-10-2026-hard"
REFERENCE_PID = RUNS / "pid-gains-20261029"
NEAT_CONFIG = ROOT / "line_follower/configs/neat-corner.txt"
SUITE_SEED = 2026110402
VALIDATION_TRACKS = 30
TEST_TRACKS = 90
TRACE_SLOTS = (0, 1, 2, 30, 31, 32)  # Two prespecified traces per difficulty.
SETTINGS = TrainConfig(
    strategy="adaptive", seed=20261104, generations=100, population=96, workers=8,
    backend="cpu", episodes_per_genome=4, track_profile="corner",
    aggregation="mean_cvar", cvar_fraction=.5,
    curriculum=CurriculumConfig(window=4, minimum_generations=6, maximum_generations=101,
                                beginner_threshold=.80, intermediate_threshold=.70, advanced_threshold=.55),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def session_lock(output):
    """An OS-owned lock releases on exit, including when the terminal closes."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "session.lock").open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process owns this run: {output}") from error
        yield


def protected_tracks():
    """Exclude known experimental geometry, including mirrors, without reading results."""
    manifests = list((ROOT / "benchmarks").rglob("suite.json")) + list(RUNS.rglob("suite.json"))
    protected = {entry["geometry_hash"] for path in manifests
                 for entries in read_suite(path)["splits"].values() for entry in entries}
    for path in RUNS.rglob("batch.json"):
        protected.update(read_json(path).get("hashes", []))
    for path in RUNS.rglob("training-manifest.json"):
        protected.update(entry["geometry_hash"] for entry in read_json(path).get("tracks", []))
    return protected


def protocol():
    """Match the frozen reference benchmark's physical and episode parameters."""
    robot = RobotConfig.load(REFERENCE_NEAT / "robot.json")
    episode = EpisodeConfig(**read_json(REFERENCE_PID / "run.json")["episode"])
    lock = REFERENCE_NEAT / "geometry-freeze.json"
    verify_geometry(lock, robot)
    old_model = REFERENCE_NEAT / "best.pkl"
    _, old_metadata = load_model(old_model)
    pid_run = read_json(REFERENCE_PID / "run.json")
    for record in (old_metadata, pid_run):
        if record["robot"] != asdict(robot) or EpisodeConfig(**record["episode"]) != episode:
            raise ValueError("Frozen reference controllers have incompatible physical or episode settings")
    if old_metadata["generation"] != 109 or old_metadata.get("time_limit_override") is not None:
        raise ValueError("Expected the matched generation-109 reference model")
    if digest(old_model) != read_json(RUNS / "pid-research-20261029/selection-freeze.json")["neat_model_sha256"]:
        raise ValueError("Reference NEAT model differs from the original freeze")

    gains = read_json(REFERENCE_PID / "selection-freeze.json")["parameters"]
    pid_policies = {"pid-filtered": gains["gain-000"], "pid-supervised-original": gains["gain-160"],
                    "pid-supervised-tuned": gains["gain-350"]}
    config = load_neat_config(NEAT_CONFIG)
    if config.genome_config.num_inputs != TEMPORAL_INPUT_COUNT or config.genome_config.activation_default != "tanh":
        raise ValueError("The corner protocol expects the 21-input signed controller")
    environment = environment_record()
    plan = {
        "study": "neat-corner-from-scratch-1", "training": asdict(SETTINGS),
        "robot": asdict(robot), "episode": asdict(episode), "suite_seed": SUITE_SEED,
        "validation_tracks": VALIDATION_TRACKS, "test_tracks": TEST_TRACKS, "trace_slots": TRACE_SLOTS,
        "controller": "mirrored-sensor-history-v1",
        "selection": "Validation completion rate, then mean/CVaR validation score",
        "comparison": "Freeze the selected NEAT model and assess all five controllers on the fresh test split",
        "reference_neat_sha256": digest(old_model), "reference_pid": pid_policies,
        "neat_config_sha256": digest(NEAT_CONFIG), "runner_sha256": digest(Path(__file__)),
        "source_sha256": environment["source_sha256"], "dependencies": environment["dependencies"],
    }
    # Normalize tuples as lists to compare a saved protocol directly on resume.
    return json.loads(json.dumps(plan)), robot, episode, lock


def assess(output, suite, plan, model):
    assessment = output / "assessment"
    selection_file = assessment / "selection-freeze.json"
    selected = assessment / "frozen-best.pkl"
    old = REFERENCE_NEAT / "best.pkl"
    _, metadata = load_model(model)
    manifest = read_suite(suite)
    selection = {
        "frozen_before_test_geometry_load": True, "new_neat_sha256": digest(model),
        "old_neat_sha256": plan["reference_neat_sha256"], "new_neat_metadata": metadata,
        "pid": plan["reference_pid"], "suite_sha256": digest(suite),
        "test_hashes": [entry["geometry_hash"] for entry in manifest["splits"]["test"]],
        "backend": "cpu", "trace_slots": list(TRACE_SLOTS),
    }
    selection = json.loads(json.dumps(selection))
    if selection_file.exists():
        if read_json(selection_file) != selection or digest(selected) != selection["new_neat_sha256"]:
            raise ValueError("The selected policy or suite differs from the frozen assessment")
    else:
        assessment.mkdir(exist_ok=True)
        shutil.copyfile(model, selected)
        write_json(selection_file, selection)
    results = assessment / "test-episodes.csv"
    status = assessment / "status.json"
    if results.exists():
        if status.exists():
            if read_json(status)["results_sha256"] != digest(results):
                raise ValueError("Saved final assessment differs from its recorded checksum")
        else:
            with results.open(newline="", encoding="utf-8") as stream:
                existing = list(csv.DictReader(stream))
            if len(existing) != TEST_TRACKS * 5 or any(row["error"] for row in existing):
                raise ValueError("Saved final assessment is incomplete")
            write_json(status, {"complete": True, "episodes": len(existing), "results_sha256": digest(results)})
        print(f"Final assessment already saved: {results}", flush=True)
        return

    tracks = load_split(suite, "test")
    assert len(tracks) == TEST_TRACKS
    policies = {"neat-fresh": {"neat_model": str(selected)}, "neat-g109": {"neat_model": str(old)}}
    policies.update({name: {"parameters": parameters} for name, parameters in plan["reference_pid"].items()})
    paths = [str(suite.parent / entry["path"]) for entry in manifest["splits"]["test"]]
    base = {"robot_json": json.dumps(plan["robot"], sort_keys=True), "episode": plan["episode"],
            "seed": SUITE_SEED, "phase": "test", "output": str(assessment), "trace_slots": TRACE_SLOTS}
    jobs = [(policy, {**base, **parameters, "candidate": policy, "tracks": [path], "slot_offset": slot})
            for slot, path in enumerate(paths) for policy, parameters in policies.items()]
    print(f"Frozen generation {metadata['generation']}. Starting {len(jobs)} paired test episodes.", flush=True)
    started = time.perf_counter()
    records = []
    with mp.get_context("spawn").Pool(SETTINGS.workers) as pool:
        evaluations = pool.imap(evaluate_candidate, [task for _, task in jobs])
        for index, ((policy, task), rows) in enumerate(zip(jobs, evaluations, strict=True), 1):
            row, = rows
            slot = task["slot_offset"]
            if row["track_hash"] != tracks[slot].geometry_hash:
                raise RuntimeError("Unexpected final-test geometry")
            seed = metadata["evolution_seed"] if policy == "neat-fresh" else 0 if policy == "neat-g109" else ""
            row.update(policy_id=policy, candidate=policy, track_slot=slot,
                       evaluation_seed=SUITE_SEED, evolution_seed=seed, tuning_seed="")
            records.append(row)
            if index % 45 == 0:
                print(f"Final assessment {index}/{len(jobs)} episodes", flush=True)
    temporary = results.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    errors = sum(bool(row["error"]) for row in records)
    append_csv(temporary, records, list(records[0]))
    if errors:
        temporary.replace(assessment / "technical-failures.csv")
        raise RuntimeError(f"Saved {errors} technical failures. Final assessment is incomplete.")
    if (digest(selected) != selection["new_neat_sha256"] or digest(old) != selection["old_neat_sha256"]
            or digest(suite) != selection["suite_sha256"]):
        raise RuntimeError("A frozen artifact changed during the assessment")
    temporary.replace(results)
    write_json(status, {"complete": True, "episodes": len(records),
               "results_sha256": digest(results), "wall_seconds": time.perf_counter() - started})
    print(f"Final assessment saved: {results}", flush=True)


def run(output, plan, robot, episode, lock, resume):
    training = output / "training"
    suite = output / "suite/suite.json"
    plan_file = output / "protocol.json"
    checkpoint = training / "checkpoints/latest.pkl.gz"
    if resume:
        if not plan_file.is_file() or not suite.is_file():
            raise ValueError("This run has no frozen protocol or suite to resume")
        if read_json(plan_file) != {**plan, "suite_sha256": digest(suite)}:
            raise ValueError("The source, settings, or suite changed since this run was prepared")
        if not checkpoint.is_file() and training.exists() and any(training.iterdir()):
            raise ValueError("No generation checkpoint exists yet. Start with a fresh --output folder.")
    else:
        if any(path.name != "session.lock" for path in output.iterdir()):
            raise FileExistsError(f"Choose a fresh --output or use --resume for {output}")
        print(f"Preparing frozen suite for {output}", flush=True)
        create_suite(suite.parent, seed=SUITE_SEED, validation_count=VALIDATION_TRACKS,
                     test_count=TEST_TRACKS, protected_hashes=protected_tracks())
        write_json(plan_file, {**plan, "suite_sha256": digest(suite)})

    status_file = training / "status.json"
    if not status_file.exists() or not read_json(status_file)["complete"]:
        train(training, suite, SETTINGS, robot, episode, NEAT_CONFIG,
              resume=checkpoint if checkpoint.exists() else None, geometry_lock=lock, live_preview=True)
    status = read_json(status_file)
    with (training / "generations.csv").open(newline="", encoding="utf-8") as stream:
        generations = [int(row["generation"]) for row in csv.DictReader(stream)]
    if not status["complete"] or generations != list(range(SETTINGS.generations)):
        raise RuntimeError("Final assessment requires exactly 100 committed training generations")
    assess(output, suite, plan, training / "best.pkl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true", help="Resume from this run's committed checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print and validate the protocol without starting")
    args = parser.parse_args()
    plan, robot, episode, lock = protocol()
    if args.dry_run:
        print(json.dumps({**plan, "output": str(args.output.resolve()), "training_started": False}, indent=2))
        return
    output = args.output.resolve()
    with session_lock(output):
        run(output, plan, robot, episode, lock, args.resume)


if __name__ == "__main__":
    mp.freeze_support()
    main()
