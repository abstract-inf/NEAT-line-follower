"""Assess the already selected NEAT policy on the PID gain-study test cohort.

No population, evolution loop or model selection is created. The shared episode
evaluator and exact saved PID robot/episode settings are reused unchanged.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import zipfile

from line_follower.controllers import load_model
from line_follower.experiments import append_csv, environment_record, snapshot_sources, write_json
from line_follower.pid_experiments import evaluate_candidate
from line_follower.suites import load_split

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL = PROJECT_ROOT / "runs/28-10-2026-hard/best.pkl"
REFERENCE_RUN = PROJECT_ROOT / "runs/pid-gains-20261029"
ORIGINAL_COMPARISON = PROJECT_ROOT / "runs/pid-research-20261029"
OUTPUT_DIRECTORY = "runs/neat-frozen-new90-20261102"
WORKERS = 4
TRACE_SLOTS = (0, 1, 2, 31, 62)  # Same predetermined indices as the PID assessment.


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; prior assessments are immutable")
    reference = read_json(REFERENCE_RUN / "run.json")
    freeze = read_json(REFERENCE_RUN / "selection-freeze.json")
    model_hash = read_json(ORIGINAL_COMPARISON / "selection-freeze.json")["neat_model_sha256"]
    suite = REFERENCE_RUN / "suite/suite.json"
    environment = environment_record()
    if digest(MODEL) != model_hash or digest(suite) != freeze["suite_sha256"]:
        raise ValueError("Frozen model or track manifest differs from the saved comparison")
    if environment["source_sha256"] != freeze["source_sha256"]:
        raise ValueError("Shared simulation/controller source has changed; refuse a mismatched comparison")
    _, metadata = load_model(MODEL)
    if metadata["robot"] != reference["robot"] or metadata["episode"] != reference["episode"]:
        raise ValueError("Frozen model and reference PID physical/episode settings differ")
    if metadata.get("time_limit_override") is not None:
        raise ValueError("A saved timeout override requires an explicitly matched protocol")

    output.mkdir(parents=True)
    started = time.perf_counter()
    preserved = read_json(REFERENCE_RUN / "preserved-records.json")
    preserved.update({p.relative_to(PROJECT_ROOT).as_posix(): digest(p)
                      for p in REFERENCE_RUN.rglob("*") if p.is_file()})
    preserved[MODEL.relative_to(PROJECT_ROOT).as_posix()] = model_hash
    write_json(output / "preserved-records.json", preserved)
    model_copy = output / "frozen-model.pkl"
    model_copy.write_bytes(MODEL.read_bytes())
    runner_hash = digest(Path(__file__))
    write_json(output / "run.json", {
        "study": "frozen-neat-new90-1", "evaluation_only": True,
        "started_utc": datetime.now(timezone.utc).isoformat(), "model": str(MODEL),
        "model_sha256": model_hash, "model_metadata": metadata,
        "suite": str(suite), "suite_sha256": digest(suite),
        "reference_pid_csv_sha256": digest(REFERENCE_RUN / "test-episodes.csv"),
        "robot": reference["robot"], "episode": reference["episode"],
        "environment": environment, "workers": WORKERS, "trace_slots": TRACE_SLOTS,
        "runner_sha256": runner_hash, "selection": "Previously frozen model; no reselection",
        "neat_training_started": False})
    snapshot_sources(output / "source.zip")
    with zipfile.ZipFile(output / "source.zip", "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__), Path(__file__).relative_to(PROJECT_ROOT))

    tracks = load_split(suite, "test")
    if len(tracks) != 90 or len({t.geometry_hash for t in tracks}) != 90:
        raise ValueError("Expected the exact 90 distinct PID test tracks")
    paths = []
    for slot, track in enumerate(tracks):
        path = output / "tracks/test" / f"{slot:03d}.json"
        track.save(path)
        paths.append(str(path.resolve()))
    tasks = [{"neat_model": str(model_copy.resolve()), "robot_json": json.dumps(reference["robot"], sort_keys=True),
              "episode": reference["episode"], "output": str(output.resolve()), "seed": reference["suite_seed"],
              "tracks": [path], "slot_offset": slot, "phase": "test", "trace_slots": TRACE_SLOTS}
             for slot, path in enumerate(paths)]
    all_rows = []
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for index, records in enumerate(pool.imap(evaluate_candidate, tasks), 1):
            for record in records:
                # The shared PID exporter has a tuning_seed column; inference is not tuning.
                record.update(tuning_seed="", evolution_seed=metadata["evolution_seed"],
                              evaluation_seed=reference["suite_seed"])
            append_csv(output / "test-episodes.csv", records, list(records[0]))
            if any(r["error"] for r in records):
                raise RuntimeError("Technical evaluation error saved; assessment is incomplete")
            all_rows.extend(records)
            if index % 15 == 0:
                print(f"Frozen NEAT g{metadata['generation']}: {index}/90 evaluated; "
                      f"{sum(r['completed'] for r in all_rows)} completed; wall {time.perf_counter()-started:.1f}s", flush=True)
    if (digest(MODEL) != model_hash or digest(model_copy) != model_hash
            or environment_record()["source_sha256"] != environment["source_sha256"]
            or digest(Path(__file__)) != runner_hash
            or any(digest(PROJECT_ROOT / p) != sha for p, sha in preserved.items())):
        raise RuntimeError("Frozen source/model/reference records changed during assessment")
    write_json(output / "status.json", {
        "complete": True, "episodes": len(all_rows), "completed": sum(r["completed"] for r in all_rows),
        "wall_seconds": time.perf_counter()-started, "generation": metadata["generation"],
        "model_sha256_unchanged": True, "preserved_files_unchanged": len(preserved),
        "neat_training_started": False})
    return output / "test-episodes.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / OUTPUT_DIRECTORY)
    args = parser.parse_args()
    print(run(args.output.resolve()))


if __name__ == "__main__":
    mp.freeze_support()
    main()
