"""Verify saved research records and create a new, non-overwriting delivery ZIP.

This archival tool performs no simulation, model selection, tuning or training.
The external manifest records every delivered file's SHA256 and archive hash.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.experiments import environment_record

RESEARCH = ROOT / "research"
NEAT_ARCHIVE = RESEARCH / "archive/neat-20261029"
PID_RUN = ROOT / "runs/pid-research-20261029"
DELIVERY = RESEARCH / "archive/research-delivery-20261029.zip"
MANIFEST = DELIVERY.with_suffix(".manifest.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(name: str) -> list[dict]:
    with (PID_RUN / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    if DELIVERY.exists() or MANIFEST.exists():
        raise FileExistsError("Delivery already exists; retain the original archive and manifest")
    raw = read_json(NEAT_ARCHIVE / "manifest.json")
    assert digest(NEAT_ARCHIVE / "raw-experiments-and-source.zip") == raw["archive_sha256"]
    original_csvs = [r for r in raw["files"] if r["path"].endswith(".csv")]
    assert all(digest(ROOT / r["path"]) == r["sha256"] for r in original_csvs)
    checkpoint = NEAT_ARCHIVE / "continuation-final-checkpoint.pkl.gz"
    assert digest(checkpoint) == "f618b4887c69977af3c2a8ba112bd3c95dbcc20a90f528d7e470fcaadd9cda27"
    assert digest(checkpoint) == digest(ROOT / "runs/28-10-2026-hard/checkpoints/latest.pkl.gz")

    freeze = read_json(PID_RUN / "selection-freeze.json")
    assert read_json(PID_RUN / "status.json")["complete"]
    assert freeze["frozen_before_test"]
    assert environment_record()["source_sha256"] == freeze["source_sha256"]
    assert digest(ROOT / "benchmarks/suite/suite.json") == freeze["suite_sha256"]
    assert digest(ROOT / "runs/28-10-2026-hard/best.pkl") == freeze["neat_model_sha256"]
    for candidate, parameters in freeze["parameters"].items():
        canonical = json.dumps(parameters, sort_keys=True).encode()
        assert hashlib.sha256(canonical).hexdigest() == freeze["configuration_hashes"][candidate]
    assert read_json(PID_RUN / "best.json") == freeze["parameters"][freeze["overall"]]
    # Only executable implementation is compared: the README was updated with results.
    with zipfile.ZipFile(PID_RUN / "source.zip") as source:
        executable = [n for n in source.namelist() if n.endswith((".py", ".xml"))]
        assert all(source.read(n) == (ROOT / n).read_bytes() for n in executable)

    counts = {"coarse": 576, "expanded": 180, "refine": 360,
              "validation": 180, "test": 270, "representative": 12}
    episodes = {}
    for phase, expected in counts.items():
        rows = read_rows(f"{phase}-episodes.csv")
        assert len(rows) == expected and not any(row["error"] for row in rows)
        assert all(float(r["mean_absolute_lateral_error_m"]) <= float(r["lateral_rmse_m"])+1e-10 for r in rows)
        episodes[phase] = rows
    assert len(read_rows("candidates.csv")) == 120
    test = episodes["test"]
    conditions = sorted({r["condition"] for r in test})
    assert len(conditions) == 3
    test_sets = [{r["track_hash"] for r in test if r["condition"] == c} for c in conditions]
    assert all(len(s) == 90 and s == test_sets[0] for s in test_sets)
    validation_hashes = {r["track_hash"] for r in episodes["validation"]}
    training_hashes = {r["geometry_hash"] for r in read_json(PID_RUN / "training-manifest.json")["tracks"]}
    assert len(validation_hashes) == 30
    assert not training_hashes & (validation_hashes | test_sets[0])
    assert not validation_hashes & test_sets[0]

    verification = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "original_csvs_unchanged": len(original_csvs),
        "last_completed_neat_generation": 173,
        "neat_checkpoint_sha256": digest(checkpoint),
        "pid_source_sha256": freeze["source_sha256"],
        "frozen_parameters_and_source_match": True,
        "candidate_count": 120, "episode_counts": counts,
        "same_test_tracks_per_policy": 90, "split_isolation_verified": True,
        "technical_episode_errors": 0,
        "prior_test_result": {
            "command": ".venv/Scripts/python.exe -m pytest tests/test_pid.py tests/test_robot.py tests/test_tracks.py -q --disable-warnings --basetemp=.cache/pytest-pid-final",
            "passed": 22, "elapsed_seconds": 15.95,
            "note": "Recorded from the completed test invocation; packaging does not rerun tests."
        }
    }
    (RESEARCH / "artifacts/final-verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")

    # Exclude archive/ during discovery, then add known NEAT archives explicitly.
    # This prevents recursive inclusion of the delivery ZIP itself.
    selected = [p for p in RESEARCH.rglob("*") if p.is_file()
                and "archive" not in p.relative_to(RESEARCH).parts
                and "__pycache__" not in p.parts]
    for folder in (PID_RUN, ROOT / "runs/pid-preflight-20261029", NEAT_ARCHIVE):
        selected.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    selected.extend(ROOT / name for name in ("README.md", "run_pid_optimization.py", "analysis.ipynb",
                                             "line_follower/pid.py", "line_follower/pid_experiments.py", "tests/test_pid.py"))
    records = []
    with zipfile.ZipFile(DELIVERY, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(selected)):
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, data)
            records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    with zipfile.ZipFile(DELIVERY) as archive:
        assert len(archive.namelist()) == len(records)
        for record in records:
            assert hashlib.sha256(archive.read(record["path"])).hexdigest() == record["sha256"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "archive": str(DELIVERY),
                "archive_sha256": digest(DELIVERY), "verification": verification, "files": records}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(DELIVERY), "files": len(records), "bytes": DELIVERY.stat().st_size,
                      "sha256": manifest["archive_sha256"], "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
