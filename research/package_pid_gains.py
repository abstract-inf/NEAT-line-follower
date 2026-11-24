"""Bundle the completed gains-only study and verified historical delivery.

This reads immutable experimental records and makes a new archive. It performs
no evaluation or optimization and refuses to overwrite a previous delivery.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/pid-gains-20261029"
RESEARCH = ROOT / "research"
DESTINATION = RESEARCH / "archive/pid-gains-delivery-20261029.zip"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = DESTINATION.with_suffix(".manifest.json")
    if DESTINATION.exists() or manifest_path.exists():
        raise FileExistsError("Research delivery exists; preserve it unchanged")
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    assert status["complete"]
    preserved = json.loads((RUN / "preserved-records.json").read_text(encoding="utf-8"))
    assert all(digest(ROOT / p) == sha for p, sha in preserved.items())
    old_manifest_path = RESEARCH / "archive/research-delivery-20261029.manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_delivery = RESEARCH / "archive/research-delivery-20261029.zip"
    assert digest(old_delivery) == old_manifest["archive_sha256"]
    selected = [p for p in RUN.rglob("*") if p.is_file()]
    selected += [p for p in (RESEARCH / "artifacts/pid-gains").rglob("*") if p.is_file()]
    selected += [ROOT / "README.md", ROOT / "run_pid_gain_tuning.py", ROOT / "tests/test_pid_gain_tuning.py",
                 RESEARCH / "PID_Gain_Tuning_Protocol.md", RESEARCH / "PID_Gain_Tuning_Results.md",
                 RESEARCH / "PID_Experimental_Results.md", RESEARCH / "Comparative_Analysis.md",
                 RESEARCH / "NEAT_Experimental_Results.md", RESEARCH / "NEAT_Results_Summary.md",
                 RESEARCH / "analyze_pid_gains.py", Path(__file__), old_delivery, old_manifest_path,
                 RESEARCH / "artifacts/pid-gains-training-console.log",
                 RESEARCH / "artifacts/pid-gains-analysis-console.json"]
    records = []
    with zipfile.ZipFile(DESTINATION, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(selected)):
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, data)
            records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    with zipfile.ZipFile(DESTINATION) as archive:
        assert len(archive.namelist()) == len(records)
        for record in records:
            assert hashlib.sha256(archive.read(record["path"])).hexdigest() == record["sha256"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "archive": str(DESTINATION),
                "archive_sha256": digest(DESTINATION), "historical_files_verified_unchanged": len(preserved),
                "study_status": status, "files": records}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(DESTINATION), "files": len(records),
                      "bytes": DESTINATION.stat().st_size, "sha256": manifest["archive_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
