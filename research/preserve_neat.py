"""Archive pre-PID research records without altering any original experiment."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.experiments import environment_record

ARCHIVE = ROOT / "research/archive/neat-20261029"


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    destination = ARCHIVE / "raw-experiments-and-source.zip"
    selected = []
    for folder in ("runs", "benchmarks", "configs", "docs", "line_follower", "tests"):
        selected.extend(p for p in (ROOT / folder).rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    selected.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in {".py", ".toml", ".md", ".txt", ".pdf"})
    records = []
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(selected)):
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, data)
            records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    with zipfile.ZipFile(destination) as archive:
        for record in records:
            assert hashlib.sha256(archive.read(record["path"])).hexdigest() == record["sha256"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "project": str(ROOT),
                "archive": str(destination), "archive_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "environment_before_pid": environment_record(), "git_commit": None,
                "git_note": "No Git repository available; source archives and SHA256 identify versions",
                "raw_csv_policy": "Original byte content retained; derived tables never overwrite original CSVs",
                "files": records}
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(records), "uncompressed_bytes": sum(r["bytes"] for r in records),
                      "archive_bytes": destination.stat().st_size, "archive_sha256": manifest["archive_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
