"""Prepare an audited checkpoint for observer/logging fixes, without training.

    python prepare_resume.py --run runs/my-curriculum-run

The original checkpoint and source.zip remain unchanged. Physics, controllers,
fitness, sampling, evolution and evaluation code must match the archived source.
Only observer/JSON-writing changes are allowed by default. --gpu-capacity-fix
also permits increasing the two GPU buffer constants, with no stepping changes.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
from pathlib import Path
import pickle
import zipfile

from line_follower.experiments import environment_record, snapshot_sources, write_json

ROOT = Path(__file__).resolve().parent
SOURCE_SUFFIXES = {".py", ".xml", ".json", ".txt"}
DISPLAY_MODULES = {"line_follower/viewer.py", "line_follower/population_view.py"}
IO_FUNCTIONS = {"write_json", "_progress_reporter"}
GPU_CAPACITY_CONSTANTS = {"CONTACTS_PER_WORLD", "CONSTRAINTS_PER_WORLD"}


def without_io_functions(source: bytes) -> str:
    """Compare every other statement, including all scientific experiment code."""
    module = ast.parse(source)
    module.body = [node for node in module.body
                   if not (isinstance(node, ast.FunctionDef) and node.name in IO_FUNCTIONS)]
    return ast.dump(module, include_attributes=False)


def capacity_increase_only(before: bytes, after: bytes) -> bool:
    """Permit larger allocations only; retain every GPU stepping statement."""
    versions, budgets = [], []
    for source in (before, after):
        module = ast.parse(source)
        values = {}
        for node in module.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in GPU_CAPACITY_CONSTANTS):
                if not isinstance(node.value, ast.Constant) or type(node.value.value) is not int:
                    return False
                values[node.targets[0].id] = node.value.value
                node.value = ast.Constant(value=None)
        budgets.append(values)
        versions.append(ast.dump(module, include_attributes=False))
    return (versions[0] == versions[1] and all(
        name in budgets[0] and name in budgets[1] and 0 < budgets[0][name] <= budgets[1][name]
        for name in GPU_CAPACITY_CONSTANTS))


def audit_sources(archive: zipfile.ZipFile, root: Path, *, allow_capacity_increase: bool = False) -> tuple[str, list[dict]]:
    names = {name for name in archive.namelist()
             if name.startswith("line_follower/") and Path(name).suffix in SOURCE_SUFFIXES}
    current = {path.relative_to(root).as_posix() for path in (root / "line_follower").rglob("*")
               if path.suffix in SOURCE_SUFFIXES}
    if names != current:
        raise ValueError("Package files were added or removed; automatic observer-only recovery is refused")
    archived_hash = hashlib.sha256()
    changes = []
    for name in sorted(names, key=lambda value: Path(value).relative_to("line_follower")):
        before, after = archive.read(name), (root / name).read_bytes()
        archived_hash.update(str(Path(name).relative_to("line_follower")).encode())
        archived_hash.update(before)
        if before == after:
            continue
        logging_only = (name == "line_follower/experiments.py"
                        and without_io_functions(before) == without_io_functions(after))
        capacity_only = (allow_capacity_increase and name == "line_follower/gpu_backend.py"
                         and capacity_increase_only(before, after))
        if name not in DISPLAY_MODULES and not logging_only and not capacity_only:
            raise ValueError(f"Scientific/non-observer source changed: {name}; recovery refused")
        changes.append({"file": name, "before_sha256": hashlib.sha256(before).hexdigest(),
                        "after_sha256": hashlib.sha256(after).hexdigest()})
    return archived_hash.hexdigest(), changes


def prepare(run: Path, *, allow_capacity_increase: bool = False) -> Path:
    """Create a derived checkpoint plus complete source-transition provenance."""
    run = run.resolve()
    original = run / "checkpoints/latest.pkl.gz"
    with gzip.open(original, "rb") as stream:
        saved = pickle.load(stream)
    with zipfile.ZipFile(run / "source.zip") as archive:
        archived_hash, changes = audit_sources(archive, ROOT, allow_capacity_increase=allow_capacity_increase)
    if archived_hash != saved["source_sha256"]:
        raise ValueError("Original source.zip does not match this checkpoint")
    environment = environment_record()
    if environment["dependencies"] != saved["dependencies"]:
        raise ValueError("Checkpoint dependencies changed; recovery refused")
    current_hash = environment["source_sha256"]
    if current_hash == archived_hash:
        raise ValueError("No source transition is needed; resume the original checkpoint")
    transition = run / "source-transitions" / current_hash[:12]
    destination = run / "checkpoints" / f"resume-{current_hash[:12]}.pkl.gz"
    for path in (destination, transition.with_suffix(".json"), transition.with_suffix(".zip")):
        if path.exists():
            raise FileExistsError(f"Recovery artifact already exists: {path}")
    transition.parent.mkdir(parents=True, exist_ok=True)
    snapshot_sources(transition.with_suffix(".zip"))
    with zipfile.ZipFile(transition.with_suffix(".zip"), "a") as archive:
        archive.write(Path(__file__), "prepare_resume.py")
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "reason": "Observer/logging fixes" + (" and increased GPU buffer capacity" if allow_capacity_increase else ""),
             "original_checkpoint": str(original),
             "original_checkpoint_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
             "original_source_sha256": archived_hash, "resumed_source_sha256": current_hash,
             "source_archive": str(transition.with_suffix(".zip")), "changes": changes,
             "allowed_io_functions": sorted(IO_FUNCTIONS),
             "allowed_gpu_capacity_increase": allow_capacity_increase,
             "next_generation": saved["population"].generation, "training_started": False}
    saved["source_sha256"] = current_hash
    saved["source_transition"] = str(transition.with_suffix(".json"))
    temporary = destination.with_suffix(".tmp")
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    audit["derived_checkpoint"] = str(destination)
    audit["derived_checkpoint_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write_json(transition.with_suffix(".json"), audit)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu-capacity-fix", action="store_true", help="Also audit increased GPU buffer constants; physics code must remain unchanged")
    args = parser.parse_args()
    print(prepare(args.run, allow_capacity_increase=args.gpu_capacity_fix))
