"""Read-only NEAT result extraction; raw CSVs are never changed.

Duplicate logical records retain their first occurrence, with every exclusion
and conflicting value recorded separately. Repeated episode rows are not trials.
"""

from collections import Counter
from dataclasses import asdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "research/artifacts/neat"
RUNS = ("28-10-2026", "28-10-2026-hard")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean(path: Path, keys: tuple[str, ...], audit: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    records = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for number, row in enumerate(csv.DictReader(stream), 2):
            key = tuple(row[k] for k in keys)
            if key in records:
                first_number, first = records[key]
                audit.append({"file": path.relative_to(ROOT).as_posix(), "key": json.dumps(key),
                              "retained_line": first_number, "excluded_line": number,
                              "conflicting_fields": ";".join(k for k in row if row[k] != first[k])})
            else:
                records[key] = number, row
    return [r for _, r in records.values()]


def summarize(rows: list[dict]) -> dict:
    successes = [r for r in rows if r["completed"] == "True"]
    failed = [r for r in rows if r["completed"] != "True"]
    mean = lambda records, key: float(np.mean([float(r[key]) for r in records])) if records else None
    return {"episodes": len(rows), "completed": len(successes), "completion_rate": len(successes)/len(rows) if rows else None,
            "mean_lateral_rmse_m": mean(rows, "lateral_rmse_m"),
            "mean_absolute_lateral_error_m": None, "mae_note": "Not recorded in historical episode summaries; cannot infer from RMSE",
            "mean_progress": mean(rows, "progress"), "mean_failed_progress": mean(failed, "progress"),
            "mean_success_completion_s": mean(successes, "completion_time_s"),
            "mean_elapsed_all_s": mean(rows, "elapsed_s"), "failure_distribution": dict(Counter(r["termination"] for r in rows)),
            "mean_line_loss_events": mean(rows, "line_loss_events")}


def plot_run(run: str, generations: list[dict], selected: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    x = [int(r["generation"]) for r in generations]
    axes[0].plot(x, [float(r["training_mean"]) for r in generations], label="Population mean")
    axes[0].plot(x, [float(r["training_best"]) for r in generations], label="Population best")
    axes[0].set_ylabel("Aggregated training fitness")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [float(r["validation_completion_rate"]) for r in generations])
    axes[1].set(ylabel="Validation completion fraction", ylim=(0, 1))
    axes[2].plot(x, [float(r["wall_seconds"])/60 for r in generations])
    axes[2].set_ylabel("Recorded generation wall time (min)")
    for axis in axes:
        axis.axvline(selected, color=".5", ls="--", lw=1)
        axis.set_xlabel("Generation (zero-based)")
        axis.grid(alpha=.2)
    figure.suptitle(f"{run}: separate experimental phase; reused validation; one evolution lineage", fontsize=10)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"{run}-learning.{extension}", dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit, results = [], {}
    inventory = []
    for path in sorted((ROOT / "runs").rglob("run.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        inventory.append({"directory": str(path.parent.relative_to(ROOT)), "settings": record.get("settings"),
                          "source_sha256": record.get("environment", {}).get("source_sha256"),
                          "source_archive_sha256": hashlib.sha256((path.parent / "source.zip").read_bytes()).hexdigest()
                          if (path.parent / "source.zip").exists() else None,
                          "status": json.loads((path.parent / "status.json").read_text()) if (path.parent / "status.json").exists() else None,
                          "recorded_generation_rows": sum(1 for _ in (path.parent / "generations.csv").open())-1
                          if (path.parent / "generations.csv").exists() else 0})
    for name in RUNS:
        run = ROOT / "runs" / name
        config = json.loads((run / "run.json").read_text())
        artifact = pickle.loads((run / "best.pkl").read_bytes())
        selected = artifact["metadata"]["generation"]
        generations = clean(run / "generations.csv", ("generation",), audit)
        champions = clean(run / "champion-episodes.csv", ("generation", "role", "track_hash"), audit)
        population = clean(run / "training-episodes.csv", ("generation", "genome_id", "track_hash"), audit)
        validation = [r for r in champions if int(r["generation"]) == selected and r["role"] == "validation_champion"]
        zigzags = [r for r in population if "zigzag" in r["features"].split(";")]
        corners = [r for r in population if "corner" in r["features"].split(";")]
        with gzip.open(run / "checkpoints/latest.pkl.gz", "rb") as stream:
            checkpoint = pickle.load(stream)
        results[name] = {"configuration": config, "generation_count": len(generations),
            "generation_range": [int(generations[0]["generation"]), int(generations[-1]["generation"])],
            "next_checkpoint_generation": checkpoint["population"].generation,
            "checkpoint_population_size": len(checkpoint["population"].population),
            "selected_generation": selected, "selected_genome": artifact["genome"].key,
            "model_sha256": hashlib.sha256((run / "best.pkl").read_bytes()).hexdigest(),
            "validation": {d: summarize(validation if d == "all" else [r for r in validation if r["difficulty"] == d])
                           for d in ("all", "beginner", "intermediate", "advanced")},
            "validation_zigzag": summarize([r for r in validation if "zigzag" in r["features"].split(";")]),
            "population_zigzag": summarize(zigzags), "population_corner": summarize(corners),
            "zigzag_unique_tracks": len({r["track_hash"] for r in zigzags}),
            "recorded_wall_s": sum(float(r["wall_seconds"]) for r in generations),
            "logical_physics_steps": sum(int(r["physics_steps"]) for r in generations),
            "training_mean_range": [min(float(r["training_mean"]) for r in generations), max(float(r["training_mean"]) for r in generations)],
            "training_best_range": [min(float(r["training_best"]) for r in generations), max(float(r["training_best"]) for r in generations)],
            "initial_generation": generations[0], "final_generation": generations[-1],
            "selected_generation_metrics": next(r for r in generations if int(r["generation"]) == selected)}
        write_csv(OUTPUT / name / "generations-clean.csv", generations)
        write_csv(OUTPUT / name / "champion-episodes-clean.csv", champions)
        plot_run(name, generations, selected)
    supporting_keys = {"episode-fitness.csv": ("generation", "episode_slot", "track_hash"),
                       "diagnostics.csv": ("generation", "role", "category", "value"),
                       "gpu-profile.csv": ("generation", "phase", "batch"),
                       "curriculum.csv": ("generation",)}
    for name in RUNS:
        for filename, keys in supporting_keys.items():
            clean(ROOT / "runs" / name / filename, keys, audit)
    write_csv(OUTPUT / "duplicate-audit.csv", audit)
    (OUTPUT / "summary.json").write_text(json.dumps({"runs": results, "inventory": inventory,
        "duplicate_count_by_file": dict(Counter(r["file"] for r in audit))}, indent=2), encoding="utf-8")
    print(json.dumps({name: {k:v for k,v in result.items() if k in {"generation_count", "selected_generation", "recorded_wall_s", "zigzag_unique_tracks"}}
                      for name, result in results.items()}, indent=2))
    print("Duplicate rows recorded:", len(audit))


if __name__ == "__main__":
    main()
