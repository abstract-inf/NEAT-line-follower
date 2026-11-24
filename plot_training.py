"""Plot saved training/validation results without training or evaluating policies.

    python plot_training.py --run runs/my-curriculum-run

Uses the recorded GPU/CPU results as saved. Validation is used for model
selection, so these figures must not be described as untouched test performance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from line_follower.controllers import load_model

DIFFICULTIES = ("beginner", "intermediate", "advanced")


def plot_diagnostics(run: Path, output: Path) -> None:
    """Plot saved descriptive aggregates; repeated validation rows are dependent."""
    path = run / "diagnostics.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for difficulty in DIFFICULTIES:
        for role, style in (("training_population", "-"), ("validation_champion", "--")):
            selected = [r for r in rows if r["role"] == role and r["category"] == "difficulty" and r["value"] == difficulty]
            if not selected:
                continue
            x = [int(r["generation"]) for r in selected]
            label = f"{difficulty} {role.replace('_', ' ')}"
            axes[0, 0].plot(x, [float(r["completion_rate"]) for r in selected], style, label=label)
            axes[0, 1].plot(x, [float(r["mean_progress"]) for r in selected], style)
            axes[0, 2].plot(x, [1000*float(r["observed_lateral_rmse_m"]) for r in selected], style)
    for axis, title in zip(axes[0], ("Completion by difficulty", "Mean observed progress", "Observed lateral RMSE (mm)")):
        axis.set_title(title)
        axis.set_xlabel("Generation")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    failures = [r for r in rows if r["role"] == "training_population" and r["category"] == "all"]
    for reason in ("off_route", "stalled", "timeout", "tipped"):
        axes[1, 0].plot([int(r["generation"]) for r in failures],
                        [float(r[f"fail_{reason}"])/int(r["episodes"]) for r in failures], label=reason)
    axes[1, 0].set_title("Training failure proportions")
    axes[1, 0].legend(fontsize=7)
    length_rows = [r for r in rows if r["role"] == "validation_champion" and r["category"] == "length_m"]
    bins = sorted({r["value"] for r in length_rows}, key=lambda v: float(v.split("-")[0]))
    counts = [sum(int(r["episodes"]) for r in length_rows if r["value"] == b) for b in bins]
    axes[1, 1].bar([f"{b}\n(n={n})" for b, n in zip(bins, counts)],
                   [sum(int(r["completed"]) for r in length_rows if r["value"] == b)/n for b, n in zip(bins, counts)])
    axes[1, 1].set_title("Validation completion vs route length (m)")
    stages = run / "curriculum.csv"
    if stages.exists():
        with stages.open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        axes[1, 2].step([int(r["generation"]) for r in history], [int(r["stage"]) for r in history], where="post")
    axes[1, 2].set_title("Training-only curriculum stage")
    figure.suptitle("Descriptive diagnostics; failures included; validation reused for selection, not independent test trials", fontsize=10)
    save_figure(figure, output, "training-diagnostics")
FIGURE_FORMATS = ("png", "pdf", "svg")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_figure(figure, output: Path, name: str) -> None:
    for extension in FIGURE_FORMATS:
        figure.savefig(output / f"{name}.{extension}", dpi=300)
    plt.close(figure)


def plot_training(run: Path, output: Path | None = None) -> Path:
    """Regenerate derived plots; never modify checkpoints or raw result CSVs."""
    output = output or run / "plots"
    generations = read_csv(run / "generations.csv")
    episodes = read_csv(run / "champion-episodes.csv")
    _, metadata = load_model(run / "best.pkl")
    selected = int(metadata["generation"])
    validation = [row for row in episodes if row["role"] == "validation_champion"
                  and int(row["generation"]) == selected]
    if not generations or not validation:
        raise ValueError("No generation results or selected-model validation episodes found")
    if len({row["track_hash"] for row in validation}) != len(validation):
        raise ValueError("Duplicate selected-model track evaluations; check the raw CSV")
    if any(row["error"] for row in validation):
        raise ValueError("Technical evaluation failures must be resolved before plotting performance")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    x = [int(row["generation"]) for row in generations]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    axes[0].plot(x, [float(row["training_mean"]) for row in generations], "o-", label="Population average")
    axes[0].plot(x, [float(row["training_best"]) for row in generations], "o-", label="Population best")
    axes[0].set_ylabel("Aggregated training fitness")
    axes[0].legend()
    axes[1].plot(x, [100*float(row["validation_completion_rate"]) for row in generations], "o-", color="#277e68")
    axes[1].set_ylabel("Validation completion (%)")
    axes[1].set_ylim(0, 100)
    axes[2].plot(x, [float(row["validation_score"]) for row in generations], "o-", color="#81559b")
    axes[2].set_ylabel("Aggregated validation score")
    for axis in axes:
        axis.set_xlabel("Generation (zero-based)")
        if len(x) <= 12:
            axis.set_xticks(x)
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        axis.axvline(selected, color="gray", linestyle="--", linewidth=1)
        axis.grid(alpha=.2)
    figure.suptitle(f"Recorded training run | evolution seed {metadata['evolution_seed']} | selected generation {selected}\n"
                   "Training tracks/difficulty change; fixed validation tracks are reused for selection", fontsize=11)
    save_figure(figure, output, "learning-curves")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(x, [float(r["wall_seconds"])/60 for r in generations], label="Generation wall time")
    for field, label in (("training_eval_wall_seconds", "Training evaluation"),
                         ("validation_eval_wall_seconds", "Validation evaluation"),
                         ("track_generation_wall_seconds", "Track generation")):
        if field in generations[0]:
            axes[0].plot(x, [float(r[field])/60 for r in generations], label=label)
    axes[0].set(xlabel="Generation", ylabel="Wall time (minutes)")
    axes[0].legend(fontsize=7)
    axes[1].plot(x, [int(r["physics_steps"])/1e6 for r in generations])
    axes[1].set(xlabel="Generation", ylabel="Logical physics steps (millions)")
    figure.suptitle("Recorded computational cost; wall time includes scheduling and pauses", fontsize=10)
    save_figure(figure, output, "runtime")

    summaries = []
    for difficulty in DIFFICULTIES:
        rows = [row for row in validation if row["difficulty"] == difficulty]
        if not rows:
            continue
        completed = sum(row["completed"].lower() == "true" for row in rows)
        summaries.append({"difficulty": difficulty, "episodes": len(rows), "completed": completed,
                          "completion_percent": 100*completed/len(rows),
                          "mean_progress_percent": 100*np.mean([float(row["progress"]) for row in rows]),
                          "observed_lateral_rmse_mean_mm": 1000*np.mean([float(row["lateral_rmse_m"]) for row in rows])})
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    labels = [f"{row['difficulty']}\n(n={row['episodes']})" for row in summaries]
    for axis, metric, ylabel in zip(axes,
            ("completion_percent", "mean_progress_percent", "observed_lateral_rmse_mean_mm"),
            ("Track completion (%)", "Mean route progress (%)", "Mean observed lateral RMSE (mm)")):
        bars = axis.bar(labels, [row[metric] for row in summaries], color="#427b9e")
        axis.bar_label(bars, fmt="%.1f", padding=3)
        axis.set_ylabel(ylabel)
        if metric.endswith("percent"):
            axis.set_ylim(0, 110)
        else:
            axis.margins(y=.2)
    completed = sum(row["completed"].lower() == "true" for row in validation)
    figure.suptitle(f"Selected best.pkl: generation {selected} | validation {completed}/{len(validation)} completed\n"
                   "Failures included; error metrics cover observed episode duration; one evolution run", fontsize=11)
    save_figure(figure, output, "selected-policy-validation")
    with (output / "selected-policy-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    selected_generation = next(row for row in generations if int(row["generation"]) == selected)
    report = {"run": str(run.resolve()), "selected_generation": selected,
              "validation_episodes": len(validation), "completed": completed,
              "completion_rate": completed/len(validation),
              "validation_score": float(selected_generation["validation_score"]),
              "termination_counts": dict(Counter(row["termination"] for row in validation)),
              "evaluation_backend": metadata.get("validation_backend", "unknown"),
              "claim_scope": "One evolution run; validation used for model selection, not final test performance",
              "training_started": False, "new_evaluation_performed": False}
    (output / "performance-summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    plot_diagnostics(run, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    print(plot_training(arguments.run, arguments.output))
