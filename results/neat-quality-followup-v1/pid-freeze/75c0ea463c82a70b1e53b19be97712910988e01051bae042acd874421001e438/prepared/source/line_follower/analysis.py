"""Data-derived tables and figures, with paired run/track bootstrap intervals.

No telemetry row or repeated deterministic replay is an independent trial here.
Intervals describe the recorded evolution runs and generated track sample; they
do not establish robustness to faults that the experiment never introduced.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from glob import glob
import hashlib
import itertools
from pathlib import Path

import numpy as np

from .experiments import append_csv, write_json


def _number(row: dict, key: str) -> float:
    return float(row[key])


def _completed(row: dict) -> float:
    return float(str(row["completed"]).lower() in {"true", "1"})


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([_number(row, key) for row in rows]))


def _summaries(rows: list[dict], category: str = "difficulty") -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        values = [row[category]] if category != "features" else row["features"].split(";")
        for value in ["all"] + values:
            key = (row["condition"], row["policy_id"], row["evolution_seed"], row["split"],
                   row["evaluation_speed_limit_m_s"], value)
            groups[key].append(row)
    summaries = []
    for key, episodes in sorted(groups.items()):
        completed = [row for row in episodes if _completed(row)]
        summaries.append(dict(zip(["condition", "policy_id", "evolution_seed", "split", "speed_limit_m_s", category], key)) |
                         {"episodes": len(episodes), "completed": len(completed), "failed": len(episodes) - len(completed),
                          "completion_rate": len(completed) / len(episodes), "mean_progress": _mean(episodes, "progress"),
                          "observed_lateral_rmse_mean_m": _mean(episodes, "lateral_rmse_m"),
                          "mean_completion_time_success_s": _mean(completed, "completion_time_s") if completed else "",
                          "mean_elapsed_all_s": _mean(episodes, "elapsed_s"),
                          "mean_forward_speed_m_s": _mean(episodes, "average_forward_speed_m_s"),
                          "line_loss_events": sum(int(row["line_loss_events"]) for row in episodes),
                          "mean_motor_asymmetry": _mean(episodes, "motor_asymmetry")})
    return summaries


def paired_bootstrap(left: np.ndarray, right: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    """Resample paired evolution-run rows and shared track columns together."""
    if left.shape != right.shape or left.ndim != 2 or not left.size:
        raise ValueError("Paired bootstrap requires equal nonempty run-by-track matrices")
    rng = np.random.default_rng(seed)
    differences = right - left
    estimates = np.empty(samples)
    for index in range(samples):
        runs = rng.integers(0, len(differences), size=len(differences))
        tracks = rng.integers(0, differences.shape[1], size=differences.shape[1])
        estimates[index] = differences[np.ix_(runs, tracks)].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def _paired_matrices(rows: list[dict], left: str, right: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    values = {condition: defaultdict(dict) for condition in (left, right)}
    for row in rows:
        if row["condition"] not in values:
            continue
        run = row["evolution_seed"] or "pid"
        value = _completed(row) if metric == "completion" else _number(row, metric)
        values[row["condition"]][run][row["track_hash"]] = value
    left_runs, right_runs = values[left], values[right]
    if set(left_runs) == {"pid"}:
        seeds = sorted(right_runs)
    elif set(right_runs) == {"pid"}:
        seeds = sorted(left_runs)
    else:
        if set(left_runs) != set(right_runs):
            raise ValueError(f"Unequal evolution-seed sets for {left} and {right}")
        seeds = sorted(left_runs)
    matrices = [[], []]
    expected_tracks = None
    for seed in seeds:
        lvalues = left_runs["pid" if set(left_runs) == {"pid"} else seed]
        rvalues = right_runs["pid" if set(right_runs) == {"pid"} else seed]
        if set(lvalues) != set(rvalues):
            raise ValueError(f"Unpaired held-out track sets for {left} and {right}; finish all attempted evaluations")
        tracks = sorted(lvalues)
        if expected_tracks is not None and tracks != expected_tracks:
            raise ValueError("Every evolution run must face the same held-out tracks")
        expected_tracks = tracks
        matrices[0].append([lvalues[track] for track in tracks])
        matrices[1].append([rvalues[track] for track in tracks])
    return np.array(matrices[0]), np.array(matrices[1])


def _figures(summaries: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    strata = sorted({(row["split"], row["speed_limit_m_s"]) for row in summaries})
    for split, speed in strata:
        records = [row for row in summaries if row["difficulty"] == "all" and row["split"] == split and row["speed_limit_m_s"] == speed]
        conditions = sorted({row["condition"] for row in records})
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for axis, metric, title in zip(axes, ("completion_rate", "mean_progress"), ("Completion rate", "Mean route progress")):
            values = [[float(row[metric]) for row in records if row["condition"] == condition] for condition in conditions]
            means = [np.mean(group) for group in values]
            deviations = [np.std(group, ddof=1) if len(group) > 1 else 0 for group in values]
            axis.bar(np.arange(len(conditions)), means, yerr=deviations, capsize=4, color="#427b9e")
            axis.set_xticks(np.arange(len(conditions)), [condition.replace("neat_", "NEAT\n") for condition in conditions], rotation=15)
            axis.set_ylim(0, 1.08)
            axis.set_ylabel(title)
        figure.suptitle(f"{split}; speed ceiling {speed} m/s\nError bars: SD across evolution runs; deterministic replays excluded", fontsize=10)
        for extension in ("png", "pdf", "svg"):
            figure.savefig(output / f"performance-{split}-{speed}.{extension}", dpi=300)
        plt.close(figure)


def analyze(inputs: list[str], output: str | Path, bootstrap_samples: int = 2000, seed: int = 0) -> Path:
    """Analyze explicit episode CSVs, retaining physical failures in denominators."""
    if bootstrap_samples < 10:
        raise ValueError("Use at least ten bootstrap samples")
    paths = sorted({Path(path).resolve() for pattern in inputs for path in glob(pattern, recursive=True)})
    if not paths:
        raise FileNotFoundError("No episode CSV inputs matched")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output must be empty")
    rows = []
    seen = set()
    repeated = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if int(row["repeat_id"]) != 0:
                    repeated += 1
                    continue
                if row.get("error"):
                    raise ValueError(f"Technical error row in {path}: {row['error']}; resolve before scientific analysis")
                identity = tuple(row[key] for key in ("condition", "policy_id", "split", "evaluation_speed_limit_m_s", "track_hash"))
                if identity in seen:
                    raise ValueError("Duplicate policy/track evaluation; do not count repeated trajectories as independent")
                seen.add(identity)
                rows.append(row)
    if not rows:
        raise ValueError("No primary episodes to analyze")
    output.mkdir(parents=True)
    summaries = _summaries(rows)
    features = _summaries(rows, "features")
    append_csv(output / "summary.csv", summaries, list(summaries[0]))
    append_csv(output / "features.csv", features, list(features[0]))
    failures = defaultdict(int)
    for row in rows:
        failures[(row["condition"], row["split"], row["evaluation_speed_limit_m_s"], row["termination"]) ] += 1
    append_csv(output / "terminations.csv", [dict(condition=c, split=s, speed_limit_m_s=v, termination=t, episodes=n) for (c, s, v, t), n in sorted(failures.items())],
               ["condition", "split", "speed_limit_m_s", "termination", "episodes"])
    comparisons = []
    for split, speed in sorted({(row["split"], row["evaluation_speed_limit_m_s"]) for row in rows if row["split"] in {"validation", "test"}}):
        subset = [row for row in rows if row["split"] == split and row["evaluation_speed_limit_m_s"] == speed]
        for left, right in itertools.combinations(sorted({row["condition"] for row in subset}), 2):
            for metric in ("completion", "progress", "lateral_rmse_m"):
                first, second = _paired_matrices(subset, left, right, metric)
                estimate, low, high = paired_bootstrap(first, second, bootstrap_samples, seed)
                comparisons.append({"split": split, "speed_limit_m_s": speed, "left": left, "right": right, "metric": metric,
                                    "effect_right_minus_left": estimate, "ci95_low": low, "ci95_high": high,
                                    "evolution_run_pairs": first.shape[0], "paired_tracks": first.shape[1]})
    if comparisons:
        append_csv(output / "paired-comparisons.csv", comparisons, list(comparisons[0]))
    latex = [r"\begin{tabular}{llrrrr}", r"Condition & Split & Speed (m/s) & Runs & Completion & Progress \\", r"\hline"]
    for condition, split, speed in sorted({(row["condition"], row["split"], row["speed_limit_m_s"]) for row in summaries}):
        records = [row for row in summaries if row["condition"] == condition and row["split"] == split and row["speed_limit_m_s"] == speed and row["difficulty"] == "all"]
        escaped = condition.replace("_", r"\_")
        escaped_split = split.replace("_", r"\_")
        latex.append(f"{escaped} & {escaped_split} & {speed} & {len(records)} & {np.mean([r['completion_rate'] for r in records]):.3f} & {np.mean([r['mean_progress'] for r in records]):.3f} " + r"\\")
    latex.append(r"\end{tabular}")
    (output / "summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    _figures(summaries, output)
    write_json(output / "analysis.json", {"sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
               "primary_episodes": len(rows), "deterministic_replays_excluded": repeated, "bootstrap_samples": bootstrap_samples,
               "bootstrap_seed": seed, "interval": "95% percentile; paired evolution-run and shared-track resampling",
               "effect_direction": "right minus left", "rmse_scope": "Observed episode duration, including incomplete runs",
               "interpretation": "Intervals are not claims of significance; prespecify confirmatory analysis before final testing"})
    return output / "summary.csv"
