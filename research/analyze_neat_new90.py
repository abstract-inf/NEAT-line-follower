"""Compare the frozen NEAT assessment with saved PID rows on identical tracks.

Reads saved records only. No policy evaluation, evolution or reselection occurs.
Run this after the earlier report generators to restore the latest comparison.
"""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.experiments import environment_record

RUN = ROOT / "runs/neat-frozen-new90-20261102"
PID_RUN = ROOT / "runs/pid-gains-20261029"
OUTPUT = ROOT / "research/artifacts/neat-new90"
MARKER = "<!-- FROZEN-NEAT-NEW90 -->"
LABELS = {"gain-000": "Filtered PID", "gain-160": "PID + corner supervision, original gains",
          "gain-350": "PID + corner supervision, tuned gains", "neat-g109": "Frozen NEAT g109"}
COLORS = ("#3979a9", "#65778a", "#26845c", "#a34c43")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(name: str, records: list[dict]) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def mean(records: list[dict], field: str):
    return float(np.mean([float(r[field]) for r in records])) if records else None


def summarize(records: list[dict]) -> dict:
    successful = [r for r in records if r["completed"] == "True"]
    failures = [r for r in records if r["completed"] != "True"]
    return {"episodes": len(records), "completed": len(successful), "completion_rate": len(successful)/len(records),
            "rmse_m": mean(records, "lateral_rmse_m"), "mae_m": mean(records, "mean_absolute_lateral_error_m"),
            "mean_progress": mean(records, "progress"), "failed_mean_progress": mean(failures, "progress"),
            "successful_time_s": mean(successful, "completion_time_s"),
            "failures": dict(Counter(r["termination"] for r in failures))}


def save_plot(summaries: list[dict]) -> None:
    plt.rcParams["pdf.fonttype"] = 42
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    x = np.arange(3)
    for index, (policy, color) in enumerate(zip(LABELS, COLORS)):
        subset = [next(r for r in summaries if r["policy"] == policy and r["difficulty"] == d)
                  for d in ("beginner", "intermediate", "advanced")]
        bars = axes[0].bar(x+(index-1.5)*.20, [r["completion_rate"]*100 for r in subset], width=.20,
                           color=color, label=LABELS[policy])
        for bar, row in zip(bars, subset):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{row['completed']}/30", ha="center", fontsize=6)
    axes[0].set(xticks=x, xticklabels=["Beginner", "Intermediate", "Advanced"], ylabel="Completion (%)", ylim=(0, 135))
    axes[0].legend(fontsize=6, loc="upper right")
    overall = [next(r for r in summaries if r["policy"] == p and r["difficulty"] == "all") for p in LABELS]
    names = ["Filtered\nPID", "PID supervisor\noriginal gains", "PID supervisor\ntuned gains", "Frozen NEAT\ng109"]
    for axis, key, label in zip(axes[1:], ("rmse_m", "mae_m"), ("Observed lateral RMSE (mm)", "Observed lateral MAE (mm)")):
        values = [r[key]*1000 for r in overall]
        bars = axis.bar(np.arange(4), values, color=COLORS)
        axis.set(xticks=np.arange(4), xticklabels=names, ylabel=label, ylim=(0, max(values)*1.2))
        axis.tick_params(axis="x", labelsize=7)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.2f}", (bar.get_x()+bar.get_width()/2, value), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8)
    figure.suptitle("Same NEW 90 tracks: frozen NEAT and three saved PID policies\nCPU MuJoCo, identical 0.35 m/s physical limits; failures included; no NEAT training or reselection", fontsize=10)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"common-new90.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def append_section(path: Path, content: str) -> None:
    original = path.read_text(encoding="utf-8").split(MARKER)[0].rstrip()
    path.write_text(original+f"\n\n{MARKER}\n"+content.rstrip()+"\n", encoding="utf-8")


def write_documents(metrics: dict) -> None:
    summaries, run, status = metrics["summaries"], metrics["run"], metrics["status"]
    table = "| Frozen policy | Completed / 90 | Rate | RMSE (mm) | MAE (mm) | Mean progress |\n|---|---:|---:|---:|---:|---:|\n"
    difficulty = "| Frozen policy | Beginner | Intermediate | Advanced |\n|---|---:|---:|---:|\n"
    for policy, label in LABELS.items():
        result = next(r for r in summaries if r["policy"] == policy and r["difficulty"] == "all")
        table += f"| {label} | {result['completed']}/90 | {result['completion_rate']*100:.2f}% | {result['rmse_m']*1000:.3f} | {result['mae_m']*1000:.3f} | {result['mean_progress']:.4f} |\n"
        counts = [next(r['completed'] for r in summaries if r['policy']==policy and r['difficulty']==d) for d in ("beginner", "intermediate", "advanced")]
        difficulty += f"| {label} | {counts[0]}/30 | {counts[1]}/30 | {counts[2]}/30 |\n"
    neat = next(r for r in summaries if r["policy"] == "neat-g109" and r["difficulty"] == "all")
    image = f"![Matched new-cohort assessment](<{(OUTPUT/'common-new90.png').as_posix()}>)"
    report = f"""# Frozen NEAT evaluation on the new PID test cohort

Evaluation and documentation: 2 November 2026. This is fixed-policy inference,
not a new NEAT training phase. The already validation-selected generation-109
model was specified before this assessment; no other genome was tried or
substituted after seeing the results.

## Exact model and protocol

Model: runs/28-10-2026-hard/best.pkl, also preserved as frozen-model.pkl in
runs/neat-frozen-new90-20261102. SHA256:
**{run['model_sha256']}**. Generation 109, evolution seed 0. The feedforward
15-binary-input/two-output controller, topology, weights, activations, motor
mapping and sensing are unchanged. No output rescaling, memory, recurrence,
training, controller tuning or model reselection occurred.

The new cohort is exactly runs/pid-gains-20261029/suite/test: 90 tracks, 30 per
difficulty. Manifest SHA256: **{run['suite_sha256']}**. Every policy row uses the
same individual geometry hashes. Robot and episode settings are copied from the
PID run and verified equal to the frozen model's saved metadata: 0.35 m/s wheel
target ceiling, 0.03 N m torque limits, 50 ms actuator filter, unchanged geometry,
600/60 Hz physics/control, and the same length-dependent 45–120 s episode limits,
gates, corridor and failure rules. CPU MuJoCo is the common assessment backend;
historical GPU validation is separate. The model's training-suite hash differs
from this new evaluation suite by design.

Four persistent CPU workers evaluated one episode per test route. It took
{status['wall_seconds']:.3f} s including setup, geometry verification, source
snapshot and integrity checks. No real-time pacing was used. Every attempted
episode is saved, including failures. Predetermined indices 0, 1, 2, 31 and 62
have full sensor/action/physics telemetry; traces are from those same episodes,
not additional independent trials. The original PID rows are reused unchanged
from their completed frozen assessment, rather than reselected or retrained.

## Common new-cohort results

{table}

{difficulty}

{image}

RMSE and MAE are macro-averages of per-episode geometric lateral errors over
observed duration, including failures. MAE is measured directly. Early failure
can reduce observed exposure, so a lower error alone does not establish a better
controller. Completion/progress and failures must be read alongside error.

Frozen NEAT failed {90-neat['completed']} routes, with reasons
`{json.dumps(neat['failures'], sort_keys=True)}`. On the {metrics['zigzag']['episodes']}
routes tagged as containing zigzags, it completed {metrics['zigzag']['completed']}.
Membership does not establish that a failure occurred at the zigzag; some
episodes can terminate before reaching that feature.

## Paired comparison and timing

The paired-comparison.csv artifact counts both-complete, PID-only, NEAT-only
and neither-complete outcomes on identical track hashes. It also reports time
only on the common-success subset for each PID/NEAT pair. Overall successful
time averages refer to different subsets and are not a fair direct speed ranking.
No p-value, statistical significance or algorithm-family superiority is claimed.

## Interpretation and relation to earlier results

The original common cohort remains a separate 90-track benchmark: the frozen
NEAT policy completed 35/90 there. Its {neat['completed']}/90 result here comes from
different routes, not training improvement. The new cohort has no geometry-hash
overlap with the original cohort. All policies have now been assessed on the new
cohort, resolving the previous missing NEAT cell for the gain-tuning comparison.

This is a descriptive comparison of these fixed implementations. The assisted
PID has explicit corner logic and temporal state. The selected NEAT model is
memoryless and its saved sigmoid outputs restrict reverse manoeuvres. Historical
training protocols and compute budgets differ, and only one NEAT lineage exists.
These results do not show that NEAT in general cannot match PID. They concern
new routes from this generator, not faults, disturbances or hardware transfer.

The new tracks were unseen by this frozen NEAT policy, but their PID outcomes
were already known when this assessment was requested. The NEAT artifact was
fixed before either test; it was not selected on these outcomes. Future tuning
must treat both evaluated cohorts as known and use a newly prespecified untouched
cohort for a fresh final claim.

## Reproducibility and preservation

Raw NEAT rows: runs/neat-frozen-new90-20261102/test-episodes.csv. Original PID
rows: runs/pid-gains-20261029/test-episodes.csv. Evaluation records include the
model copy, run.json, status.json, source.zip, tracks/test, telemetry/ and
preserved-records.json. All {status['preserved_files_unchanged']} preserved historical
files and the original model remain byte-identical. Shared package source SHA256:
**{run['environment']['source_sha256']}**. The run records Python, dependencies,
hardware, worker count, script checksum and exact metadata.

Run `.venv\\Scripts\\python.exe evaluate_frozen_neat.py --output runs/neat-new90-replay`
for a fixed-policy reproduction in a new directory; it is not independent
training evidence. Regenerate saved-data tables/plots with
`.venv\\Scripts\\python.exe research/analyze_neat_new90.py`, after the earlier
report generators. Combined 360-row analysis data, summary tables, failures and
paired comparisons are under research/artifacts/neat-new90. Raw inputs are not
overwritten. No manuscript rewrite is started by these research records.
"""
    (ROOT / "research/NEAT_New_Cohort_Evaluation.md").write_text(report, encoding="utf-8")
    brief = "## Subsequent frozen NEAT assessment on the new cohort — 2 November 2026\n\n"+table+"\n"+difficulty+"\n"
    brief += "All rows use identical new-cohort track hashes, CPU physics and physical/episode limits. NEAT g109 is unchanged; this adds inference results, not training or reselection. The original 90-track comparison remains separate. See [NEAT_New_Cohort_Evaluation.md](NEAT_New_Cohort_Evaluation.md) for model/suite hashes, failures, paired comparisons and limitations.\n"
    append_section(ROOT / "research/Comparative_Analysis.md", brief)
    append_section(ROOT / "research/NEAT_Experimental_Results.md", brief)
    append_section(ROOT / "research/NEAT_Results_Summary.md",
                   f"**New matched cohort:** frozen g109 completed {neat['completed']}/90; tuned supervised PID 84/90. "
                   "Inference only; see [evaluation](NEAT_New_Cohort_Evaluation.md) (2 November 2026).\n")
    append_section(ROOT / "README.md", "## Frozen NEAT on the new PID cohort\n\n"
                   "The unchanged generation-109 model has now been assessed on the same new 90 tracks as the gain-tuned PID. "
                   "See [matched results](research/NEAT_New_Cohort_Evaluation.md). "
                   "`evaluate_frozen_neat.py` performs evaluation only; existing outputs are protected. "
                   "Run `research/analyze_neat_new90.py` last when rebuilding research documents.\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run, status = read_json(RUN / "run.json"), read_json(RUN / "status.json")
    assert status["complete"] and run["evaluation_only"] and not status["neat_training_started"]
    assert environment_record()["source_sha256"] == run["environment"]["source_sha256"]
    assert digest(ROOT / "evaluate_frozen_neat.py") == run["runner_sha256"]
    assert digest(RUN / "frozen-model.pkl") == digest(Path(run["model"])) == run["model_sha256"]
    assert digest(PID_RUN / "test-episodes.csv") == run["reference_pid_csv_sha256"]
    assert digest(PID_RUN / "suite/suite.json") == run["suite_sha256"]
    assert read_json(PID_RUN / "run.json")["robot"] == run["robot"]
    assert read_json(PID_RUN / "run.json")["episode"] == run["episode"]
    preserved = read_json(RUN / "preserved-records.json")
    assert all(digest(ROOT / p) == sha for p, sha in preserved.items())
    neat, pid = read_csv(RUN / "test-episodes.csv"), read_csv(PID_RUN / "test-episodes.csv")
    records = [{**r, "source_csv": "runs/neat-frozen-new90-20261102/test-episodes.csv"} for r in neat]
    records += [{**r, "source_csv": "runs/pid-gains-20261029/test-episodes.csv"} for r in pid]
    assert len(records) == 360 and not any(r["error"] for r in records)
    assert all(r["backend"] == "cpu" and float(r["evaluation_speed_limit_m_s"]) == .35 for r in records)
    assert len({(r["policy_id"], r["track_hash"]) for r in records}) == 360
    expected = {r["geometry_hash"] for r in read_json(PID_RUN / "suite/suite.json")["splits"]["test"]}
    assert len(expected) == 90
    original = read_csv(ROOT / "runs/pid-research-20261029/test-episodes.csv")
    assert not expected & {r["track_hash"] for r in original}
    summaries = []
    for policy in LABELS:
        subset = [r for r in records if r["policy_id"] == policy]
        assert len(subset) == 90 and {r["track_hash"] for r in subset} == expected
        for level in ("all", "beginner", "intermediate", "advanced"):
            sample = subset if level == "all" else [r for r in subset if r["difficulty"] == level]
            summaries.append({"policy": policy, "label": LABELS[policy], "difficulty": level, **summarize(sample)})
    neat_by_hash = {r["track_hash"]: r for r in neat}
    paired = []
    for policy in list(LABELS)[:-1]:
        sample = [r for r in pid if r["policy_id"] == policy]
        assert all(r["time_limit_s"] == neat_by_hash[r["track_hash"]]["time_limit_s"]
                   and r["route_length_m"] == neat_by_hash[r["track_hash"]]["route_length_m"] for r in sample)
        cells = Counter((r["completed"] == "True", neat_by_hash[r["track_hash"]]["completed"] == "True") for r in sample)
        common = [r for r in sample if r["completed"] == neat_by_hash[r["track_hash"]]["completed"] == "True"]
        paired.append({"pid_policy": policy, "both_complete": cells[(True, True)], "pid_only_complete": cells[(True, False)],
                       "neat_only_complete": cells[(False, True)], "neither_complete": cells[(False, False)],
                       "pid_common_time_s": mean(common, "completion_time_s"),
                       "neat_common_time_s": mean([neat_by_hash[r["track_hash"]] for r in common], "completion_time_s")})
    metrics = {"summaries": summaries, "paired": paired, "zigzag": summarize([r for r in neat if "zigzag" in r["features"].split(";")]),
               "run": run, "status": status, "analysis_sha256": digest(Path(__file__))}
    (OUTPUT / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export("combined-episodes.csv", records)
    export("summary.csv", [{**r, "failures": json.dumps(r["failures"], sort_keys=True)} for r in summaries])
    export("paired-comparison.csv", paired)
    export("neat-failures.csv", [r for r in neat if r["completed"] != "True"])
    save_plot(summaries)
    write_documents(metrics)
    print(json.dumps({"overall": [r for r in summaries if r["difficulty"] == "all"], "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
