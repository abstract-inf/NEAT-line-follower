"""Analyze the completed gains-only study; never run or select a controller.

Historical reports retain their original results and receive an explicit
follow-up section. All new tables/figures derive from saved episode CSVs.
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

plt.rcParams["pdf.fonttype"] = 42  # Embed editable TrueType text in research PDFs.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.experiments import environment_record
from line_follower.tracks import TrackMask, TrackSpec

RUN = ROOT / "runs/pid-gains-20261029"
OUTPUT = ROOT / "research/artifacts/pid-gains"
LABELS = {"pid_filtered": "Filtered PID", "pid_corner_supervised": "PID + corner supervision"}
COLORS = {"baseline": "#65778a", "selected": "#26845c"}
FOLLOWUP_MARKER = "<!-- PID-GAINS-FOLLOWUP -->"
REPORT_DATE = "2 November 2026"  # Client date when analysis/documentation was finalized.


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(name: str) -> list[dict]:
    with (RUN / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def export(name: str, records: list[dict]) -> None:
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def mean(records: list[dict], key: str):
    return float(np.mean([float(r[key]) for r in records])) if records else None


def summary(records: list[dict]) -> dict:
    complete = [r for r in records if r["completed"] == "True"]
    failed = [r for r in records if r["completed"] != "True"]
    return {"episodes": len(records), "completed": len(complete), "completion_rate": len(complete)/len(records),
            "rmse_m": mean(records, "lateral_rmse_m"), "mae_m": mean(records, "mean_absolute_lateral_error_m"),
            "mean_progress": mean(records, "progress"), "successful_time_s": mean(complete, "completion_time_s"),
            "failure_distribution": dict(Counter(r["termination"] for r in failed)),
            "line_loss_events": sum(int(r["line_loss_events"]) for r in records)}


def save_figure(figure, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def comparison_figure(summaries: list[dict]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for row, family in enumerate(LABELS):
        all_results = [next(r for r in summaries if r["phase"] == "test" and r["family"] == family
                            and r["role"] == role and r["difficulty"] == "all") for role in COLORS]
        for column, (metric, scale, title) in enumerate((("completion_rate", 100, "Completion (%)"),
                                                       ("rmse_m", 1000, "Observed lateral RMSE (mm)"),
                                                       ("mae_m", 1000, "Observed lateral MAE (mm)"))):
            axis = axes[row, column]
            values = [r[metric]*scale for r in all_results]
            bars = axis.bar(["Original gains", "Selected gains"], values, color=list(COLORS.values()))
            for bar, result, value in zip(bars, all_results, values):
                label = f"{result['completed']}/90" if column == 0 else f"{value:.2f}"
                axis.annotate(label, (bar.get_x()+bar.get_width()/2, value), xytext=(0, 4),
                              textcoords="offset points", ha="center", fontsize=9)
            axis.set(title=f"{LABELS[family]}\n{title}", ylim=(0, 110) if column == 0 else (0, max(values)*1.2))
    figure.suptitle("Gains-only follow-up: paired assessment on 90 NEW frozen tracks\nFiltered PID retained its original gains (same policy); errors include failures", fontsize=11)
    save_figure(figure, "new-test-comparison")


def search_figure() -> None:
    records = rows("candidate-summary.csv")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, phases, title in zip(axes, (("coarse",), ("expanded", "refine"), ("validation",)),
                                   ("15 training cases", "45 training cases", "30 validation routes")):
        for family, color in zip(LABELS, ("#3979a9", "#26845c")):
            subset = [r for r in records if r["phase"] in phases and r["family"] == family]
            axis.scatter([int(r["candidate"].split("-")[1]) for r in subset],
                         [float(r["completion_rate"])*100 for r in subset], label=LABELS[family], s=15, alpha=.7, color=color)
        axis.set(title=title, xlabel="Candidate ID", ylabel="Completion (%)", ylim=(-3, 104))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    figure.suptitle("Only P, I, D varied; stages use different cohorts and must remain separate", fontsize=10)
    save_figure(figure, "gain-search")


def sensitivity_figure(candidates: dict, freeze: dict) -> None:
    """One-axis training checks show sensitivity without changing other gains."""
    coarse = {r["candidate"]: r for r in rows("candidate-summary.csv") if r["phase"] == "coarse"}
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    records = []
    for row, family in enumerate(LABELS):
        baseline_key = freeze["baseline_ids"][family]
        for column, gain in enumerate(("kp", "ki", "kd")):
            selected = [key for key, p in candidates.items() if p["family"] == family
                        and (key == baseline_key or p["origin"].startswith(f"axis-{gain}-"))]
            selected.sort(key=lambda k: float(candidates[k][gain]))
            x = [float(candidates[key][gain]) for key in selected]
            y = [float(coarse[key]["completion_rate"])*100 for key in selected]
            axes[row, column].plot(x, y, "o-", color="#26845c")
            axes[row, column].scatter([float(candidates[baseline_key][gain])],
                                     [float(coarse[baseline_key]["completion_rate"])*100],
                                     color="#a34c43", marker="x", s=70, label="Original gain", zorder=3)
            axes[row, column].set(title=f"{LABELS[family]}\n{gain.upper()} only", xlabel=f"{gain.upper()} (normalized-error units)",
                                  ylabel="15-case training completion (%)", ylim=(-3, 104))
            axes[row, column].grid(alpha=.2)
            for key in selected:
                records.append({"family": family, "gain_changed": gain, "candidate": key,
                                "gain_value": float(candidates[key][gain]), "completed": int(coarse[key]["completed"]),
                                "episodes": 15, "original_gains": key == baseline_key})
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("One-gain perturbations on training cases; other two gains held fixed\nThis sensitivity check is not a held-out performance estimate", fontsize=11)
    save_figure(figure, "gain-sensitivity")
    export("gain-sensitivity.csv", records)


def trajectory_figure(freeze: dict) -> None:
    """Plot saved sensor-controlled paths, including failed original trajectories."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for row, family in enumerate(LABELS):
        for column, slot in enumerate((3, 4)):
            axis = axes[row, column]
            track = TrackSpec.load(RUN / "tracks/training" / f"{slot:03d}.json")
            mask = TrackMask(track)
            half = track.arena_size/2
            axis.imshow(mask.field, extent=(-half-track.start[0], half-track.start[0],
                        -half-track.start[1], half-track.start[1]), cmap="gray", vmin=0, vmax=255)
            for role, linestyle in (("baseline", "--"), ("selected", "-")):
                key = freeze["baseline_ids" if role == "baseline" else "selected"][family]
                with (RUN / "telemetry/representative" / f"{key}-{slot:03d}.csv").open(newline="", encoding="utf-8") as stream:
                    trace = list(csv.DictReader(stream))
                axis.plot([float(r["x_m"])-track.start[0] for r in trace],
                          [float(r["y_m"])-track.start[1] for r in trace], linestyle=linestyle,
                          color=COLORS[role], label=role, linewidth=1.5)
            axis.set(title=f"{LABELS[family]}: {'left' if slot == 3 else 'right'}-first zigzag",
                     xlabel="x from start (m)", ylabel="y from start (m)", xlim=(-.05, 1.0),
                     ylim=(-.05, .9) if slot == 3 else (-.9, .05), aspect="equal")
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
    figure.suptitle("Actual diagnostic training trajectories: original versus selected gains\nThese handcrafted cases are not unseen test successes", fontsize=11)
    save_figure(figure, "diagnostic-trajectories")


def update_section(path: Path, content: str) -> None:
    original = path.read_text(encoding="utf-8").split(FOLLOWUP_MARKER)[0].rstrip()
    # These statements describe the initial study, not a prohibition on a later registered study.
    original = original.replace("No further tuning followed this final assessment.",
                                "No further tuning was part of that initial assessment; the separate follow-up is below.")
    if path.name == "PID_Experimental_Results.md":
        original = original.replace("Scientific research draft, 29 October 2026.",
            "Initial-study research draft, 29 October 2026. The initial results below remain separate from the gains-only follow-up appended at the end.")
    if path.name == "Comparative_Analysis.md":
        original = original.replace("Finalized 29 October 2026.",
            f"Initial comparison finalized 29 October 2026; separate gains-only follow-up documented {REPORT_DATE}.")
    path.write_text(original+f"\n\n{FOLLOWUP_MARKER}\n"+content.rstrip()+"\n", encoding="utf-8")


def write_report(metrics: dict, candidates: dict, freeze: dict, run: dict) -> None:
    summaries, comparisons = metrics["summaries"], metrics["paired"]
    table = "| Family | Gains | Completed / 90 | RMSE (mm) | MAE (mm) | Mean progress |\n|---|---|---:|---:|---:|---:|\n"
    for family in LABELS:
        for role in COLORS:
            result = next(r for r in summaries if r["phase"] == "test" and r["family"] == family and r["role"] == role and r["difficulty"] == "all")
            table += f"| {LABELS[family]} | {role} | {result['completed']}/90 | {result['rmse_m']*1000:.3f} | {result['mae_m']*1000:.3f} | {result['mean_progress']:.4f} |\n"
    gains = "| Family | Gains | P | I | D |\n|---|---|---:|---:|---:|\n"
    for family in LABELS:
        for role, key in (("baseline", freeze["baseline_ids"][family]), ("selected", freeze["selected"][family])):
            c = candidates[key]
            gains += f"| {LABELS[family]} | {role} | {float(c['kp']):.9f} | {float(c['ki']):.9f} | {float(c['kd']):.9f} |\n"
    paired_text = []
    for result in comparisons:
        paired_text.append(f"- {LABELS[result['family']]}: {result['selected_only_complete']} newly completed, "
                          f"{result['baseline_only_complete']} lost completions, {result['both_complete']} completed by both, "
                          f"{result['neither_complete']} by neither. Net change {result['completion_delta']:+d}/90 "
                          f"({result['completion_delta']/90*100:+.2f} percentage points). On {result['both_complete']} common successes, "
                          f"mean time is {result['baseline_common_time_s']:.3f} s original versus "
                          f"{result['selected_common_time_s']:.3f} s selected; mean paired change "
                          f"{result['paired_time_change_s']:+.3f} s.")
    conclusions = []
    for result in comparisons:
        delta = result["completion_delta"]
        if freeze["selected"][result["family"]] == freeze["baseline_ids"][result["family"]]:
            conclusion = "validation retained the original gains; this bounded search did not supply a selected replacement"
        elif delta > 0:
            conclusion = f"selected gains completed {delta} more new-test routes than the original gains"
        elif delta < 0:
            conclusion = f"selected gains completed {-delta} fewer new-test routes; validation selection did not improve test completion"
        else:
            conclusion = "selected gains did not change the total number of completed new-test routes"
        conclusions.append(f"- {LABELS[result['family']]}: {conclusion}. This is an observed cohort result, not a significance claim.")
    selection_table = "| Family | Gains | 45-case training completion | 30-route validation completion |\n|---|---|---:|---:|\n"
    for family in LABELS:
        for role in COLORS:
            training = next(r for r in summaries if r["phase"] == "expanded-training" and r["family"] == family and r["role"] == role and r["difficulty"] == "all")
            validation = next(r for r in summaries if r["phase"] == "validation" and r["family"] == family and r["role"] == role and r["difficulty"] == "all")
            selection_table += f"| {LABELS[family]} | {role} | {training['completed']}/45 | {validation['completed']}/30 |\n"
    difficulty_table = "| Family | Gains | Difficulty | Completed / n |\n|---|---|---|---:|\n"
    legacy_table = "| Family | Gains | Known original cohort completion |\n|---|---|---:|\n"
    for r in summaries:
        if r["phase"] == "test" and r["difficulty"] != "all":
            difficulty_table += f"| {LABELS[r['family']]} | {r['role']} | {r['difficulty']} | {r['completed']}/{r['episodes']} |\n"
        if r["phase"] == "legacy-test" and r["difficulty"] == "all":
            legacy_table += f"| {LABELS[r['family']]} | {r['role']} | {r['completed']}/90 |\n"
    search_counts = {phase: len(rows(f"{phase}-episodes.csv")) for phase in ("coarse", "expanded", "refine", "validation", "test", "legacy-test", "representative")}
    image = lambda name: f"![{name}](<{(OUTPUT / (name+'.png')).as_posix()}>)"
    report = f"""# PID gains-only follow-up results

Experiment conducted 29 October 2026; analysis and documentation finalized
{REPORT_DATE}. This is a separate experiment after
the initial PID/NEAT comparison; its original raw records and archive remain
unchanged. The question is whether **only P, I, D tuning** can improve either
fixed PID variant. No new NEAT training or selection was performed.

## Protocol and implementation boundary

The prespecified protocol is research/PID_Gain_Tuning_Protocol.md. Exactly
{len(candidates)} distinct gain configurations were evaluated. The initial 320
include each baseline, one-gain perturbations and local/wider stratified samples
from random streams 11, 29 and 47. These streams were pooled into one selection;
they are not three independent replicated studies. Twenty-four configurations
were promoted to 45-case training; another 48 were locally refined. Baselines
were always eligible for validation and final selection. The search is bounded,
not an exhaustive optimization or demonstration of an optimum.

Only kp, ki, kd change. All other values in each original selected configuration
are bit-for-bit numerically equal. Physics, sensors, controller/supervisor code,
speed settings, derivative filter, integral bound, recovery and corner settings
are unchanged. Thus this isolates gain changes within each implementation;
supervised PID remains a different architecture from filtered PID.

The 45 training cases comprise nine reused handcrafted training cases and 36
new procedural routes (12 per difficulty). The coarse subset contains those
nine cases and six procedural routes. New validation/test contain 30/90 routes,
balanced by difficulty, generated with suite seed {run['suite_seed']} using the
unchanged version-1 generator distribution. New procedural geometry excludes
known historical routes by canonical hash, including mirrored equivalents.
Handcrafted mirrors intentionally coexist within training. These tests concern
new routes from a familiar generator, not generator-family extrapolation.

Selection is lexicographic completion fraction, mean authenticated progress,
negative observed RMSE, negative successful time. Exact configurations, source,
runner, protocol and suite hashes were frozen at **{freeze['frozen_utc']}** before
test geometry was loaded for evaluation. No reselection or tuning followed
outcomes. The original 90 tracks are only a secondary, already-known benchmark.

Episode allocations: `{json.dumps(search_counts, sort_keys=True)}`. In addition,
{len(rows('test-trace-replays.csv'))} fixed-policy trace replays are labelled
separately, never counted as independent final-test observations. Four persistent
Windows CPU workers ran without real-time pacing. Total wall time, including
track preparation, selection, assessment and trace replays, was
**{metrics['status']['wall_seconds']:.3f} s ({metrics['status']['wall_seconds']/60:.3f} min)**.
There are {sum(search_counts.values())+len(rows('test-trace-replays.csv')):,} recorded
episodes including these explicitly labelled replays; none has a technical
episode error. Valid off-route/stall failures remain in the data.

## Frozen gains

For the complete controller definition and paper-writing reference, see [PID_Controller_Methodology.md](PID_Controller_Methodology.md). It distinguishes derivative filtering from observation processing and documents the added corner supervisor and its information limits.

The filtered reference is the original pid-112 configuration; the supervised
reference is pid-111. The unchanged tracking law is

```text
e = mean(active sensor indices)/7 - 1
turn = P*e + I_gain*integral_state + D_gain*filtered_derivative
wheel_commands = clip([forward + turn, forward - turn], -1, 1)
```

The bounded integral, derivative filter, forward schedule and supervisor are
defined in the initial PID report and line_follower/pid.py. With dimensionless
commands and normalized sensor error, P is dimensionless, I_gain has units 1/s,
and D_gain has units s. These are not the historical unnormalized PID gains.

{gains}

Complete configuration files are in runs/pid-gains-20261029/selected-*.json and
best.json. The overall best is chosen on validation, not this final test.
The filtered selection retained gain-000, so its baseline/selected table rows
refer to the same 90 episodes, not two independent runs. Only three distinct
policies were assessed on each final cohort.

## Primary result: the same 90 new held-out tracks

{chr(10).join(conclusions)}

{table}

{chr(10).join(paired_text)}

Errors are macro-averages of per-episode observed-duration metrics and include
failures. MAE is accumulated directly. Common-success time differences are
paired on identical successfully completed tracks; differing success subsets
must not be compared as evidence of a speed improvement.

{difficulty_table}

{image('new-test-comparison')}

## Search and validation

{selection_table}

Both supervised policies completed 28/30 validation routes. The selected gains
won the prespecified progress tie-break: mean valid progress was
{next(r['mean_progress'] for r in summaries if r['phase']=='validation' and r['family']=='pid_corner_supervised' and r['role']=='baseline' and r['difficulty']=='all'):.6f}
for the original gains and
{next(r['mean_progress'] for r in summaries if r['phase']=='validation' and r['family']=='pid_corner_supervised' and r['role']=='selected' and r['difficulty']=='all'):.6f}
for the selected gains. Test outcomes did not decide this choice. Retaining the
filtered baseline is not proof that its gains are globally optimal: some other
settings improved training completion, but were not chosen by validation.

{image('gain-search')}

{image('gain-sensitivity')}

The one-gain sensitivity figure uses only the deliberately controlled axis
perturbations and original gains on the same 15 training cases. Other two gains
and all non-gain parameters remain fixed in each panel. It measures training-case
sensitivity, not an independent validation or final-test result.

Every attempted configuration, its origin/seed stream, phase and per-episode
metrics are saved. Stage curves are separate because cohorts differ. Validation
decides selection; gains that happen to do better on test are not substituted.
Gain changes may trade completion against error or speed. A remaining failed
route does not prove that its gains caused failure: fixed detection, recovery
and observation limitations were not ablated here.

## Secondary known-cohort replay

{legacy_table}

This is the previously opened cohort from the original study. Its outcomes were
not used in this gain search or selection. Replaying it helps assess continuity,
but cannot restore its status as an untouched confirmatory test. The original
NEAT numbers must not be compared with the new-cohort PID numbers as if the
tracks were identical. NEAT was not evaluated during this gains-only phase.
A subsequent assessment of the unchanged frozen NEAT model on this same new
cohort is now documented in [NEAT_New_Cohort_Evaluation.md](NEAT_New_Cohort_Evaluation.md).

## Representative trajectories

{image('diagnostic-trajectories')}

These are recorded physical trajectories on the reused diagnostic training
zigzags, not final-test examples. Complete 60 Hz traces contain actual sensors,
PID signals, motor commands, route error, wheel motion and supervisor state.
Predetermined new-test trace replays are saved separately and do not increase
the final sample count.

## Failures, limits and interpretation

All final failures, including regressions, are retained in all-test-failures.csv
under research/artifacts/pid-gains. The paired table exposes gained and lost
completions. Raw all-zero events may include intentional corner manoeuvres;
their counts alone do not demonstrate fault robustness. No sensor faults,
motor degradation, disturbances or hardware transfer were tested.

The conclusion is limited to the observed paired differences above. One pooled
gain search, one selected policy per family, deterministic episodes, a fixed
robot and one generator do not establish statistical significance, a global
optimum, or algorithm-wide superiority. Future improvement cannot be assumed;
negative findings are retained as evidence about this bounded gain-only search.
The original controllers remain preserved, even when a new selection is made.

## Reproducibility

Run: runs/pid-gains-20261029. Source snapshot: source.zip. Files include run.json,
candidates.csv, candidate-summary.csv, all phase episode CSVs, full training
manifest, suite/validation and suite/test, preserved-records.json,
selection-freeze.json, selected configurations, status.json and telemetry/.
Source SHA256: **{freeze['source_sha256']}**. Runner SHA256:
**{freeze['runner_sha256']}**. Suite SHA256: **{freeze['suite_sha256']}**.
No Git commit is available; snapshots and hashes identify the code.
SHA256 checks verified **{metrics['preserved_files_verified']} historical files
unchanged**, including the original NEAT CSVs and completed initial PID study.
Additional geometry, replay-consistency and frozen-selection audit details are
saved in research/artifacts/pid-gains/integrity.json.

Editable controls are at the top of run_pid_gain_tuning.py. A fresh output is
mandatory: `.venv\\Scripts\\python.exe run_pid_gain_tuning.py --output runs/pid-gains-replication`.
The historical exclusion registry can grow between invocations, so a seed alone
does not reproduce the accepted geometry. Exact reproduction requires the saved
source and accepted training/suite manifests (and their JSON geometries). Fixed
policy replays on those saved tracks are reproductions, not independent evidence.
Use a new prespecified suite seed for another fresh assessment and include any
newly opened suite geometry in the exclusion registry. Analyze saved results with
`.venv\\Scripts\\python.exe research/analyze_pid_gains.py` (the analysis script's
RUN constant selects the completed run). Do not use now-known test outcomes for
further tuning without declaring a new development phase and untouched cohort.
"""
    (ROOT / "research/PID_Gain_Tuning_Results.md").write_text(report, encoding="utf-8")
    brief = f"## Separate gains-only follow-up (29 October; report finalized {REPORT_DATE})\n\n"+table+"\n"+"\n".join(paired_text)+"\n\n"
    brief += "Filtered PID retained its original gains; its two role rows refer to the same episodes, not independent repetitions. The supervised selection changed only P/I/D.\n\n"
    brief += "These are paired results on a **new** frozen 90-route cohort. They must not be pooled with the original NEAT/PID table above. Only P/I/D varied; no controller logic or non-gain settings changed. The original study and raw CSVs remain intact. Full protocol, gains, validation, failures, limitations and figures: [PID_Gain_Tuning_Results.md](PID_Gain_Tuning_Results.md).\n"
    update_section(ROOT / "research/PID_Experimental_Results.md", brief)
    update_section(ROOT / "research/Comparative_Analysis.md", brief)
    readme = "## PID gain-only tuning follow-up\n\n"
    readme += "The completed follow-up is `runs/pid-gains-20261029`. It changes only P/I/D of each frozen PID variant; its selected JSON files and `best.json` use `ResearchPIDConfig`. Original runs are preserved. See [results](research/PID_Gain_Tuning_Results.md) and [prespecified protocol](research/PID_Gain_Tuning_Protocol.md).\n\n"
    readme += "Run a fresh study with `.venv\\Scripts\\python.exe run_pid_gain_tuning.py --output runs/pid-gains-new-run`. Seed, workers and bounded search controls are editable globals. Rebuild the original reports first, then run `research/analyze_pid_gains.py` to restore follow-up sections and figures. Existing test cohorts are now known; fresh confirmatory work needs a new prespecified suite seed.\n"
    update_section(ROOT / "README.md", readme)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status, freeze, run = (load_json(RUN / name) for name in ("status.json", "selection-freeze.json", "run.json"))
    assert status["complete"] and freeze["frozen_before_test"]
    assert environment_record()["source_sha256"] == freeze["source_sha256"]
    assert hashlib.sha256((ROOT / "run_pid_gain_tuning.py").read_bytes()).hexdigest() == freeze["runner_sha256"]
    assert hashlib.sha256((RUN / "suite/suite.json").read_bytes()).hexdigest() == freeze["suite_sha256"]
    assert hashlib.sha256((ROOT / "research/PID_Gain_Tuning_Protocol.md").read_bytes()).hexdigest() == freeze["protocol_sha256"]
    preserved = load_json(RUN / "preserved-records.json")
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in preserved.items())
    candidates = {r["candidate"]: r for r in rows("candidates.csv")}
    assert len(candidates) == status["candidates"]
    assert len({(r["family"], r["kp"], r["ki"], r["kd"]) for r in candidates.values()}) == len(candidates)
    for name in ("coarse-episodes.csv", "expanded-episodes.csv", "refine-episodes.csv", "validation-episodes.csv",
                 "test-episodes.csv", "legacy-test-episodes.csv", "representative-episodes.csv", "test-trace-replays.csv"):
        records = rows(name)
        assert not any(r["error"] for r in records)
        assert all(float(r["mean_absolute_lateral_error_m"]) <= float(r["lateral_rmse_m"])+1e-10 for r in records)
    for family in LABELS:
        path = RUN / f"selected-{family}.json"
        assert load_json(path) == freeze["parameters"][freeze["selected"][family]]
    assert load_json(RUN / "best.json") == freeze["parameters"][freeze["overall"]]
    for candidate in candidates.values():
        baseline = run["baseline_parameters"][candidate["family"]]
        for name, value in baseline.items():
            if name not in ("kp", "ki", "kd"):
                actual = candidate[name] == "True" if isinstance(value, bool) else float(candidate[name])
                assert actual == value
    test, validation = rows("test-episodes.csv"), rows("validation-episodes.csv")
    training_hashes = {r["geometry_hash"] for r in load_json(RUN / "training-manifest.json")["tracks"]}
    test_hashes = {r["track_hash"] for r in test}
    val_hashes = {r["track_hash"] for r in validation}
    assert len(test_hashes) == 90 and len(val_hashes) == 30
    assert not training_hashes & (test_hashes | val_hashes) and not test_hashes & val_hashes
    protected = set(load_json(RUN / "suite/suite.json")["protected_geometry_hashes"])
    assert not protected & (test_hashes | val_hashes)
    summaries, paired = [], []
    for phase in ("test", "legacy-test", "validation", "expanded-training"):
        records = rows("expanded-episodes.csv")+rows("refine-episodes.csv") if phase == "expanded-training" else rows(f"{phase}-episodes.csv")
        assert not any(r["error"] for r in records)
        for family in LABELS:
            for role, key in (("baseline", freeze["baseline_ids"][family]), ("selected", freeze["selected"][family])):
                policy_rows = [r for r in records if r["candidate"] == key]
                assert len(policy_rows) == {"validation": 30, "expanded-training": 45}.get(phase, 90)
                if phase == "test":
                    assert {r["track_hash"] for r in policy_rows} == test_hashes
                levels = ("all", "manual", "beginner", "intermediate", "advanced") if phase == "expanded-training" else ("all", "beginner", "intermediate", "advanced")
                for difficulty in levels:
                    subset = policy_rows if difficulty == "all" else [r for r in policy_rows if r["difficulty"] == difficulty]
                    summaries.append({"phase": phase, "family": family, "role": role, "candidate": key,
                                      "difficulty": difficulty, **summary(subset)})
    for family in LABELS:
        baseline = {r["track_hash"]: r for r in test if r["candidate"] == freeze["baseline_ids"][family]}
        selected = {r["track_hash"]: r for r in test if r["candidate"] == freeze["selected"][family]}
        cells = Counter((baseline[h]["completed"] == "True", selected[h]["completed"] == "True") for h in baseline)
        common = [h for h in baseline if baseline[h]["completed"] == selected[h]["completed"] == "True"]
        old_time = mean([baseline[h] for h in common], "completion_time_s")
        new_time = mean([selected[h] for h in common], "completion_time_s")
        paired.append({"family": family, "both_complete": cells[(True, True)], "baseline_only_complete": cells[(True, False)],
                       "selected_only_complete": cells[(False, True)], "neither_complete": cells[(False, False)],
                       "completion_delta": cells[(False, True)]-cells[(True, False)], "baseline_common_time_s": old_time,
                       "selected_common_time_s": new_time, "paired_time_change_s": new_time-old_time})
    metrics = {"summaries": summaries, "paired": paired, "status": status, "freeze": freeze,
               "preserved_files_verified": len(preserved),
               "analysis": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                            "python": sys.version,
                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (OUTPUT / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export("summary.csv", [{**r, "failure_distribution": json.dumps(r["failure_distribution"], sort_keys=True)} for r in summaries])
    export("paired-comparison.csv", paired)
    failures = [{k: r[k] for k in ("candidate", "condition", "track_name", "track_hash", "difficulty", "features", "termination", "progress", "elapsed_s")}
                for r in test if r["completed"] != "True"]
    if failures:
        export("all-test-failures.csv", failures)
    comparison_figure(summaries)
    search_figure()
    sensitivity_figure(candidates, freeze)
    trajectory_figure(freeze)
    write_report(metrics, candidates, freeze, run)
    print(json.dumps({"paired": paired, "status": status, "preserved_files_verified": len(preserved)}, indent=2))


if __name__ == "__main__":
    main()
