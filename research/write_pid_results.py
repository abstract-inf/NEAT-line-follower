"""Verify frozen results and generate research tables, figures and drafts.

This script analyzes saved data only. It never tunes or evaluates controllers.
All means retain failed episodes and weight tracks equally unless stated.
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
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.experiments import environment_record
from line_follower.tracks import TrackMask, TrackSpec

RUN = ROOT / "runs/pid-research-20261029"
RESEARCH = ROOT / "research"
OUTPUT = RESEARCH / "artifacts/pid"
LABELS = {"pid_filtered": "Filtered PID", "pid_corner_supervised": "PID + corner supervision", "neat_frozen_g109": "Frozen NEAT g109"}
COLORS = {"pid_filtered": "#3979a9", "pid_corner_supervised": "#26845c", "neat_frozen_g109": "#a34c43"}


def read(name: str) -> list[dict]:
    with (RUN / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def export(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], key: str):
    return float(np.mean([float(r[key]) for r in rows])) if rows else None


def summarize(rows: list[dict]) -> dict:
    successes = [r for r in rows if r["completed"] == "True"]
    failures = [r for r in rows if r["completed"] != "True"]
    return {"episodes": len(rows), "completed": len(successes), "completion_rate": len(successes)/len(rows),
            "observed_lateral_rmse_m": mean(rows, "lateral_rmse_m"),
            "observed_lateral_mae_m": mean(rows, "mean_absolute_lateral_error_m"),
            "mean_progress": mean(rows, "progress"), "failed_mean_progress": mean(failures, "progress"),
            "successful_completion_time_s": mean(successes, "completion_time_s"),
            "mean_elapsed_s": mean(rows, "elapsed_s"), "mean_forward_speed_m_s": mean(rows, "average_forward_speed_m_s"),
            "line_loss_events": sum(int(r["line_loss_events"]) for r in rows),
            "failures": dict(Counter(r["termination"] for r in failures))}


def save_figure(figure, name: str) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def performance_figure(summaries: list[dict]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    families = list(LABELS)
    x = np.arange(3)
    for index, family in enumerate(families):
        rows = [next(r for r in summaries if r["condition"] == family and r["difficulty"] == d)
                for d in ("beginner", "intermediate", "advanced")]
        bars = axes[0].bar(x+(index-1)*.25, [r["completion_rate"] for r in rows], width=.25, label=LABELS[family], color=COLORS[family])
        for bar, row in zip(bars, rows):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.015, f"{row['completed']}/30", ha="center", fontsize=7)
    axes[0].set(xticks=x, xticklabels=["Beginner", "Intermediate", "Advanced"], ylabel="Completion fraction", ylim=(0, 1.15))
    axes[0].legend(fontsize=7)
    all_rows = [next(r for r in summaries if r["condition"] == f and r["difficulty"] == "all") for f in families]
    labels = [LABELS[f].replace(" + ", "\n+ ") for f in families]
    for axis, key, title in zip(axes[1:], ("observed_lateral_rmse_m", "observed_lateral_mae_m"), ("Observed lateral RMSE (mm)", "Observed lateral MAE (mm)")):
        axis.bar(np.arange(3), [r[key]*1000 for r in all_rows], color=[COLORS[f] for f in families])
        axis.set(xticks=np.arange(3), xticklabels=labels, ylabel=title)
        axis.tick_params(axis="x", labelsize=7)
    figure.suptitle("Frozen common test: same 90 tracks, CPU physics, 0.35 m/s ceiling\nFailures included; PID supervisor has additional temporal/structural prior; one NEAT lineage", fontsize=10)
    save_figure(figure, "common-test")


def tuning_figure(rows: list[dict]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for axis, phases, title in zip(axes, (("coarse",), ("expanded", "refine"), ("validation",)),
                                   ("Short training suite (6)", "Expanded training suite (15)", "Validation selection (30)")):
        for family, color in (("pid_filtered", "#3979a9"), ("pid_corner_supervised", "#26845c")):
            selected = [r for r in rows if r["phase"] in phases and r["family"] == family]
            axis.scatter([int(r["candidate"].split("-")[1]) for r in selected], [float(r["completion_rate"]) for r in selected],
                         s=20, color=color, label=LABELS[family], alpha=.75)
        axis.set(title=title, xlabel="Candidate ID", ylabel="Completion fraction", ylim=(-.03, 1.05))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    figure.suptitle("Bounded tuning: 120 configurations; separate stages face different cohorts", fontsize=10)
    save_figure(figure, "tuning")


def trajectory_figure(representatives: list[dict], selected: list[str]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 5.5), layout="constrained")
    for axis, slot in zip(axes, (3, 4)):
        track = TrackSpec.load(RUN / "tracks/training" / f"{slot:03d}.json")
        mask = TrackMask(track)
        half = track.arena_size/2
        axis.imshow(mask.field, extent=(-half-track.start[0], half-track.start[0], -half-track.start[1], half-track.start[1]),
                    cmap=ListedColormap([".82", "white"]), vmin=0, vmax=255, interpolation="nearest")
        for key in selected:
            data = list(csv.DictReader((RUN / "telemetry/representative" / f"{key}-{slot:03d}.csv").open()))
            row = next(r for r in representatives if r["candidate"] == key and r["track_name"] == track.name)
            axis.plot([float(r["x_m"])-track.start[0] for r in data], [float(r["y_m"])-track.start[1] for r in data],
                      label=LABELS[row["condition"]], lw=1.5, color=COLORS[row["condition"]])
        axis.set(title="Left-first zigzag" if slot == 3 else "Right-first zigzag", xlabel="x from start (m)", ylabel="y from start (m)", xlim=(-.05, 1.0),
                 ylim=(-.05, .9) if slot == 3 else (-.9, .05), aspect="equal")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.suptitle("Real sensor-driven PID trajectories; diagnostic training benchmarks, not held-out results", fontsize=10)
    save_figure(figure, "diagnostic-trajectories")


def signals_figure(overall: str) -> None:
    path = RUN / "telemetry/representative" / f"{overall}-003.csv"
    rows = list(csv.DictReader(path.open()))
    t = [float(r["observation_time_s"]) for r in rows]
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8), layout="constrained")
    axes[0].plot(t, [float(r["pid_error"]) for r in rows], label="Normalized sensor error")
    axes[0].set_ylabel("Sensor error\n(dimensionless)")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, [float(r["pid_derivative"]) for r in rows], color="#a34c43")
    axes[1].set_ylabel("Filtered derivative\n(1/s)")
    for name, label in (("action_L", "Left"), ("action_R", "Right")):
        axes[2].plot(t, [float(r[name]) for r in rows], label=label)
    axes[2].set(ylabel="Motor command", ylim=(-1.1, 1.1))
    axes[2].legend(fontsize=8)
    modes = {"tracking": 0, "approach": 1, "pivot": 2}
    axes[3].step(t, [modes[r["pid_mode"]] for r in rows], where="post")
    axes[3].set(xlabel="Simulation time (s)", ylabel="Supervisor state", yticks=[0, 1, 2], yticklabels=list(modes))
    figure.suptitle("Selected PID: actual diagnostic-zigzag signals and signed wheel commands", fontsize=10)
    save_figure(figure, "pid-signals")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((RUN / "selection-freeze.json").read_text())
    status = json.loads((RUN / "status.json").read_text())
    run = json.loads((RUN / "run.json").read_text())
    assert status["complete"] and freeze["frozen_before_test"]
    assert environment_record()["source_sha256"] == freeze["source_sha256"]
    for candidate, parameters in freeze["parameters"].items():
        assert hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest() == freeze["configuration_hashes"][candidate]
    test, validation, representatives = read("test-episodes.csv"), read("validation-episodes.csv"), read("representative-episodes.csv")
    assert len(test) == 270 and not any(r["error"] for r in test+validation+representatives)
    summaries = []
    expected = None
    for family in LABELS:
        rows = [r for r in test if r["condition"] == family]
        identities = {r["track_hash"] for r in rows}
        assert len(rows) == len(identities) == 90
        if expected is None:
            expected = identities
        assert identities == expected
        for difficulty in ("all", "beginner", "intermediate", "advanced"):
            subset = rows if difficulty == "all" else [r for r in rows if r["difficulty"] == difficulty]
            metrics = summarize(subset)
            summaries.append({"condition": family, "difficulty": difficulty,
                              **{k:v for k,v in metrics.items() if k != "failures"}, "failure_distribution": json.dumps(metrics["failures"], sort_keys=True)})
    training_hashes = {r["geometry_hash"] for r in json.loads((RUN / "training-manifest.json").read_text())["tracks"]}
    assert not training_hashes & expected
    paired = []
    neat = {r["track_hash"]: r for r in test if r["condition"] == "neat_frozen_g109"}
    for family in ("pid_filtered", "pid_corner_supervised"):
        cells = Counter()
        for row in (r for r in test if r["condition"] == family):
            a, b = row["completed"] == "True", neat[row["track_hash"]]["completed"] == "True"
            cells[(a,b)] += 1
        paired.append({"condition": family, "both_complete": cells[(True, True)], "pid_only_complete": cells[(True, False)],
                       "neat_only_complete": cells[(False, True)], "neither_complete": cells[(False, False)]})
    export(OUTPUT / "test-summary.csv", summaries)
    export(OUTPUT / "paired-completion.csv", paired)
    failures = [{"condition": r["condition"], "track_name": r["track_name"], "track_hash": r["track_hash"],
                 "difficulty": r["difficulty"], "features": r["features"], "termination": r["termination"],
                 "progress": r["progress"], "lateral_rmse_m": r["lateral_rmse_m"], "elapsed_s": r["elapsed_s"]}
                for r in test if r["completed"] != "True"]
    export(OUTPUT / "all-test-failures.csv", failures)
    candidate_rows = read("candidate-summary.csv")
    performance_figure(summaries)
    tuning_figure(candidate_rows)
    trajectory_figure(representatives, freeze["selected"])
    signals_figure(freeze["overall"])
    validation_summary = {key: summarize([r for r in validation if r["candidate"] == key]) for key in freeze["selected"]}
    feature_summary = {family: summarize([r for r in test if r["condition"] == family and "zigzag" in r["features"].split(";")])
                       for family in LABELS}
    metrics = {"test": summaries, "validation": validation_summary, "test_contains_zigzag": feature_summary,
               "paired": paired, "status": status, "freeze": freeze}
    (OUTPUT / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_documents(run, metrics, representatives, candidate_rows)
    print(json.dumps({"validation": validation_summary, "test": [r for r in summaries if r["difficulty"] == "all"],
                      "test_contains_zigzag": feature_summary, "paired": paired}, indent=2))


def write_documents(run: dict, metrics: dict, representatives: list[dict], candidate_rows: list[dict]) -> None:
    freeze, status = metrics["freeze"], metrics["status"]
    table = ["| Controller / difficulty | Completed / n | Mean RMSE (mm) | Mean MAE (mm) | Mean progress | Successful time (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in metrics["test"]:
        table.append(f"| {LABELS[row['condition']]}: {row['difficulty']} | {row['completed']}/{row['episodes']} | "
                     f"{1000*row['observed_lateral_rmse_m']:.3f} | {1000*row['observed_lateral_mae_m']:.3f} | "
                     f"{row['mean_progress']:.4f} | {row['successful_completion_time_s']:.3f} |")
    diagnostic_table = ["| Policy | Diagnostic route | Completed | Progress | RMSE (mm) | Time (s) |",
                        "|---|---|---|---:|---:|---:|"]
    for r in representatives:
        if "zigzag" in r["track_name"]:
            diagnostic_table.append(f"| {LABELS[r['condition']]} | {r['track_name']} | {r['completed']} | {float(r['progress']):.4f} | "
                                    f"{1000*float(r['lateral_rmse_m']):.3f} | {float(r['elapsed_s']):.3f} |")
    image = (OUTPUT.as_posix())
    failure_table = "\n".join(f"- {LABELS[r['condition']]}: {r['failure_distribution']}" for r in metrics["test"] if r["difficulty"] == "all")
    zigzag_table = "\n".join(f"| {LABELS[family]} | {m['completed']}/{m['episodes']} | {m['mean_progress']:.4f} |"
                              for family, m in metrics["test_contains_zigzag"].items())
    protocol = (RESEARCH / "PID_Protocol.md").read_text(encoding="utf-8")
    body = f"""# PID development, optimization and experimental results

Scientific research draft, 29 October 2026. The NEAT baseline was stopped and archived before PID implementation and tuning. This report concerns the implemented controllers and the frozen protocol, not universal algorithm superiority.

## 1. Robot, observations and controller variants

The detailed implementation reference is [PID_Controller_Methodology.md](PID_Controller_Methodology.md). It records the exact PID equations, corner-state transitions and thresholds, command-odometry approximations, telemetry semantics, parameter provenance, and manuscript comparison limits.

The existing 0.55 kg competition robot, 0.175 m wheel spacing, 0.0165 m wheel radius, 0.35 m/s rim-speed ceiling, 0.03 N m torque limits and 50 ms actuator filter were retained. Physics/control run at 600/60 Hz. The same 15 pose-dependent binary floor sensors, 90 mm central and 52.6 mm outer longitudinal offsets, reflectance raster, caster, contacts, gates, corridor and stopping rules are used. No physical geometry or legacy NEAT/PID implementation was changed.

`line_follower/pid.py` defines two new, declared variants. Both receive only 15 readings and dt; no pose, yaw, wheel velocity, route, progress, difficulty or track identity enters their controller interface. Both have temporal state, unlike the preserved feedforward NEAT model. The old PID class and historical gains remain in controllers.py unchanged.

**Filtered PID:** error e=mean(active_sensor_indices)/7-1, with negative error meaning the line is to the left. Raw derivative (e-e_previous)/dt is low-pass filtered with fraction 1-exp(-dt/tauD). Integral is conditionally accumulated and clipped to +/-0.4 error-seconds; integration stops while the requested turn saturates, unless it would unwind the error. Turn demand u=Kp*e+Ki*I+Kd*D. Forward command is max(minimum_command,base/(1+slowdown_gain*abs(e))). Motor targets are clip([forward+u,forward-u],-1,1), passed through the unchanged physical actuator model.

During all-zero sensing, the last valid command is held for a bounded configured duration, then a signed pivot searches in the last-seen direction, alternating after one second. The integral is not accumulated during loss. This is explicitly a PID with recovery logic, not the unchanged historical PID. There is no positive minimum imposed separately on each wheel, so opposite wheel commands can pivot.

**PID with corner supervision:** the same controller plus a finite-state supervisor. At least three active sensors, unilateral normalized error magnitude >=0.18, and a sensor in indices 6-8 indicate a potential hard turn. It estimates distance to the corner from the median known forward coordinates of active sensors minus a tuned lead distance. A straight approach integrates estimated forward speed from past commands through a known 50 ms first-order response. This is command odometry, not true simulator displacement or an encoder observation. A continuing central line at the end of the approach cancels the turn (e.g. a straight-through branch). Otherwise opposite wheel commands pivot until a central line is reacquired after at least 0.65 rad estimated rotation, or a 2.5 s timeout. A 0.25 s cooldown prevents immediate retriggering. Sensor readings, not a stored sequence of track turns, determine each manoeuvre.

This supervisor encodes geometric/mechanical prior knowledge and extra memory. Its results must be labelled **PID + corner supervision**, not attributed to plain PID alone. Estimated command odometry is approximate and may be sensitive to friction, slip and model errors; none of those robustness perturbations was tested.

## 2. Frozen tuning and evaluation methodology

Protocol: research/PID_Protocol.md, saved before optimization. Seed {run['settings']['seed']}, {run['settings']['workers']} persistent Windows-compatible CPU workers. Ninety-six initial candidates (48 per family), then the best six per family on expanded training, then 24 bounded local refinements around training-selected parents: **120 distinct configurations**. There were 576 coarse, 180 expanded and 360 refinement episode evaluations; these are repeated deterministic evaluations on fixed training cases, not independent random trials. Training cohorts use mirrored short corners/zigzags, curves, then 45-degree-entry zigzags and six separately seeded procedural tracks (two per difficulty). All accepted geometry, hashes and seeds are saved; internal motif-search rejections are not separately logged. Mirrored cases intentionally coexist within training; none overlaps validation/test canonical hashes. The procedural training sample is small and is a limitation of this bounded study.

The candidate ranking is lexicographic: completion fraction, mean authenticated progress, negative mean observed lateral RMSE, negative mean successful completion time. It does not optimize against final-test outcomes. The top three per family from expanded training were evaluated on 30 frozen validation tracks (180 validation episodes). Selected policies were {freeze['selected']}; overall choice was **{freeze['overall']}**, based exclusively on validation. The filtered variant completed {metrics['validation'][freeze['selected'][0]]['completed']}/30 validation tracks; the supervised variant {metrics['validation'][freeze['selected'][1]]['completed']}/30.

Before loading test geometry, selection-freeze.json saved exact parameters, source/suite/model hashes and UTC freeze time **{freeze['frozen_utc']}**. The already selected generation-109 NEAT artifact was hash-frozen before PID tuning. After freezing, three policies were evaluated once on the same 90 held-out tracks (30 per difficulty), CPU backend, same robot and 45-120 s length-dependent time limits: **270 final-test episodes**. There was no retuning, reselection or controller-code change after observing test outcomes. Failures remain in the results. No NEAT training or PPO work occurred.

Total study wall time was **{status['wall_seconds']:.3f} s ({status['wall_seconds']/60:.3f} min)**. This includes preparation, tuning, validation, final test and representative trace exports; it is not a pure controller inference benchmark. It is not a compute-matched comparison to the much larger NEAT evolution budget. Sources/dependencies, raw per-episode data and every candidate parameter vector accompany the run.

## 3. Selected configuration

Overall best JSON: runs/pid-research-20261029/best.json. This is the **supervised** variant; the filtered variant is separately preserved.

```json
{json.dumps(freeze['parameters'][freeze['overall']], indent=2)}
```

The gain units correspond to normalized sensor-index error, not the old unnormalized [-7,7] error. Values therefore must not be copied into the historical PID class without an explicit adapter.

## 4. Final held-out results

{chr(10).join(table)}

All errors include incomplete episodes over their observed duration. Tables average per-episode metrics equally across tracks. Completion time is averaged only over successful episodes; a controller failing early can misleadingly have low error or short elapsed time. Progress is greatest ordered-gate-authenticated route fraction. MAE is directly accumulated, not inferred from RMSE. These are deterministic policy evaluations on one frozen cohort, not independent controller-training replications.

Both held-out cohorts are balanced by difficulty, but feature composition differs: validation has nine zigzag-containing routes out of 30; test has 40 out of 90. Validation-to-test differences therefore cannot isolate overfitting from challenge-mix differences. The final paired comparison uses identical individual tracks for every method.

For the **40 final-test routes containing zigzags** (membership does not localize each failure):

| Controller | Completed / n | Mean progress |
|---|---:|---:|
{zigzag_table}

The supervised variant is slower: successful completion time averages 29.518 s versus 10.932 s for filtered PID and 11.268 s for NEAT, but those averages refer to different successful subsets and are not paired speed comparisons. Mean measured forward speed across all observed runs is 0.116 m/s for supervision, 0.226 for filtered PID and 0.183 for NEAT. A 0.35 m/s wheel target ceiling is not a constant chassis speed.

Raw line-loss counts are 578 (supervised), 101 (filtered) and 153 (NEAT) across the 90-track assessment. The supervised approach/pivot intentionally passes through all-zero sensor intervals while the axle remains in the route corridor; its higher raw loss count is not a contradiction of its higher completion. These are event counts over different observed durations, not rates or sensor-fault robustness measurements. Full loss durations and unplanned-event flags remain in the episode CSVs.

![Common held-out performance](<{image}/common-test.png>)

![Training and validation stages kept separate](<{image}/tuning.png>)

## 5. Diagnostic zigzag and signal traces

{chr(10).join(diagnostic_table)}

Both mirrored diagnostic zigzags are completed by the selected supervised policy. These routes were explicitly included in the training-only engineering suite; they are not unseen test successes. The filtered PID's remaining failure is preserved rather than hidden. The earlier NEAT diagnostic and successful fixed-command mechanical probe remain in the NEAT research document.

![Measured diagnostic trajectories](<{image}/diagnostic-trajectories.png>)

![Recorded error, motor commands and supervisor states](<{image}/pid-signals.png>)

Telemetry CSVs contain pre-action sensor readings, next readings, physical pose/velocity/torque, route error/progress, PID error/integral/filtered derivative, motor commands and supervisor state at 60 Hz. Representative training routes and selected test indices were designated for trace export; summaries include all attempted test routes. No trajectory plot is an image-generation mockup.

## 6. Remaining failures and interpretation

Final-test failure reasons:

{failure_table}

Every failed route is enumerated with track hash, features, elapsed time and progress in research/artifacts/pid/all-test-failures.csv. Feature membership indicates the route contains that challenge, not that the actual failure occurred there. The assisted controller still has unsolved cases; the bounded search was stopped as prespecified rather than extending optimization against the test set. No claim of comprehensive line-loss recovery, sensor-fault tolerance, disturbance robustness or real-world performance is supported.

The results demonstrate that a sensor-only PID with explicit corner handling can traverse the diagnostic geometry and many unseen routes in this simulator. They do not show that ordinary PID alone explains the supervised variant's improvement. The supervisor's temporal logic and signed motor access differ from the selected sigmoid feedforward NEAT policies. Filtered PID and supervised PID are therefore kept as separate baselines in all plots and tables. One NEAT lineage, one tuning seed, changed historical NEAT protocols and unequal computational budgets limit algorithm-level conclusions.

## 7. Reproducibility and future work

Run directory: runs/pid-research-20261029. Files: run.json; source.zip (controller, runner, tests, protocol and shared environment); candidates.csv; coarse/expanded/refine/validation/test/representative-episodes.csv; candidate-summary.csv; training-manifest.json and complete tracks/; selection-freeze.json; best.json; selected-pid_filtered.json; selected-pid_corner_supervised.json; telemetry/; status.json. Derived tables/figures are under research/artifacts/pid. The source SHA256 is **{freeze['source_sha256']}**; suite SHA256 **{freeze['suite_sha256']}**; NEAT model SHA256 **{freeze['neat_model_sha256']}**. No Git commit was available. Original NEAT CSVs remain unchanged.

Reproduce a new exploratory run with `.venv\\Scripts\\python.exe run_pid_optimization.py --output runs/pid-new-run`. Editable constants in that runner set budgets/seed/workers; ResearchPIDConfig in line_follower/pid.py defines controller parameters. Existing output directories are rejected to prevent overwriting data. Because final test outcomes are now known, any further tuning must be labelled development and use a newly frozen untouched test cohort for a fresh confirmatory claim. Exact replication of this fixed run is not another independent trial.

Recommended future experiments, not implemented here: independent tuning/evolution seeds; explicit ablations of supervisor, temporal state and action activation; varied sensor offsets under a separate geometry protocol; hardware-calibrated dynamics; disturbance/fault studies; and a separately prespecified PPO baseline. PPO results remain unavailable. No further NEAT optimization was performed.

## Appendix: prespecified protocol

{protocol}
"""
    (RESEARCH / "PID_Experimental_Results.md").write_text(body, encoding="utf-8")
    short_table = ["| Controller | Test completion | Mean RMSE (mm) | Mean MAE (mm) |",
                   "|---|---:|---:|---:|"]
    for row in metrics["test"]:
        if row["difficulty"] == "all":
            short_table.append(f"| {LABELS[row['condition']]} | {row['completed']}/90 ({100*row['completion_rate']:.2f}%) | "
                               f"{1000*row['observed_lateral_rmse_m']:.3f} | {1000*row['observed_lateral_mae_m']:.3f} |")
    short_table.append("| PPO | Not evaluated | Not available | Not available |")
    comparative = f"""# Common research comparison: NEAT, PID and potentially PPO

Finalized 29 October 2026. This is a descriptive comparison of frozen implementations, not evidence of universal algorithm superiority.

## Planned and executed methodology

All evaluated policies faced the **same 90 held-out tracks**, with identical hashes, 30 per difficulty, unchanged robot geometry, 0.35 m/s motor-rim speed ceiling, 0.03 N m torque limits, 600/60 Hz physics/control, binary reflectance and ordered-route evaluator. CPU MuJoCo is the common assessment backend. The historical NEAT GPU validation results remain separate. Neither validation nor training rows are pooled into this test table.

NEAT g109 was fixed before PID tuning. PID trained/tuned only on dedicated training tracks, selected on the same 30 validation tracks, and froze exact parameters and source before test geometry was loaded by the pipeline. After test outcomes, there was no tuning or reselection. Protocol and freeze are in research/PID_Protocol.md and runs/pid-research-20261029/selection-freeze.json. The selected overall PID is {freeze['overall']} (with corner supervision); filtered PID is separately shown.

## Actual common held-out results

{chr(10).join(short_table)}

Macro error averages include failures over observed duration. Successful completion times and difficulty-stratified tables are in PID_Experimental_Results.md. The same per-episode columns, including directly measured MAE, are available in runs/pid-research-20261029/test-episodes.csv for Pandas/paired-track analysis. No p-value or statistical significance is claimed.

![Common assessment](<{image}/common-test.png>)

Paired completion contingency counts, without treating repeated telemetry as independent:

```json
{json.dumps(metrics['paired'], indent=2)}
```

## Scope and fairness limits

Controller equations, exact supervisor logic, observable information, telemetry interpretation, and recommended manuscript terminology are documented in [PID_Controller_Methodology.md](PID_Controller_Methodology.md).

Physical limits and physical sensors match. Internal controller classes do not: filtered PID has temporal derivative/integral/recovery state; the supervised variant additionally encodes corner-detection, approximate command odometry and pivot logic. The selected NEAT model is memoryless with sigmoid motor outputs and cannot command the same reverse manoeuvres, although the simulator interface permits them. The strong supervised result cannot be attributed to unmodified textbook PID. Training budgets and methods are not matched, and only one NEAT evolution lineage and one PID tuning seed exist. Results compare these implementations under the documented conditions, not NEAT/PID algorithm families in general.

The historical parent/continuation phases use different speed/timeout/curriculum conditions and are documented separately in NEAT_Experimental_Results.md. Current final-test failures remain in the datasets. Unseen generated routes are tested; disturbances, sensor faults, motor degradation and real-world transfer are not. There is no basis for corresponding robustness claims.

## PID results

Bounded search completed with 120 configurations, 6 validation candidates and two frozen variants. See PID_Experimental_Results.md for equations, selected gains, all failure reasons, traces, training-stage plots and protocol. No further tuning followed this final assessment.

## PPO results

No PPO implementation, training or assessment was performed. Result cells are intentionally unavailable. A future PPO experiment must prespecify observations, actions, reward, training/selection budget and independent seeds. Now-known test outcomes must not drive PPO tuning; a new unopened confirmatory cohort may be needed for a stronger future comparison.
"""
    (RESEARCH / "Comparative_Analysis.md").write_text(comparative, encoding="utf-8")


if __name__ == "__main__":
    main()
