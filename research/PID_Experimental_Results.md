# PID development, optimization and experimental results

Initial-study research draft, 29 October 2026. The initial results below remain separate from the gains-only follow-up appended at the end. The NEAT baseline was stopped and archived before PID implementation and tuning. This report concerns the implemented controllers and the frozen protocol, not universal algorithm superiority.

## 1. Robot, observations and controller variants

The detailed implementation reference is [PID_Controller_Methodology.md](PID_Controller_Methodology.md). It records the exact PID equations, corner-state transitions and thresholds, command-odometry approximations, telemetry semantics, parameter provenance, and manuscript comparison limits.

The existing 0.55 kg competition robot, 0.175 m wheel spacing, 0.0165 m wheel radius, 0.35 m/s rim-speed ceiling, 0.03 N m torque limits and 50 ms actuator filter were retained. Physics/control run at 600/60 Hz. The same 15 pose-dependent binary floor sensors, 90 mm central and 52.6 mm outer longitudinal offsets, reflectance raster, caster, contacts, gates, corridor and stopping rules are used. No physical geometry or legacy NEAT/PID implementation was changed.

`line_follower/pid.py` defines two new, declared variants. Both receive only 15 readings and dt; no pose, yaw, wheel velocity, route, progress, difficulty or track identity enters their controller interface. Both have temporal state, unlike the preserved feedforward NEAT model. The old PID class and historical gains remain in controllers.py unchanged.

**Filtered PID:** error e=mean(active_sensor_indices)/7-1, with negative error meaning the line is to the left. Raw derivative (e-e_previous)/dt is low-pass filtered with fraction 1-exp(-dt/tauD). Integral is conditionally accumulated and clipped to +/-0.4 error-seconds; integration stops while the requested turn saturates, unless it would unwind the error. Turn demand u=Kp*e+Ki*I+Kd*D. Forward command is max(minimum_command,base/(1+slowdown_gain*abs(e))). Motor targets are clip([forward+u,forward-u],-1,1), passed through the unchanged physical actuator model.

During all-zero sensing, the last valid command is held for a bounded configured duration, then a signed pivot searches in the last-seen direction, alternating after one second. The integral is not accumulated during loss. This is explicitly a PID with recovery logic, not the unchanged historical PID. There is no positive minimum imposed separately on each wheel, so opposite wheel commands can pivot.

**PID with corner supervision:** the same controller plus a finite-state supervisor. At least three active sensors, unilateral normalized error magnitude >=0.18, and a sensor in indices 6-8 indicate a potential hard turn. It estimates distance to the corner from the median known forward coordinates of active sensors minus a tuned lead distance. A straight approach integrates estimated forward speed from past commands through a known 50 ms first-order response. This is command odometry, not true simulator displacement or an encoder observation. A continuing central line at the end of the approach cancels the turn (e.g. a straight-through branch). Otherwise opposite wheel commands pivot until a central line is reacquired after at least 0.65 rad estimated rotation, or a 2.5 s timeout. A 0.25 s cooldown prevents immediate retriggering. Sensor readings, not a stored sequence of track turns, determine each manoeuvre.

This supervisor encodes geometric/mechanical prior knowledge and extra memory. Its results must be labelled **PID + corner supervision**, not attributed to plain PID alone. Estimated command odometry is approximate and may be sensitive to friction, slip and model errors; none of those robustness perturbations was tested.

## 2. Frozen tuning and evaluation methodology

Protocol: research/PID_Protocol.md, saved before optimization. Seed 20261029, 4 persistent Windows-compatible CPU workers. Ninety-six initial candidates (48 per family), then the best six per family on expanded training, then 24 bounded local refinements around training-selected parents: **120 distinct configurations**. There were 576 coarse, 180 expanded and 360 refinement episode evaluations; these are repeated deterministic evaluations on fixed training cases, not independent random trials. Training cohorts use mirrored short corners/zigzags, curves, then 45-degree-entry zigzags and six separately seeded procedural tracks (two per difficulty). All accepted geometry, hashes and seeds are saved; internal motif-search rejections are not separately logged. Mirrored cases intentionally coexist within training; none overlaps validation/test canonical hashes. The procedural training sample is small and is a limitation of this bounded study.

The candidate ranking is lexicographic: completion fraction, mean authenticated progress, negative mean observed lateral RMSE, negative mean successful completion time. It does not optimize against final-test outcomes. The top three per family from expanded training were evaluated on 30 frozen validation tracks (180 validation episodes). Selected policies were ['pid-112', 'pid-111']; overall choice was **pid-111**, based exclusively on validation. The filtered variant completed 17/30 validation tracks; the supervised variant 28/30.

Before loading test geometry, selection-freeze.json saved exact parameters, source/suite/model hashes and UTC freeze time **2026-10-29T06:43:27.960881+00:00**. The already selected generation-109 NEAT artifact was hash-frozen before PID tuning. After freezing, three policies were evaluated once on the same 90 held-out tracks (30 per difficulty), CPU backend, same robot and 45-120 s length-dependent time limits: **270 final-test episodes**. There was no retuning, reselection or controller-code change after observing test outcomes. Failures remain in the results. No NEAT training or PPO work occurred.

Total study wall time was **189.320 s (3.155 min)**. This includes preparation, tuning, validation, final test and representative trace exports; it is not a pure controller inference benchmark. It is not a compute-matched comparison to the much larger NEAT evolution budget. Sources/dependencies, raw per-episode data and every candidate parameter vector accompany the run.

## 3. Selected configuration

Overall best JSON: runs/pid-research-20261029/best.json. This is the **supervised** variant; the filtered variant is separately preserved.

```json
{
  "kp": 1.3862436572865304,
  "ki": 0.11438062569735973,
  "kd": 0.0842552583118724,
  "base": 0.400487424737064,
  "minimum_command": 0.16279216758065698,
  "derivative_filter_s": 0.02996832153811821,
  "integral_limit": 0.4,
  "slowdown_gain": 2.4183516971523678,
  "loss_hold_s": 0.7250891707453846,
  "search_command": 0.48514936066550596,
  "corner_supervision": true,
  "approach_command": 0.3666210999339319,
  "turn_lead_m": 0.004156311172421112,
  "pivot_command": 0.3424007013749916
}
```

The gain units correspond to normalized sensor-index error, not the old unnormalized [-7,7] error. Values therefore must not be copied into the historical PID class without an explicit adapter.

## 4. Final held-out results

| Controller / difficulty | Completed / n | Mean RMSE (mm) | Mean MAE (mm) | Mean progress | Successful time (s) |
|---|---:|---:|---:|---:|---:|
| Filtered PID: all | 43/90 | 35.932 | 21.597 | 0.6697 | 10.932 |
| Filtered PID: beginner | 30/30 | 6.881 | 5.634 | 1.0000 | 7.909 |
| Filtered PID: intermediate | 8/30 | 46.957 | 27.521 | 0.5813 | 16.310 |
| Filtered PID: advanced | 5/30 | 53.957 | 31.636 | 0.4278 | 20.463 |
| PID + corner supervision: all | 76/90 | 8.805 | 5.643 | 0.9122 | 29.518 |
| PID + corner supervision: beginner | 30/30 | 4.197 | 3.279 | 1.0000 | 14.685 |
| PID + corner supervision: intermediate | 28/30 | 7.523 | 5.413 | 0.9668 | 35.564 |
| PID + corner supervision: advanced | 18/30 | 14.695 | 8.237 | 0.7698 | 44.835 |
| Frozen NEAT g109: all | 35/90 | 26.525 | 18.170 | 0.5714 | 11.268 |
| Frozen NEAT g109: beginner | 30/30 | 7.442 | 6.556 | 1.0000 | 9.292 |
| Frozen NEAT g109: intermediate | 2/30 | 36.313 | 23.920 | 0.4000 | 20.433 |
| Frozen NEAT g109: advanced | 3/30 | 35.818 | 24.033 | 0.3141 | 24.922 |

All errors include incomplete episodes over their observed duration. Tables average per-episode metrics equally across tracks. Completion time is averaged only over successful episodes; a controller failing early can misleadingly have low error or short elapsed time. Progress is greatest ordered-gate-authenticated route fraction. MAE is directly accumulated, not inferred from RMSE. These are deterministic policy evaluations on one frozen cohort, not independent controller-training replications.

Both held-out cohorts are balanced by difficulty, but feature composition differs: validation has nine zigzag-containing routes out of 30; test has 40 out of 90. Validation-to-test differences therefore cannot isolate overfitting from challenge-mix differences. The final paired comparison uses identical individual tracks for every method.

For the **40 final-test routes containing zigzags** (membership does not localize each failure):

| Controller | Completed / n | Mean progress |
|---|---:|---:|
| Filtered PID | 0/40 | 0.3409 |
| PID + corner supervision | 28/40 | 0.8309 |
| Frozen NEAT g109 | 0/40 | 0.2734 |

The supervised variant is slower: successful completion time averages 29.518 s versus 10.932 s for filtered PID and 11.268 s for NEAT, but those averages refer to different successful subsets and are not paired speed comparisons. Mean measured forward speed across all observed runs is 0.116 m/s for supervision, 0.226 for filtered PID and 0.183 for NEAT. A 0.35 m/s wheel target ceiling is not a constant chassis speed.

Raw line-loss counts are 578 (supervised), 101 (filtered) and 153 (NEAT) across the 90-track assessment. The supervised approach/pivot intentionally passes through all-zero sensor intervals while the axle remains in the route corridor; its higher raw loss count is not a contradiction of its higher completion. These are event counts over different observed durations, not rates or sensor-fault robustness measurements. Full loss durations and unplanned-event flags remain in the episode CSVs.

![Common held-out performance](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid/common-test.png>)

![Training and validation stages kept separate](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid/tuning.png>)

## 5. Diagnostic zigzag and signal traces

| Policy | Diagnostic route | Completed | Progress | RMSE (mm) | Time (s) |
|---|---|---|---:|---:|---:|
| Filtered PID | diagnostic-zigzag-1 | False | 0.5152 | 69.526 | 4.900 |
| Filtered PID | diagnostic-zigzag--1 | False | 0.5152 | 69.526 | 4.900 |
| PID + corner supervision | diagnostic-zigzag-1 | True | 1.0000 | 3.627 | 15.333 |
| PID + corner supervision | diagnostic-zigzag--1 | True | 1.0000 | 3.627 | 15.333 |

Both mirrored diagnostic zigzags are completed by the selected supervised policy. These routes were explicitly included in the training-only engineering suite; they are not unseen test successes. The filtered PID's remaining failure is preserved rather than hidden. The earlier NEAT diagnostic and successful fixed-command mechanical probe remain in the NEAT research document.

![Measured diagnostic trajectories](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid/diagnostic-trajectories.png>)

![Recorded error, motor commands and supervisor states](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid/pid-signals.png>)

Telemetry CSVs contain pre-action sensor readings, next readings, physical pose/velocity/torque, route error/progress, PID error/integral/filtered derivative, motor commands and supervisor state at 60 Hz. Representative training routes and selected test indices were designated for trace export; summaries include all attempted test routes. No trajectory plot is an image-generation mockup.

## 6. Remaining failures and interpretation

Final-test failure reasons:

- Filtered PID: {"off_route": 47}
- PID + corner supervision: {"off_route": 13, "stalled": 1}
- Frozen NEAT g109: {"off_route": 55}

Every failed route is enumerated with track hash, features, elapsed time and progress in research/artifacts/pid/all-test-failures.csv. Feature membership indicates the route contains that challenge, not that the actual failure occurred there. The assisted controller still has unsolved cases; the bounded search was stopped as prespecified rather than extending optimization against the test set. No claim of comprehensive line-loss recovery, sensor-fault tolerance, disturbance robustness or real-world performance is supported.

The results demonstrate that a sensor-only PID with explicit corner handling can traverse the diagnostic geometry and many unseen routes in this simulator. They do not show that ordinary PID alone explains the supervised variant's improvement. The supervisor's temporal logic and signed motor access differ from the selected sigmoid feedforward NEAT policies. Filtered PID and supervised PID are therefore kept as separate baselines in all plots and tables. One NEAT lineage, one tuning seed, changed historical NEAT protocols and unequal computational budgets limit algorithm-level conclusions.

## 7. Reproducibility and future work

Run directory: runs/pid-research-20261029. Files: run.json; source.zip (controller, runner, tests, protocol and shared environment); candidates.csv; coarse/expanded/refine/validation/test/representative-episodes.csv; candidate-summary.csv; training-manifest.json and complete tracks/; selection-freeze.json; best.json; selected-pid_filtered.json; selected-pid_corner_supervised.json; telemetry/; status.json. Derived tables/figures are under research/artifacts/pid. The source SHA256 is **78acab6866de9a70228f5c8e6b3e84a74592c0d00290927a0c87ccdc5909da8d**; suite SHA256 **1e24fd7d847e76768069aa07caae2a1da4c581f7ae27b7a208fdbf7375eea4dc**; NEAT model SHA256 **4cf2d8af1bc4b7759f4ede52b00ff12b78c8ecec316fa7aaea623d8662540dcd**. No Git commit was available. Original NEAT CSVs remain unchanged.

Reproduce a new exploratory run with `.venv\Scripts\python.exe run_pid_optimization.py --output runs/pid-new-run`. Editable constants in that runner set budgets/seed/workers; ResearchPIDConfig in line_follower/pid.py defines controller parameters. Existing output directories are rejected to prevent overwriting data. Because final test outcomes are now known, any further tuning must be labelled development and use a newly frozen untouched test cohort for a fresh confirmatory claim. Exact replication of this fixed run is not another independent trial.

Recommended future experiments, not implemented here: independent tuning/evolution seeds; explicit ablations of supervisor, temporal state and action activation; varied sensor offsets under a separate geometry protocol; hardware-calibrated dynamics; disturbance/fault studies; and a separately prespecified PPO baseline. PPO results remain unavailable. No further NEAT optimization was performed.

## Appendix: prespecified protocol

# Frozen PID development and final-assessment protocol

Prepared 29 October 2026 after NEAT preservation and before PID tuning. No NEAT training will be performed. Robot geometry, 0.35 m/s rim-speed ceiling, 600/60 Hz clocks, shared reflectance field and route/termination rules remain unchanged.

## Controller families

1. `pid_filtered`: normalized sensor-centroid PID, filtered derivative, bounded integral, error-dependent forward command, and last-seen-direction line-loss recovery.
2. `pid_corner_supervised`: the same PID plus an explicitly labelled sensor-only corner supervisor. Broad unilateral sensor patterns indicate a possible sharp corner; a short command-odometry approach is followed by counter-rotation until the central line is reacquired. A continuing central line cancels the turn at branches. No route, position, yaw, progress or track label is available to either controller. This is an assisted PID baseline, not a claim about unmodified textbook PID.

Both receive the existing 15 binary readings and dt, and share physical motor bounds with NEAT. Their temporal state and supervisor distinguish them from the preserved feedforward NEAT policies; this difference will be disclosed.

## Bounded optimization

- Fixed NumPy tuning seed 20261029, four persistent Windows-spawn CPU workers.
- At most 96 initial candidates: seeded historical-gain adapters plus randomized bounded configurations, balanced between the two families.
- Coarse evaluation on short training-only straight, mirrored 90-degree corners, the diagnostic zigzag and its mirror, and a curve.
- Keep the best six of each family; evaluate on a larger frozen training-only suite with 45-degree entries, tighter alternating turns, curves and separate generated beginner/intermediate/advanced tracks.
- Up to 24 local refinements around the best two configurations per family, using the same larger training suite. This is a fixed maximum of 120 candidates, not an indefinite search.
- Every attempted candidate and every episode is recorded, including failures and technical exceptions. Candidate ranking is lexicographic: completion fraction, mean authenticated progress, negative mean observed lateral RMSE, then negative mean successful completion time. Completion dominates error and speed.
- Bounds: normalized proportional gain 0.25-3.5, integral gain 0-0.25, derivative gain 0-0.12, forward command 0.25-0.75, minimum command 0.02-0.25, derivative-filter time 0.015-0.10 s, slowdown gain 0-5, line-loss hold 0.10-0.8 s, search command 0.2-0.65, corner approach command 0.2-0.5, corner lead distance 0-0.025 m, pivot command 0.25-0.6.

## Validation, freezing and final test

Training tracks drive all parameter exploration and refinement. Validate the top three larger-suite candidates per family on the same original 30 frozen validation tracks. Select one policy per family by the same lexicographic criteria; choose the overall PID policy from those validation results. Save exact configuration, hashes, sources, split hashes and selection criteria before loading any final-test track geometry.

After freezing, evaluate the selected PID policies and the already frozen generation-109 NEAT model on the same 90 held-out test tracks, CPU backend, identical robot/episode settings. No tuning, architecture changes or reselection follows test outcomes. Preserve failures. One evolution lineage and deterministic PID do not support population-level algorithm superiority or significance claims. Report descriptive paired-track results and identify the additional PID memory/supervision.

## Metrics and artifacts

Use the existing EpisodeResult columns plus per-episode mean absolute lateral error, policy ID, condition, seed, backend and wall time. RMSE and MAE use 60 Hz error to the authenticated local route window over observed duration; macro-averages weight tracks equally. Successful completion times condition on success. Export config/candidate/episode/validation/test CSVs and representative 60 Hz sensor, error, derivative, integral, motor-command, torque and trajectory traces. Archive a source snapshot and hardware/dependency record. PPO remains unimplemented with empty result cells.

<!-- PID-GAINS-FOLLOWUP -->
## Separate gains-only follow-up (29 October; report finalized 2 November 2026)

| Family | Gains | Completed / 90 | RMSE (mm) | MAE (mm) | Mean progress |
|---|---|---:|---:|---:|---:|
| Filtered PID | baseline | 51/90 | 30.108 | 17.712 | 0.7602 |
| Filtered PID | selected | 51/90 | 30.108 | 17.712 | 0.7602 |
| PID + corner supervision | baseline | 78/90 | 8.606 | 5.593 | 0.9218 |
| PID + corner supervision | selected | 84/90 | 6.654 | 4.479 | 0.9670 |

- Filtered PID: 0 newly completed, 0 lost completions, 51 completed by both, 39 by neither. Net change +0/90 (+0.00 percentage points). On 51 common successes, mean time is 12.429 s original versus 12.429 s selected; mean paired change +0.000 s.
- PID + corner supervision: 7 newly completed, 1 lost completions, 77 completed by both, 5 by neither. Net change +6/90 (+6.67 percentage points). On 77 common successes, mean time is 29.228 s original versus 29.519 s selected; mean paired change +0.290 s.

Filtered PID retained its original gains; its two role rows refer to the same episodes, not independent repetitions. The supervised selection changed only P/I/D.

These are paired results on a **new** frozen 90-route cohort. They must not be pooled with the original NEAT/PID table above. Only P/I/D varied; no controller logic or non-gain settings changed. The original study and raw CSVs remain intact. Full protocol, gains, validation, failures, limitations and figures: [PID_Gain_Tuning_Results.md](PID_Gain_Tuning_Results.md).
