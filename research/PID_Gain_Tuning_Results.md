# PID gains-only follow-up results

Experiment conducted 29 October 2026; analysis and documentation finalized
2 November 2026. This is a separate experiment after
the initial PID/NEAT comparison; its original raw records and archive remain
unchanged. The question is whether **only P, I, D tuning** can improve either
fixed PID variant. No new NEAT training or selection was performed.

## Protocol and implementation boundary

The prespecified protocol is research/PID_Gain_Tuning_Protocol.md. Exactly
368 distinct gain configurations were evaluated. The initial 320
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
balanced by difficulty, generated with suite seed 2026102902 using the
unchanged version-1 generator distribution. New procedural geometry excludes
known historical routes by canonical hash, including mirrored equivalents.
Handcrafted mirrors intentionally coexist within training. These tests concern
new routes from a familiar generator, not generator-family extrapolation.

Selection is lexicographic completion fraction, mean authenticated progress,
negative observed RMSE, negative successful time. Exact configurations, source,
runner, protocol and suite hashes were frozen at **2026-10-29T08:48:31.126053+00:00** before
test geometry was loaded for evaluation. No reselection or tuning followed
outcomes. The original 90 tracks are only a secondary, already-known benchmark.

Episode allocations: `{"coarse": 4800, "expanded": 1080, "legacy-test": 270, "refine": 2160, "representative": 27, "test": 270, "validation": 240}`. In addition,
15 fixed-policy trace replays are labelled
separately, never counted as independent final-test observations. Four persistent
Windows CPU workers ran without real-time pacing. Total wall time, including
track preparation, selection, assessment and trace replays, was
**1766.941 s (29.449 min)**.
There are 8,862 recorded
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

| Family | Gains | P | I | D |
|---|---|---:|---:|---:|
| Filtered PID | baseline | 0.723985005 | 0.168134489 | 0.021582282 |
| Filtered PID | selected | 0.723985005 | 0.168134489 | 0.021582282 |
| PID + corner supervision | baseline | 1.386243657 | 0.114380626 | 0.084255258 |
| PID + corner supervision | selected | 1.278230244 | 0.035193151 | 0.005676495 |


Complete configuration files are in runs/pid-gains-20261029/selected-*.json and
best.json. The overall best is chosen on validation, not this final test.
The filtered selection retained gain-000, so its baseline/selected table rows
refer to the same 90 episodes, not two independent runs. Only three distinct
policies were assessed on each final cohort.

## Primary result: the same 90 new held-out tracks

- Filtered PID: validation retained the original gains; this bounded search did not supply a selected replacement. This is an observed cohort result, not a significance claim.
- PID + corner supervision: selected gains completed 6 more new-test routes than the original gains. This is an observed cohort result, not a significance claim.

| Family | Gains | Completed / 90 | RMSE (mm) | MAE (mm) | Mean progress |
|---|---|---:|---:|---:|---:|
| Filtered PID | baseline | 51/90 | 30.108 | 17.712 | 0.7602 |
| Filtered PID | selected | 51/90 | 30.108 | 17.712 | 0.7602 |
| PID + corner supervision | baseline | 78/90 | 8.606 | 5.593 | 0.9218 |
| PID + corner supervision | selected | 84/90 | 6.654 | 4.479 | 0.9670 |


- Filtered PID: 0 newly completed, 0 lost completions, 51 completed by both, 39 by neither. Net change +0/90 (+0.00 percentage points). On 51 common successes, mean time is 12.429 s original versus 12.429 s selected; mean paired change +0.000 s.
- PID + corner supervision: 7 newly completed, 1 lost completions, 77 completed by both, 5 by neither. Net change +6/90 (+6.67 percentage points). On 77 common successes, mean time is 29.228 s original versus 29.519 s selected; mean paired change +0.290 s.

Errors are macro-averages of per-episode observed-duration metrics and include
failures. MAE is accumulated directly. Common-success time differences are
paired on identical successfully completed tracks; differing success subsets
must not be compared as evidence of a speed improvement.

| Family | Gains | Difficulty | Completed / n |
|---|---|---|---:|
| Filtered PID | baseline | beginner | 30/30 |
| Filtered PID | baseline | intermediate | 11/30 |
| Filtered PID | baseline | advanced | 10/30 |
| Filtered PID | selected | beginner | 30/30 |
| Filtered PID | selected | intermediate | 11/30 |
| Filtered PID | selected | advanced | 10/30 |
| PID + corner supervision | baseline | beginner | 30/30 |
| PID + corner supervision | baseline | intermediate | 29/30 |
| PID + corner supervision | baseline | advanced | 19/30 |
| PID + corner supervision | selected | beginner | 30/30 |
| PID + corner supervision | selected | intermediate | 29/30 |
| PID + corner supervision | selected | advanced | 25/30 |


![new-test-comparison](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid-gains/new-test-comparison.png>)

## Search and validation

| Family | Gains | 45-case training completion | 30-route validation completion |
|---|---|---:|---:|
| Filtered PID | baseline | 22/45 | 22/30 |
| Filtered PID | selected | 22/45 | 22/30 |
| PID + corner supervision | baseline | 40/45 | 28/30 |
| PID + corner supervision | selected | 41/45 | 28/30 |


Both supervised policies completed 28/30 validation routes. The selected gains
won the prespecified progress tie-break: mean valid progress was
0.972243
for the original gains and
0.982165
for the selected gains. Test outcomes did not decide this choice. Retaining the
filtered baseline is not proof that its gains are globally optimal: some other
settings improved training completion, but were not chosen by validation.

![gain-search](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid-gains/gain-search.png>)

![gain-sensitivity](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid-gains/gain-sensitivity.png>)

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

| Family | Gains | Known original cohort completion |
|---|---|---:|
| Filtered PID | baseline | 43/90 |
| Filtered PID | selected | 43/90 |
| PID + corner supervision | baseline | 76/90 |
| PID + corner supervision | selected | 78/90 |


This is the previously opened cohort from the original study. Its outcomes were
not used in this gain search or selection. Replaying it helps assess continuity,
but cannot restore its status as an untouched confirmatory test. The original
NEAT numbers must not be compared with the new-cohort PID numbers as if the
tracks were identical. NEAT was not evaluated during this gains-only phase.
A subsequent assessment of the unchanged frozen NEAT model on this same new
cohort is now documented in [NEAT_New_Cohort_Evaluation.md](NEAT_New_Cohort_Evaluation.md).

## Representative trajectories

![diagnostic-trajectories](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid-gains/diagnostic-trajectories.png>)

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
Source SHA256: **78acab6866de9a70228f5c8e6b3e84a74592c0d00290927a0c87ccdc5909da8d**. Runner SHA256:
**17f1cad24fc7aa27aba8f626bc2451364f1a4f4d51f127c1453f525e67f1a396**. Suite SHA256: **16399ef0aa1d6d07ac5779b2126cd597cd3a006e1b943c4c8cce9d89a0922ef7**.
No Git commit is available; snapshots and hashes identify the code.
SHA256 checks verified **531 historical files
unchanged**, including the original NEAT CSVs and completed initial PID study.
Additional geometry, replay-consistency and frozen-selection audit details are
saved in research/artifacts/pid-gains/integrity.json.

Editable controls are at the top of run_pid_gain_tuning.py. A fresh output is
mandatory: `.venv\Scripts\python.exe run_pid_gain_tuning.py --output runs/pid-gains-replication`.
The historical exclusion registry can grow between invocations, so a seed alone
does not reproduce the accepted geometry. Exact reproduction requires the saved
source and accepted training/suite manifests (and their JSON geometries). Fixed
policy replays on those saved tracks are reproductions, not independent evidence.
Use a new prespecified suite seed for another fresh assessment and include any
newly opened suite geometry in the exclusion registry. Analyze saved results with
`.venv\Scripts\python.exe research/analyze_pid_gains.py` (the analysis script's
RUN constant selects the completed run). Do not use now-known test outcomes for
further tuning without declaring a new development phase and untouched cohort.
