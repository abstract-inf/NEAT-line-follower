# Common research comparison: NEAT, PID and potentially PPO

Initial comparison finalized 29 October 2026; separate gains-only follow-up documented 2 November 2026. This is a descriptive comparison of frozen implementations, not evidence of universal algorithm superiority.

## Planned and executed methodology

All evaluated policies faced the **same 90 held-out tracks**, with identical hashes, 30 per difficulty, unchanged robot geometry, 0.35 m/s motor-rim speed ceiling, 0.03 N m torque limits, 600/60 Hz physics/control, binary reflectance and ordered-route evaluator. CPU MuJoCo is the common assessment backend. The historical NEAT GPU validation results remain separate. Neither validation nor training rows are pooled into this test table.

NEAT g109 was fixed before PID tuning. PID trained/tuned only on dedicated training tracks, selected on the same 30 validation tracks, and froze exact parameters and source before test geometry was loaded by the pipeline. After test outcomes, there was no tuning or reselection. Protocol and freeze are in research/PID_Protocol.md and runs/pid-research-20261029/selection-freeze.json. The selected overall PID is pid-111 (with corner supervision); filtered PID is separately shown.

## Actual common held-out results

| Controller | Test completion | Mean RMSE (mm) | Mean MAE (mm) |
|---|---:|---:|---:|
| Filtered PID | 43/90 (47.78%) | 35.932 | 21.597 |
| PID + corner supervision | 76/90 (84.44%) | 8.805 | 5.643 |
| Frozen NEAT g109 | 35/90 (38.89%) | 26.525 | 18.170 |
| PPO | Not evaluated | Not available | Not available |

Macro error averages include failures over observed duration. Successful completion times and difficulty-stratified tables are in PID_Experimental_Results.md. The same per-episode columns, including directly measured MAE, are available in runs/pid-research-20261029/test-episodes.csv for Pandas/paired-track analysis. No p-value or statistical significance is claimed.

![Common assessment](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/pid/common-test.png>)

Paired completion contingency counts, without treating repeated telemetry as independent:

```json
[
  {
    "condition": "pid_filtered",
    "both_complete": 34,
    "pid_only_complete": 9,
    "neat_only_complete": 1,
    "neither_complete": 46
  },
  {
    "condition": "pid_corner_supervised",
    "both_complete": 34,
    "pid_only_complete": 42,
    "neat_only_complete": 1,
    "neither_complete": 13
  }
]
```

## Scope and fairness limits

Controller equations, exact supervisor logic, observable information, telemetry interpretation, and recommended manuscript terminology are documented in [PID_Controller_Methodology.md](PID_Controller_Methodology.md).

Physical limits and physical sensors match. Internal controller classes do not: filtered PID has temporal derivative/integral/recovery state; the supervised variant additionally encodes corner-detection, approximate command odometry and pivot logic. The selected NEAT model is memoryless with sigmoid motor outputs and cannot command the same reverse manoeuvres, although the simulator interface permits them. The strong supervised result cannot be attributed to unmodified textbook PID. Training budgets and methods are not matched, and only one NEAT evolution lineage and one PID tuning seed exist. Results compare these implementations under the documented conditions, not NEAT/PID algorithm families in general.

The historical parent/continuation phases use different speed/timeout/curriculum conditions and are documented separately in NEAT_Experimental_Results.md. Current final-test failures remain in the datasets. Unseen generated routes are tested; disturbances, sensor faults, motor degradation and real-world transfer are not. There is no basis for corresponding robustness claims.

## PID results

Bounded search completed with 120 configurations, 6 validation candidates and two frozen variants. See PID_Experimental_Results.md for equations, selected gains, all failure reasons, traces, training-stage plots and protocol. No further tuning was part of that initial assessment; the separate follow-up is below.

## PPO results

No PPO implementation, training or assessment was performed. Result cells are intentionally unavailable. A future PPO experiment must prespecify observations, actions, reward, training/selection budget and independent seeds. Now-known test outcomes must not drive PPO tuning; a new unopened confirmatory cohort may be needed for a stronger future comparison.

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

<!-- FROZEN-NEAT-NEW90 -->
## Subsequent frozen NEAT assessment on the new cohort — 2 November 2026

| Frozen policy | Completed / 90 | Rate | RMSE (mm) | MAE (mm) | Mean progress |
|---|---:|---:|---:|---:|---:|
| Filtered PID | 51/90 | 56.67% | 30.108 | 17.712 | 0.7602 |
| PID + corner supervision, original gains | 78/90 | 86.67% | 8.606 | 5.593 | 0.9218 |
| PID + corner supervision, tuned gains | 84/90 | 93.33% | 6.654 | 4.479 | 0.9670 |
| Frozen NEAT g109 | 38/90 | 42.22% | 23.194 | 15.308 | 0.6814 |

| Frozen policy | Beginner | Intermediate | Advanced |
|---|---:|---:|---:|
| Filtered PID | 30/30 | 11/30 | 10/30 |
| PID + corner supervision, original gains | 30/30 | 29/30 | 19/30 |
| PID + corner supervision, tuned gains | 30/30 | 29/30 | 25/30 |
| Frozen NEAT g109 | 30/30 | 5/30 | 3/30 |

All rows use identical new-cohort track hashes, CPU physics and physical/episode limits. NEAT g109 is unchanged; this adds inference results, not training or reselection. The original 90-track comparison remains separate. See [NEAT_New_Cohort_Evaluation.md](NEAT_New_Cohort_Evaluation.md) for model/suite hashes, failures, paired comparisons and limitations.
