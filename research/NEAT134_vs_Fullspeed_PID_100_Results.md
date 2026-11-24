# Corrected comparison: full-cruise PID versus frozen NEAT generation 134

Recorded 19 November 2026. This comparison corrects the earlier PID cruise restriction. The earlier [restricted-cruise comparison](NEAT134_vs_Tuned_PID_100_Results.md) remains archived with its original data and is not evidence against a full-speed-capable PID.

## Main result

**Both controllers use the same 1.00 m/s wheel-speed cap, and PID now requests the full target on a centered straight.** The retuned PID has base exactly **1.00** and measured minimum sustained straight speed **0.991 m/s** across 9/9 dedicated test straights. Frozen NEAT generation 134 retains its learned throttle and sustained at least **0.972 m/s** on the same straights.

On **150 fresh shared test courses**, NEAT completed **142/150** and PID completed **146/150**. On the **139 shared successful courses**, mean finish times were **5.338 s for NEAT** and **6.923 s for PID**. NEAT's mean shared finish time was **22.89% shorter**.

The paired relative time change, NEAT minus PID, is **-22.89%**, with a 95% paired-bootstrap interval of **[-23.83%, -21.94%]**. NEAT was quicker on 119 shared courses and PID on 19, with 1 ties.

NEAT alone finished 3 courses, PID alone 7, and both failed 1. The paired completion-rate difference is **-2.67 percentage points**, 95% interval **[-6.67, +1.33]**. Exact two-sided McNemar **p=0.3438**.

| Metric | NEAT | Full-cruise PID |
| --- | ---: | ---: |
| Mean route progress, all episodes | 97.48% | 99.02% |
| Geometric RMSE, all episodes (mm) | 19.089 | 24.635 |
| Geometric RMSE, shared finishes (mm) | 15.189 | 22.244 |
| Command jitter, all episodes | 0.007505 | 0.001173 |
| Command jitter, shared finishes | 0.007401 | 0.001120 |
| Mean episode forward speed, all episodes (m/s) | 0.710 | 0.579 |
| Peak measured forward speed (m/s) | 1.002 | 1.003 |
| Minimum central-2 m straight speed (m/s) | 0.972 | 0.991 |
| Shared-speed-course yaw oscillation RMS (rad/s) | 0.017951 | 0.001058 |

Failures: NEAT `{"stalled": 8}`; PID `{"off_route": 3, "stalled": 1}`.

| Course group | NEAT completion | PID completion | Shared finishes | NEAT mean time (s) | PID mean time (s) | NEAT RMSE (mm) | PID RMSE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed | 87/90 | 88/90 | 85 | 5.565 | 7.922 | 11.284 | 21.266 |
| speed | 29/30 | 30/30 | 29 | 6.975 | 7.006 | 3.310 | 5.641 |
| corner | 26/30 | 28/30 | 25 | 2.669 | 3.429 | 42.247 | 44.828 |

Group time and RMSE use each group's shared successes. Completion/progress and all-episode quality include every attempt. No failures were dropped from the completion analysis.

## Why this corrects the earlier design

The previous PID had base 0.75, which imposed a nominal straight-line request of 0.75 m/s under a 1.00 m/s cap. Retaining that setting during the smoothing refinement restricted its cruise speed. It was inappropriate to treat the earlier finish-time difference as evidence of NEAT outperforming a full-speed-capable PID.

The new design fixes base at **1.00 for every candidate**, then tunes P, I, D, derivative filtering, and error-dependent corner slowdown around that full-cruise setting. Both controllers may use the full cap on straights and reduce commands for turns. Their actual speeds are measured. A standalone 6 m feasibility probe measured **0.990512 m/s** for PID's central 2 m at this cap, compared with **0.742884 m/s** for the old base-0.75 configuration.

This preserves identical robot physics, 15 physical binary sensors, control/physics rates, actuator limits, route evaluator, 100 mm corridor, two-second departure allowance without sensor contact, five-second stall check, and timeout formula. Controller algorithms and recovery rules are unchanged. NEAT-Python remains stock version 0.92. A small measured velocity overshoot can occur in the actuator dynamics; the cap limits requested wheel-speed targets.

## PID tuning and independent selection

The full-speed PID was selected on **50 development courses and 60 separate validation courses**. The search evaluated 196 configurations, including the old restricted-cruise reference. That old reference was ineligible for selection. Full-speed candidates had to complete every dedicated validation straight with central-2 m speed at least **0.95 m/s**.

Selection maximized validation completion. Among those candidates, policies within 2% of the fastest failure-penalized mean episode time were shortlisted; the smoothest steady-section yaw response was selected, with command jitter, RMSE, and time as tie-breaks. Failures received their configured timeout for the selection time cost. This rule was documented before the new test. Candidate **gain-005** was frozen before new shared test generation.

| Configuration | Validation completions | Minimum straight speed (m/s) | Failure-penalized mean time (s) |
| --- | ---: | ---: | ---: |
| previous-base-075 | 58/60 | 0.743 | 8.024 |
| full-cruise-reference | 37/60 | 0.991 | 21.645 |
| gain-005 | 59/60 | 0.991 | 7.889 |

Raising base alone, without retuning, completed **37/60 validation courses**. The selected full-cruise retune completed **59/60**. This development observation documents the need to retune steering and slowdown when increasing cruise speed; it is not an independent post-selection test result.

| Parameter | Earlier restricted-cruise tune | New full-cruise tune |
| --- | ---: | ---: |
| base | 0.750000000 | 1.000000000 |
| kp | 0.390665196 | 0.332065416 |
| ki | 0.087212318 | 0.087212318 |
| kd | 0.050498472 | 0.030299083 |
| derivative_filter_s | 0.080000000 | 0.100000000 |
| slowdown_gain | 0.567546418 | 4.000000000 |

Other PID parameters, including remembered-side recovery, remain fixed. The new shared test consists of 90 mixed tracks (30 per difficulty), 30 speed courses, and 30 corner drills, with geometry disjoint from known earlier suites and training batches. The original five-seed NEAT final test set remains untouched.

The selected PID's stronger error-dependent slowdown reduces its speed during bends and corner recovery. Its straight-line speed is unrestricted by a sub-unit base. Shared-course time differences therefore include the policies' speed scheduling and cornering strategies, rather than demonstrating an effect of neural steering alone.

## Interpretation for the paper

This is a comparison of **two fixed resulting controllers with equal speed opportunity**, evaluated on unseen shared geometry. The NEAT checkpoint is unchanged from the earlier comparison. No controller or setting was selected from these new test outcomes.

The relative-time interval uses 20,000 paired bootstrap resamples stratified by course group and difficulty. The completion test uses paired success/failure outcomes. These estimates describe track variation for this fixed controller pair, rather than variation across independently evolved NEAT seeds. Algorithm-wide claims still require replication. Higher speed alone does not establish better stability or completion.

The new and earlier comparisons use different fresh track suites, so their headline percentages should not be treated as a matched causal ablation of the base parameter alone. The earlier result remains historical evidence about its explicitly restricted-cruise configuration.

## Frozen controllers and source

Canonical folder: **`runs/neat134-vs-pid100-fullspeed-20261119/frozen/`**.

- [Unchanged NEAT generation 134](../runs/neat134-vs-pid100-fullspeed-20261119/frozen/neat-generation-0134.pkl), SHA-256 `6114a6859544fec99c26dad124d7ac44d9b0e077a43102532f5fca02114b4314`.
- [Full-cruise PID parameters](../runs/neat134-vs-pid100-fullspeed-20261119/frozen/pid-filtered-100.json), SHA-256 `c6c68b53342600db0078e991bbb729064d829d360c4840a720dbf5e66d94c24b`.
- [Freeze manifest, original source paths, robot/episode settings, and software identities](../runs/neat134-vs-pid100-fullspeed-20261119/frozen/manifest.json).
- [Pre-test PID selection record](../runs/pid-fullspeed-100-20261119/selection-freeze.json), [tuning protocol](../runs/pid-fullspeed-100-20261119/protocol.json), and [shared-test protocol](../runs/neat134-vs-pid100-fullspeed-20261119/protocol.json).
- [Evaluation source snapshot](../runs/neat134-vs-pid100-fullspeed-20261119/source.zip) and [declared methodology/correction](PID_Fullspeed_100_Protocol.md).

Original frozen models, old measurements, and the historical comparison are preserved.

## Terminal replay and reproduction

```powershell
$py = ".\.venv\Scripts\python.exe"

# Ten prespecified shared test tracks; starts with NEAT
& $py replay_neat_pid_100.py --run runs/neat134-vs-pid100-fullspeed-20261119

# Verify/reuse the completed full-speed study
& $py run_pid_fullspeed_100.py all

# Regenerate this report and figures from saved results
& $py runs/neat134-vs-pid100-fullspeed-20261119/analyze_results.py
```

Press **C** for the exact frozen PID/NEAT switch, **N/P** for next/previous track, **R** to reset, **Space** to pause, and **T** for the route. The terminal prints the shared cap and actual PID base/nominal straight target. The replay retains each original physical sensor mask.

Replay identity was checked by counting calls to the real controller methods while switching controllers and tracks. When PID is displayed, only `ResearchPIDController.act()` runs. The replay now prints the active controller class and frozen filename. A separate complete straight-track audit reproduced the saved measurements: PID made 323 control calls with zero NEAT calls, and NEAT made 330 with zero PID calls. See the [audit results](../runs/neat134-vs-pid100-fullspeed-20261119/replay-controller-audit.json) and [reproducible audit script](../runs/neat134-vs-pid100-fullspeed-20261119/audit_replay_controllers.py).

## Exploratory long course

These observations are separate from the fresh primary suite and were not used for selection.

| Policy | Completed | Progress | Finish time (s) | Mean forward speed (m/s) | Peak forward speed (m/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| neat | True | 100.0% | 26.233 | 0.796 | 0.983 |
| pid | True | 100.0% | 27.233 | 0.767 | 1.003 |

## Raw evidence and figures

- [Structured paired results and confidence intervals](../runs/neat134-vs-pid100-fullspeed-20261119/results/summary.json), [all raw episodes](../runs/neat134-vs-pid100-fullspeed-20261119/results/episodes.csv), and [six-metre feasibility probe](../runs/pid-fullspeed-100-20261119/feasibility.json).
- [Primary comparison, PNG](../runs/neat134-vs-pid100-fullspeed-20261119/plots/01-fullspeed-comparison.png), [SVG](../runs/neat134-vs-pid100-fullspeed-20261119/plots/01-fullspeed-comparison.svg).
- [Feasible straight speed and paired times, PNG](../runs/neat134-vs-pid100-fullspeed-20261119/plots/02-feasible-speed-and-paired-time.png), [SVG](../runs/neat134-vs-pid100-fullspeed-20261119/plots/02-feasible-speed-and-paired-time.svg).
