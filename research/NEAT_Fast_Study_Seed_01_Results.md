# Faster NEAT driving: seed 1 results

Historical snapshot from 19 November 2026. All five runs subsequently completed; see the [five-seed results and fresh PID comparison](NEAT_Fast_Five_Seed_Results.md) for the current study-wide assessment. The original seed-1 measurements below are preserved.

**Recorded:** 19 November 2026  
**Experiment:** `neat-fast-study-v1`, fitness `route-speed-1.0`  
**Status:** Seed 1 completed all 300 generations in approximately 4 hours 15 minutes. Seeds 2–5 have not run. These are validation results; final test evaluation has not been performed.

## Main finding

The new run produced controllers that sustained approximately 1 m/s on long straights and completed every validation course. It did **not** produce a final-stage controller that simultaneously preserved the original ramp reference's tracking accuracy and command smoothness. The recorded final winner is therefore `null`.

Generation 134 is a promising **1 m/s controller**: at a 1.00 m/s wheel-speed cap, it sustained at least **0.972 m/s on all six dedicated straights** and completed **60/60 validation courses**. Its tracking error and command jitter exceeded the reference limits. Subsequent user replay feedback described it as very fast and apparently reliable; that qualitative observation is recorded below.

Generation 258 is the recorded final-stage diagnostic trade-off: at the primary 1.50 m/s cap, it completed **60/60**, but sustained only **0.638 m/s** on the dedicated straights and failed the smoothness and shared-finish tracking checks. It remains a diagnostic checkpoint rather than an accepted winner.

## Experiment and measurements

The run started from seed 1's saved 0.50 m/s winner, with a target population of 100, six shared training courses per generation, and eight CPU workers. Caps increased through 0.65, 0.80, 1.00, and 1.25 m/s for 50 generations each, followed by 100 generations at 1.50 m/s. The stock NEAT-Python 0.92 evaluator was retained.

Every generation's champion was evaluated on the same 60 validation courses: 30 mixed, 20 speed, and 10 corner courses. The corresponding original ramp winner was the reference, evaluated at a 1.50 m/s cap.

- **Sustained straight speed:** speed over the central 2 m of each dedicated straight. The table reports the minimum across all six straights.
- **Geometric RMSE:** distance from the chassis center to the intended route polyline, calculated per episode and then averaged across episodes.
- **Command jitter:** normalized mean squared changes in left and right wheel commands. It measures command smoothness, not physical chassis vibration directly.
- **Paired finish time:** compared only on courses both controllers completed, with that denominator reported.

The approximate 1 m/s target requires at least 0.95 m/s on every dedicated straight. Wheel-speed cap, sustained straight speed, and average speed over complete courses are distinct quantities.

## Representative checkpoints

Generation numbers are zero-based. Each checkpoint below uses its recorded cap; the original reference always uses 1.50 m/s.

| Controller | Cap (m/s) | Completed | Minimum sustained straight (m/s) | RMSE, all episodes (mm) | Command jitter, all episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original ramp reference | 1.50 | 59/60 | 0.520 | 11.795 | 0.002526 |
| Generation 31 | 0.65 | 60/60 | 0.644 | 2.727 | 0.002355 |
| Generation 134 | 1.00 | 60/60 | 0.972 | 14.647 | 0.006246 |
| Generation 150 | 1.25 | 52/60 | 1.091 | 21.608 | 0.008813 |
| Generation 258 | 1.50 | 60/60 | 0.638 | 7.228 | 0.006348 |
| Generation 299 | 1.50 | 51/60 | 0.710 | 26.656 | 0.016078 |

Generation 150 demonstrates sustained travel above 1 m/s, but its lower completion count shows the accompanying loss in reliability. Generation 299 also illustrates why the final generation should not automatically be treated as the strongest controller.

| Candidate | Shared finishes | Reference mean finish time (s) | Candidate mean finish time (s) | Relative change | Reference shared RMSE (mm) | Candidate shared RMSE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generation 134, 1.00 m/s cap | 59 | 8.028 | 5.597 | −30.3% | 5.346 | 14.616 |
| Generation 258, 1.50 m/s cap | 59 | 8.028 | 6.889 | −14.2% | 5.346 | 7.164 |

The generation 134 comparison is exploratory and uses a different cap from the reference. Generation 258 and the reference share the primary 1.50 m/s cap. These comparisons do not include a newly evaluated PID baseline.

## User replay observation: generation 134

On **19 November 2026**, the user reported replaying generation 134 using the supplied visualization command and described it as very fast, impressive, and apparently reliable, despite its higher RMSE. This is user-reported visual evidence; no additional structured per-track outcomes or replay counts were supplied.

The observation is consistent with the checkpoint's recorded **60/60 validation completions** and **0.972 m/s minimum sustained straight speed**. Generation 134 should remain a priority candidate for focused evaluation at the 1.00 m/s cap.

Completion reliability, centerline accuracy, command smoothness, and visual driving quality describe different aspects of behavior. A controller can complete courses reliably while following the line with a larger lateral offset or making larger command corrections. Higher RMSE or command jitter alone does not establish poor completion reliability, and command jitter does not directly measure visible chassis vibration.

Generation 134 exceeded the reference's accuracy and smoothness limits and was outside the final-stage selection window. Those facts preserve the original protocol outcome; they do not erase its successful speed and completion results. Further evaluation should investigate whether its observed driving quality and completion performance persist on fresh evaluation geometry and against the saved 1.00 m/s PID tune under a separately declared comparison.

Saved candidate: [generation-0134.pkl](../runs/neat-fast-study/seed-01/training/champions/generation-0134.pkl). The replay feedback adds no new numerical completion, RMSE, timing, or PID-comparison result.

## What the quality checks require

Only champions from generations **200–299**, evaluated at the 1.50 m/s cap, can become the formal final winner. Each must satisfy all of the following relative to its reference:

1. Complete at least **29/30 mixed**, **20/20 speed**, and **10/10 corner** courses. Completion is protected separately within each group.
2. Match or improve mean geometric RMSE across all 60 episodes and, separately, on courses both controllers finish.
3. Match or improve mean command jitter across all 60 episodes and, separately, on courses both controllers finish.

These are non-regression checks with a numerical tolerance of `1e-12`. The sustained-speed goal is recorded separately. Faster finish times do not compensate for a failed quality check.

**None of the 100 final-stage champions passed every quality check.** All 100 exceeded the reference's overall command jitter, shared-finish command jitter, and shared-finish tracking RMSE. Fifteen nevertheless completed all 60 validation courses.

Generation 258's all-episode RMSE improved from 11.795 to 7.228 mm, but its shared-finish RMSE worsened from 5.346 to 7.164 mm. The reference's failed course inflates its all-episode mean. Checking shared successful courses prevents that failure from concealing a regression on the courses both controllers finish. Generation 258's overall command jitter was approximately **2.51 times** the reference's.

Eight earlier champions passed the quality checks, all at the 0.65 m/s cap. They were outside the prespecified final selection phase. No final-stage champion reached the 0.95 m/s threshold on every straight; the highest minimum was **0.947 m/s at generation 201**, which completed 49/60 courses.

## Trend across speed phases

| Cap (m/s) | Generations | Mean completed courses, out of 60 | Champions completing 60/60 | Champions passing quality checks |
| --- | --- | ---: | ---: | ---: |
| 0.65 | 0–49 | 59.86 | 45/50 | 8/50 |
| 0.80 | 50–99 | 59.80 | 41/50 | 0/50 |
| 1.00 | 100–149 | 59.20 | 33/50 | 0/50 |
| 1.25 | 150–199 | 56.68 | 12/50 | 0/50 |
| 1.50 | 200–299 | 55.60 | 15/100 | 0/100 |

These are descriptive averages across champions in one lineage, not independent experimental repetitions. The curves show declining completion and increasing command jitter at the higher caps. Only three champions reached the straight-speed target: generations 101, 134, and 150; none also passed the quality checks.

Both drive and steering were evaluated in all 300 champions. The skipped-drive-output behavior observed in the previous ramp references does not explain this run's results.

## Interpretation and follow-up

The observations support a speed/quality trade-off. The present training fitness rewards successful speed while applying a relatively small penalty to measured command jitter. Geometric RMSE is protected during validation selection, without an explicit training term.

For generation 134 on the 59 shared finishes, the mean speed-bonus gain was **0.102525**, while the additional jitter penalty was only **0.000759**. This illustrates the scale of those two terms and suggests that stronger stability pressure during evolution deserves investigation. It does not isolate the causal effect of any individual experimental change.

Preserve these checkpoints. The user's positive replay observation strengthens the case for focused follow-up of generation 134 at 1.00 m/s, with completion, finish time, tracking accuracy, and physical driving behavior reported separately. Any revised reward, stability constraint, or continuation schedule should receive a separate protocol version. Additional seeds under the current protocol would assess reproducibility of the current approach.

These results do not yet establish that NEAT outperforms tuned PID, that improvements generalize to fresh test courses, or that later generations are inherently worse. The study combines changes in course mixture, training budget, and reward; attributing improvements to a particular change requires a matched-budget control.

## Evidence and replay

- [Protocol, assumptions, and replay commands](../docs/fast-neat-study.md).
- [Speed and quality curves, SVG](../runs/neat-fast-study/review-seed-01/seed-01-speed-and-quality.svg) and [PNG](../runs/neat-fast-study/review-seed-01/seed-01-speed-and-quality.png).
- [Per-generation validation metrics](../runs/neat-fast-study/review-seed-01/validation-review.csv), [phase summaries](../runs/neat-fast-study/review-seed-01/phase-summary.csv), and [structured findings](../runs/neat-fast-study/review-seed-01/summary.json).
- [Original selection record](../runs/neat-fast-study/seed-01/training/selection.json), [quality decisions](../runs/neat-fast-study/seed-01/training/quality.csv), and [prepared reference measurements](../runs/neat-fast-study/prepared/references/seed-01.json).
- [Analysis script](../runs/neat-fast-study/review-seed-01/analyze.py), which regenerates the detailed review without evaluating new policies or opening test-course files.

No new evolution, policy tuning, or fresh test evaluation was performed when writing this summary. The user-reported replay observation was added on 19 November 2026; it is qualitative and does not alter the recorded numerical results or formal winner selection.

Subsequent work is documented in the [fresh frozen-controller comparison against the retuned 1.00 m/s PID](NEAT134_vs_Tuned_PID_100_Results.md). That separate report contains new shared-track measurements and does not change this original validation record.
