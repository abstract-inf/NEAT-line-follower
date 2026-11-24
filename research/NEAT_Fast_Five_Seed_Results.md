# Five-seed faster NEAT: completed training and PID comparison

Recorded **21 November 2026**. All five fast-stage lineages completed 300 generations; each began from its own earlier 150-generation 0.50 m/s winner. No evolution or PID tuning was performed for this review.

## Main results at a glance

- **Finish-time gains repeat across all five seeds:** the 1 m/s validation selections finish shared successful courses 20.7–22.3% sooner than the frozen full-speed PID on a fresh shared suite.
- **Tracking versus smoothness trade-off:** shared-finish RMSE is lower in all five, but command jitter is higher in all five. These are different aspects of controller quality.
- **Completion superiority is not established:** NEAT completes 140–149/150 versus PID 145/150. All five exact McNemar comparisons have p>0.05; this does not prove equivalence either.
- **The original protected-quality speed goal remains unmet:** only one final-stage controller qualifies, and none of the 500 final-stage champions meets the every-straight speed threshold.

## Findings from all five completed training runs

The logs contain **1,500 committed fast-stage generations**, **990,294 evaluated episode slots**, and **90,000 champion validation episodes**. Actual populations fluctuate around the target of 100. Counts exclude abandoned work repeated after crashes; generation-time sums exclude downtime and failed partial evaluations.

**Only one of five runs produced a formal final-stage quality-qualified winner, seed 3 generation 216. None combined the final quality gate with the sustained-speed goal.** This is not evidence of a consistent protected-quality speed improvement. Separately, the same validation rule identifies a 1.00 m/s checkpoint completing all 60 validation courses in every lineage; that repeatable behavior motivates the independent follow-up below.

| Seed | Formal winner | Final-phase quality passes /100 | 60/60 champions /100 | Straight-goal champions /100 |
| --- | --- | --- | --- | --- |
| 1 | None | 0 | 15 | 0 |
| 2 | None | 0 | 2 | 0 |
| 3 | g216 | 1 | 12 | 0 |
| 4 | None | 0 | 1 | 0 |
| 5 | None | 0 | 3 | 0 |

The final window is generations 200–299 at the 1.50 m/s wheel-speed cap. Gate checks protect completion separately within mixed/speed/corner groups and mean geometric RMSE and command jitter both across all observed episodes and on shared finishes, relative to that seed's old ramp winner. A cap is an allowed wheel-speed target, not an actual chassis speed. Straight goal means at least 0.95 m/s over the central 2 m on **every** dedicated validation straight; it is distinct from the quality gate.

These quality gates compare with the **old ramp NEAT reference, not PID**. Missing a gate does not establish that a candidate is worse than PID; the separate shared-track comparisons answer that question.

Seed 3's qualified generation 216 completed 60/60 but its paired mean finish time was **0.07194 s slower** than its reference. Its drive output was not evaluated by the unchanged NEAT-Python 0.92 network evaluator. In our existing action map, a zero drive output implies normalized forward command 0.35 before steering/clipping; at the 1.50 m/s cap the nominal straight request is 0.525 m/s. This explains a conservative route to satisfying quality constraints; a quality pass does not imply the speed goal. It is implementation behavior we retained and logged, not proof of a defect in the NEAT algorithm.

## Reproducibility of the promising 1 m/s checkpoints

Before opening the new follow-up test, we selected one champion per seed from generations 100–149 by maximum validation completion, then validation fitness, retaining the earliest tie. Selection uses the same 60 validation tracks, never test outcomes. This rule reproduces seed 1 generation 134. These are **exploratory validation-selected checkpoints**, not replacements for the original formal winners.

| Seed | Generation | Validation finishes | Min straight m/s | RMSE mm | Command jitter | Nodes / enabled edges |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 134 | 60/60 | 0.972 | 14.65 | 0.006246 | 5 / 26 |
| 2 | 136 | 60/60 | 0.858 | 12.79 | 0.005826 | 3 / 21 |
| 3 | 131 | 60/60 | 0.914 | 12.97 | 0.011484 | 8 / 37 |
| 4 | 107 | 60/60 | 0.905 | 13.14 | 0.004474 | 6 / 30 |
| 5 | 134 | 60/60 | 0.923 | 14.79 | 0.005529 | 6 / 34 |

All five 1 m/s selections evaluate their drive output, and all exceed their respective original reference's RMSE/jitter quality limits. Visual reliability, completion, geometric accuracy, and command smoothness describe different properties. The saved PID comparator is the newer full-speed filtered PID with error-based speed control and line-loss recovery, with base 1.00; its original tuning is unchanged.

## Curriculum effects and stopping-budget observations

| Cap m/s | Mean validation finishes /60 | Range of seed means | Perfect champions | Straight-goal champions |
| --- | --- | --- | --- | --- |
| 0.65 | 59.32 | 58.78–60.00 | 182/250 | 0 |
| 0.8 | 59.62 | 59.14–59.92 | 185/250 | 0 |
| 1.0 | 58.30 | 57.36–59.20 | 100/250 | 20 |
| 1.25 | 56.64 | 55.94–57.86 | 40/250 | 3 |
| 1.5 | 55.17 | 54.01–55.91 | 33/500 | 0 |

The larger caps generally expose a speed/quality trade-off, but phases differ in cap and inherited evolutionary state. Training batches also vary. A score change at a phase boundary cannot be interpreted as a same-condition learning plateau.

Across all **500 final-phase champions**, none reached the sustained-straight goal, even without requiring a quality pass. During the 1.00 m/s phase, at least one champion reached it in seeds 1, 2, 3, and 5; seed 4 did not. That does not mean those particular champions also completed all 60 courses or won the frozen selection rule. Only seed 1's selected 1 m/s checkpoint meets the every-straight threshold on validation; the other selected checkpoints have minimum straight speeds of approximately 0.858–0.923 m/s.

| Seed | 1.5 m/s best validation-fitness generation | First 10 score mean | Last 10 score mean |
| --- | --- | --- | --- |
| 1 | 262 | 1.8400 | 1.9352 |
| 2 | 261 | 1.8902 | 2.1042 |
| 3 | 294 | 1.7914 | 2.1705 |
| 4 | 251 | 1.9710 | 1.9636 |
| 5 | 252 | 2.0478 | 2.0818 |

These observations do not establish that 150 or 200 generations is universally sufficient, or that later evolution is inherently worse. Validation champions vary after reproduction, and the final generation need not be the best. A fixed-cap matched-budget or stopping-rule experiment remains necessary for causal claims about evolution length and population size.

## Reward scale and the stability trade-off

On the validation tracks each selected 1 m/s checkpoint and its old ramp reference both finish, the additional successful-speed bonus is much larger than the additional command-jitter penalty:

| Seed | Paired speed-bonus gain | Additional jitter penalty | Gain / extra penalty |
| --- | --- | --- | --- |
| 1 | 0.102525 | 0.000759 | 135.2× |
| 2 | 0.090722 | 0.000564 | 160.8× |
| 3 | 0.095222 | 0.001827 | 52.1× |
| 4 | 0.097197 | 0.000307 | 316.7× |
| 5 | 0.099949 | 0.000443 | 225.8× |

The comparison uses the reference at its frozen 1.50 m/s cap and the checkpoint at 1.00 m/s; it is a descriptive scale diagnostic, not an equal-cap causal experiment. It helps explain why the training objective can favor a faster controller even when the stricter external quality gate rejects it. Geometric RMSE is not directly penalized by this fitness. Increasing a penalty or adding an RMSE term would be a new experiment, not a correction applied to these completed runs.

## Species and network structure

| Seed | Species ever observed | Maximum concurrent species | Champion nodes: first → final | Committed-generation hours |
| --- | --- | --- | --- | --- |
| 1 | 30 | 8 | 4 → 3 | 4.24 |
| 2 | 25 | 8 | 3 → 4 | 4.79 |
| 3 | 8 | 4 | 7 → 14 | 4.37 |
| 4 | 29 | 7 | 2 → 6 | 4.38 |
| 5 | 9 | 4 | 7 → 6 | 5.86 |

Species IDs are local to each run. “Ever observed” includes founding species; disappearance means absent from the next recorded generation, not a diagnosed cause of extinction. Node counts describe stored champion node genes, including outputs and potentially inactive nodes; they are not counts of active hidden computations. Full birth/disappearance counts and champion enabled-edge counts are saved in `species-history.csv`.

Seed 3 skipped drive evaluation in **82 final-phase champions**, between generations 203 and 299 (not every generation in that range). The other four lineages never skipped drive evaluation. Its node-gene growth therefore does not imply that both motor-control outputs remained actively computed. The selected 1 m/s networks contain only **3–8 node genes**, showing that repeatable successful behavior does not require identical network sizes or a large topology in this setting.

## Original formal held-out assessment

The original prepared 150 test tracks were evaluated under the original frozen selection rule: five old ramp references, the sole new winner, and three prepared PID configurations, each at 0.50, 1.00, and 1.50 m/s (**4,050 episodes**). The cap-specific rows are shown below; all cross-cap PID results remain in the raw files.

| Cap | Policy | Completed /150 | Progress % | RMSE mm | Jitter | Min straight m/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5 | reference-01 | 145 | 98.0 | 12.87 | 0.002993 | 0.520 |
| 1.5 | reference-02 | 149 | 99.5 | 5.62 | 0.003535 | 0.520 |
| 1.5 | reference-03 | 147 | 99.0 | 8.27 | 0.002984 | 0.520 |
| 1.5 | reference-04 | 150 | 100.0 | 6.29 | 0.003874 | 0.520 |
| 1.5 | reference-05 | 150 | 100.0 | 6.24 | 0.003992 | 0.520 |
| 1.5 | new-03 | 149 | 99.7 | 6.23 | 0.002122 | 0.520 |
| 1.5 | pid-150 | 85 | 72.9 | 52.61 | 0.031217 | 1.114 |
| 0.5 | reference-01 | 133 | 94.0 | 16.47 | 0.001911 | 0.173 |
| 0.5 | reference-02 | 146 | 98.6 | 7.55 | 0.003153 | 0.173 |
| 0.5 | reference-03 | 147 | 99.1 | 10.17 | 0.001765 | 0.173 |
| 0.5 | reference-04 | 139 | 96.2 | 8.08 | 0.002340 | 0.173 |
| 0.5 | reference-05 | 145 | 98.5 | 7.06 | 0.001938 | 0.173 |
| 0.5 | new-03 | 141 | 96.5 | 8.25 | 0.001018 | 0.173 |
| 0.5 | pid-050 | 149 | 99.5 | 12.09 | 0.000915 | 0.371 |
| 1.0 | reference-01 | 145 | 98.7 | 9.37 | 0.002524 | 0.347 |
| 1.0 | reference-02 | 150 | 100.0 | 5.36 | 0.002968 | 0.347 |
| 1.0 | reference-03 | 146 | 98.7 | 8.00 | 0.002187 | 0.347 |
| 1.0 | reference-04 | 146 | 99.1 | 6.50 | 0.002710 | 0.347 |
| 1.0 | reference-05 | 149 | 99.5 | 6.57 | 0.002982 | 0.347 |
| 1.0 | new-03 | 147 | 99.2 | 7.48 | 0.001726 | 0.347 |
| 1.0 | pid-100 | 149 | 99.7 | 27.54 | 0.004995 | 0.743 |

| Cap | New winner | Comparator | Shared finishes | New time s | Comparator time s | Mean difference s |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5 | new-03 | reference-03 | 146 | 7.312 | 7.199 | +0.113 |
| 1.5 | new-03 | pid-150 | 85 | 7.968 | 7.744 | +0.224 |
| 0.5 | new-03 | reference-03 | 139 | 20.847 | 20.846 | +0.001 |
| 0.5 | new-03 | pid-050 | 140 | 20.907 | 11.006 | +9.901 |
| 1.0 | new-03 | reference-03 | 143 | 10.655 | 10.555 | +0.100 |
| 1.0 | new-03 | pid-100 | 146 | 10.778 | 5.963 | +4.815 |

All three prepared PID configurations have base 0.75, so their nominal centered-straight requests are 0.375, 0.75, and 1.125 m/s at the matching caps. Preserve these planned results, but use the separate full-speed comparison below for the current 1 m/s baseline question. No unqualified new winner was silently substituted into the formal assessment.

As a post-test diagnostic, applying the same quality comparisons to the new winner versus its old reference at 1.50 m/s gives **eligible=True**, reasons **none**. On 146 shared test finishes, RMSE is 5.512 versus 6.628 mm and jitter is 0.002009 versus 0.002754. This measures generalization of the validation checks; it does not change the frozen winner or select a replacement.

## Fresh five-seed 1 m/s comparison against full-speed PID

All six frozen policies drove the same **150 fresh geometry-disjoint tracks**: 90 mixed (30 per difficulty), 30 speed, and 30 corner drills, at a **1.00 m/s cap**. The PID tune was frozen before this suite was generated; there was no further tuning or model reselection. This comparison contributes **900 new episodes**.

| Policy | Completed /150 | Progress % | RMSE all mm | Jitter all | Mean chassis m/s | Peak m/s | Min straight m/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed-01 | 149 | 99.6 | 18.22 | 0.007495 | 0.709 | 1.002 | 0.972 |
| seed-02 | 147 | 98.8 | 16.02 | 0.006651 | 0.696 | 1.003 | 0.858 |
| seed-03 | 147 | 98.8 | 20.54 | 0.011942 | 0.698 | 0.997 | 0.914 |
| seed-04 | 146 | 98.4 | 18.01 | 0.005127 | 0.702 | 1.003 | 0.905 |
| seed-05 | 140 | 96.4 | 19.59 | 0.006408 | 0.709 | 0.957 | 0.923 |
| pid | 145 | 98.0 | 23.61 | 0.001092 | 0.578 | 1.003 | 0.991 |

| Seed | Shared finishes | NEAT / PID time s | Time change % [95% CI] | NEAT / PID RMSE mm | NEAT / PID jitter |
| --- | --- | --- | --- | --- | --- |
| 1 | 144 | 5.422 / 6.975 | -22.3 [-23.5, -20.7] | 17.16 / 23.08 | 0.007383 / 0.001058 |
| 2 | 143 | 5.470 / 6.896 | -20.7 [-22.0, -19.2] | 15.27 / 23.21 | 0.006542 / 0.001057 |
| 3 | 142 | 5.348 / 6.863 | -22.1 [-23.1, -21.1] | 14.80 / 23.18 | 0.011700 / 0.001046 |
| 4 | 141 | 5.345 / 6.801 | -21.4 [-22.5, -20.3] | 15.66 / 23.14 | 0.004874 / 0.001042 |
| 5 | 136 | 5.218 / 6.578 | -20.7 [-21.9, -19.3] | 17.27 / 23.05 | 0.006137 / 0.001026 |

Across five seeds, NEAT completion averaged **145.8/150**, SD **3.42**, range **140–149**. Mean paired time change across seed-specific shared subsets was **-21.4%**, SD **0.8 percentage points**; negative means NEAT finished sooner. This average is descriptive, with five lineages.

Relative to full-speed PID on each seed's paired successful tracks, **5/5** NEAT controllers finish sooner, **5/5** have lower geometric RMSE, and **0/5** have lower command jitter. **1/5** meet the every-straight sustained-speed goal on this fresh set; PID goal: **True**. These are separate performance dimensions.

| Seed | Completion difference pp [95% CI] | NEAT-only / PID-only finishes | Exact McNemar p |
| --- | --- | --- | --- |
| 1 | +2.7 [+0.0, +6.0] | 5 / 1 | 0.2188 |
| 2 | +1.3 [-2.0, +4.7] | 4 / 2 | 0.6875 |
| 3 | +1.3 [-2.0, +5.3] | 5 / 3 | 0.7266 |
| 4 | +0.7 [-3.3, +4.7] | 5 / 4 | 1.0000 |
| 5 | -3.3 [-8.0, +1.3] | 4 / 9 | 0.2668 |

Intervals use 20,000 paired bootstrap draws stratified by track group and mixed-course difficulty. They quantify generated-track uncertainty for each frozen controller, **not uncertainty across training seeds**. Five pairwise tests are exploratory and unadjusted for multiplicity. Shared tracks and the common PID baseline make the comparisons dependent; 750 NEAT episodes are not 750 independent track draws.

| Policy | Mixed /90 | Speed /30 | Corner /30 | Failure reasons |
| --- | --- | --- | --- | --- |
| seed-01 | 90 | 30 | 29 | {"stalled": 1} |
| seed-02 | 87 | 30 | 30 | {"stalled": 3} |
| seed-03 | 87 | 30 | 30 | {"off_route": 2, "stalled": 1} |
| seed-04 | 86 | 30 | 30 | {"stalled": 4} |
| seed-05 | 80 | 30 | 30 | {"stalled": 10} |
| pid | 85 | 30 | 30 | {"stalled": 5} |

| Policy | Mixed beginner /30 | Mixed intermediate /30 | Mixed advanced /30 |
| --- | --- | --- | --- |
| seed-01 | 30 | 30 | 30 |
| seed-02 | 30 | 30 | 27 |
| seed-03 | 30 | 28 | 29 |
| seed-04 | 30 | 28 | 28 |
| seed-05 | 30 | 26 | 24 |
| pid | 30 | 29 | 26 |

The similarity figure compares correlations of each NEAT controller's **NEAT/PID time-ratio profile** on **132 tracks all six policies finish**, plus successful-track intersection/union overlap. Dividing by PID time reduces simple track-duration effects; all profiles still share the same PID denominator, so correlations are descriptive. Raw finish-time correlations are retained in the JSON. Similar course-level outcomes do not imply identical neural networks, steering actions, or trajectories.

Seeds 2–4 have strongly correlated relative-time profiles (approximately r=0.984–0.994), while seed 1's correlations with the other seeds are approximately 0.532–0.633. Their similar total paired time gains therefore conceal meaningful differences in which courses favor each controller.

For direct seed-to-seed comparison, the following table uses that exact same common successful subset for every controller (rather than different pairwise subsets):

| Policy | Common successful tracks | Mean time s | RMSE mm | Command jitter |
| --- | --- | --- | --- | --- |
| seed-01 | 132 | 5.119 | 16.96 | 0.007184 |
| seed-02 | 132 | 5.174 | 14.06 | 0.006486 |
| seed-03 | 132 | 5.133 | 14.40 | 0.011757 |
| seed-04 | 132 | 5.139 | 14.90 | 0.004782 |
| seed-05 | 132 | 5.165 | 16.97 | 0.006029 |
| pid | 132 | 6.483 | 22.79 | 0.001014 |

### Where any finish-time advantage occurs

The course-type plot separates pure dedicated straights from the other speed courses, rather than attributing all differences to straight-line speed or cornering. Each entry below averages the five seed-specific paired percentage changes; the shared-finish denominator range shows the differing successful subsets. This is descriptive subgroup analysis, with no additional policy selection.

| Course type | Courses | Shared finishes: seed range | Mean time change % | Seed SD, pp |
| --- | --- | --- | --- | --- |
| beginner | 30 | 30–30 | -21.7 | 1.0 |
| intermediate | 30 | 25–29 | -30.3 | 0.5 |
| advanced | 30 | 21–26 | -31.0 | 1.0 |
| speed-straight | 9 | 9–9 | +8.2 | 4.7 |
| speed-shaped | 21 | 21–21 | +2.5 | 3.4 |
| corner | 30 | 29–30 | -21.5 | 3.7 |

PID is quicker on dedicated straights for all five comparisons. NEAT's overall advantage appears on the mixed courses and shared successful corner drills; it is not explained by exceeding PID's straight-line speed. This is a course-level observation, not a segment-level causal analysis of individual turns.

### Sensitivity to which test tracks are sampled

The **same frozen seed-1 genome, PID gains, robot, and episode settings** previously completed 142/150 versus 146/150 on the 19 November suite. On this new cohort the counts are 149/150 versus 145/150. That reversal in the raw completion ranking illustrates why one cohort cannot establish a general completion advantage, and why track uncertainty and independent seeds must be reported. The different cohorts remain separate; no old measurements are overwritten.

A `stalled` failure means five seconds without new authenticated route progress under the common evaluator; it does not necessarily mean the chassis stopped moving. Off-route failures use the common 100 mm corridor and two-second no-sensor-contact rule.

## Interpretation limits and experiment record

- Completion, geometric RMSE, and command jitter must be considered together. Command jitter measures motor-command changes, not direct chassis vibration. All-episode RMSE averages over whatever portion each policy observed before termination; shared-finish RMSE avoids comparing different successful subsets.
- Course-average forward speed, peak speed, the speed cap, and central-straight sustained speed are different metrics. A cap or a brief peak does not prove sustained travel at that speed.
- The study changes fitness, training mixture, and training budget together. Without matched controls, we cannot attribute a change specifically to the reward or speed courses, prove NEAT is universally superior to PID, or claim an optimal generation/population count.
- The five low-speed parents were independent earlier lineages, and fast-stage evolutionary seeds differ. All fast runs share the validation suite and training speed bank. The 1 m/s follow-up was motivated by seed 1; treat it as exploratory replication of that observation.
- The formal and follow-up assessments use different test cohorts and corner-drill recipes. Compare controllers within each shared cohort; do not interpret differences between headline percentages across cohorts as controller improvements.
- Seed 5 had memory-related interruptions and execution-only recovery changes, including a recorded reduction in actual evaluation workers. Earlier runs used eight workers. Do not directly treat elapsed runtime as a controlled hardware-efficiency comparison. A separate machine inspection found unusually high OneDrive memory commitment, but did not establish its precise cause or the memory state at each crash.
- This review's assessments use two CPU workers, clear the PID helper's image cache between tasks, recycle processes every 32 tasks, and save episode-level progress. Physics, controller files, test suites, fitness, and episode termination rules are unchanged. No training is resumed.

## Frozen models, raw results, and reproduction

- Original formal selections: `runs/neat-fast-study/seed-XX/training/selection.json`; sole selected genome: `seed-03/training/best.pkl`. Formal frozen copies and hashes: [selection-freeze.json](../runs/neat-fast-study/assessment/frozen/selection-freeze.json).
- The five 1 m/s genomes and full-speed PID are copied to [comparison-1m/frozen](../runs/neat-fast-study/final-review-20261121/comparison-1m/frozen/manifest.json); the manifest records exact source paths, SHA-256 hashes, generation numbers, and the pre-test selection rule.
- Formal raw episodes: [test-episodes.csv](../runs/neat-fast-study/assessment/results/test-episodes.csv); per-policy/cap/group summaries and paired finish-time comparisons are alongside it.
- Follow-up raw episodes: [episodes.csv](../runs/neat-fast-study/final-review-20261121/comparison-1m/results/episodes.csv); [structured comparison](../runs/neat-fast-study/final-review-20261121/comparison-1m-summary.json) contains every per-seed interval, group metric, failure count, and similarity matrix.
- Training evidence: [training-summary.json](../runs/neat-fast-study/final-review-20261121/training-summary.json), [all 1,500 validation summaries](../runs/neat-fast-study/final-review-20261121/validation-generations.csv), [phase summaries](../runs/neat-fast-study/final-review-20261121/phase-summary.csv), [species history](../runs/neat-fast-study/final-review-20261121/species-history.csv).
- Paper figures, each SVG and 300-dpi PNG: [paper-figures folder](../runs/neat-fast-study/final-review-20261121/paper-figures). Figure filenames cover learning, phase quality, species/network complexity, fitness components, five-seed PID comparison, controller similarity, and original formal assessment.
- [Verification results](../runs/neat-fast-study/final-review-20261121/verification.json), [controller-output probe](../runs/neat-fast-study/final-review-20261121/controller-output-probe.json), and [evidence/figure hashes](../runs/neat-fast-study/final-review-20261121/review-evidence.json) accompany the report. The ten figure families include measured speed definitions, final-phase speed/quality trade-offs, and finish-time changes by course type.
- Historical context: [seed 1 validation review](NEAT_Fast_Study_Seed_01_Results.md), [previous full-speed PID comparison](NEAT134_vs_Fullspeed_PID_100_Results.md), and [training/recovery guide](../docs/fast-neat-repeats.md). Earlier reports remain historical snapshots.

```powershell
$py = ".\.venv\Scripts\python.exe"

# Reproduce/reuse assessments; resumes episode-level partial results; no evolution
& $py runs/neat-fast-study/final-review-20261121/assess.py --workers 2

# Regenerate the review, statistics, and SVG/PNG figures from saved results
& $py runs/neat-fast-study/final-review-20261121/report.py
```
