# Line following experiment methodology

This is the working methods record for the paper. It connects the controller and evaluator code to each experiment's protocol and evidence. Each run's saved protocol and source snapshot are the authority for its exact settings.

## Research questions and experiment record

The main question is how a learned NEAT controller compares with sensor-based PID on unseen line following courses under matched simulation conditions. The speed experiment asks whether additional evolution from g94 improves completion and actual driving speed when the wheel speed cap rises.

| Experiment | Method and evidence |
| --- | --- |
| Historical NEAT and zigzag diagnosis | The original 100-generation run and 74 completed continuation generations produced the previously selected g109 controller. The research report records configuration changes, seen versus held-out tracks, and the corner failure diagnosis. [Historical NEAT results](../research/NEAT_Experimental_Results.md) · [Corner diagnosis](../runs/diagnostics/zigzag-20261029/report.md) |
| PID development and frozen comparisons | Bounded searches produced filtered PID and corner-supervised PID baselines. The original comparison and later gain tuning used distinct test cohorts, each with 90 tracks. Compare controllers within each matched cohort. [PID implementation](../research/PID_Controller_Methodology.md) · [Initial protocol](../research/PID_Protocol.md) · [Gain tuning protocol](../research/PID_Gain_Tuning_Protocol.md) · [Matched results](../research/NEAT_PID_Comparison.md) |
| Fresh NEAT corner run | A fresh population of 96 evolved for 100 generations at 0.35 m/s. Four shared courses per generation included three short corner drills and a separate procedural course. The selected g94 finished 30/30 validation tracks and 88/90 fresh test tracks. Four frozen comparators ran on the same test tracks. [Training protocol](corner-training.md) · [Results](../research/NEAT_Corner_Results.md) · [Reproducible figures](../research/NEAT_Corner_Plots.ipynb) |
| Warm start with increasing speed | The run finished 100 generations from one exact g94 copy and 99 independently mutated copies. The wheel speed cap rose from 0.50 to 1.50 m/s. Under corrected evaluation, the same frozen g98 finished 67/90 at 0.50 m/s and 90/90 at 1.50 m/s. Frozen recovery PID finished 90/90 and 34/90 respectively. [Training guide](speed-training.md) · [Corrected results and notebook](../research/NEAT_Speed_Results.md) · [Frozen run protocol](../runs/neat-speed-100-20261104/protocol.json) |

The fresh corner run changed several elements together, including the controller inputs and action mapping, training tracks, and evolutionary settings. Treat its comparison with g109 as a comparison of frozen controllers. A change in the warm-start run also bundles added evolution with the new training schedule. Independent runs or controlled comparisons are needed to estimate run-to-run variation or attribute gains to one design choice.

## Planned retraining and high-speed PID comparison

The primary question is how reliably a speed-trained NEAT controller finishes unseen tracks at a 1.50 m/s wheel speed cap compared with PID tuned at that cap. A secondary question is how well the same controllers transfer to other caps.

1. **Use the corrected evaluator throughout.** Use `progress-1.2` with the 100 mm corridor half-width and two-second off-route allowance from the corrected assessment for NEAT training, PID tuning, validation, and final test. The off-route timer accumulates only while the robot is outside the corridor and every sensor has lost the painted line. Retain ordered progress and the stall check. Before launch, verify that a valid corner cut can continue and that genuine departures and stalls still terminate. The historical launchers inherit the earlier corridor and departure settings. [Corrected assessment](../research/NEAT_Speed_Results.md#what-changed-in-evaluation)
2. **Train from scratch at 0.50 m/s.** Give each independent seed a fresh population and 150 generations at this cap. Start with the existing corner drills and progressively harder procedural tracks. All genomes in a generation see the same fresh batch. Select the seed's best genome on fixed validation tracks by completion rate, then validation fitness.
3. **Warm-start another 150 generations from that seed's winner.** Use one exact copy and 99 mutated copies. Spend ten generations each at 0.65, 0.80, 1.00, and 1.25 m/s, followed by 110 at 1.50 m/s. Use fresh mixed-difficulty tracks and select the final genome from the 1.50 m/s phase by validation completion, then fitness. Evaluate both saved genomes at 0.50 and 1.50 m/s to measure low-speed retention and high-speed improvement. Evaluate intermediate caps as well if the paper makes claims about performance across the full range.
4. **Run 150 + 150 generations for each of the five seeds.** Record evolution and track-generation seeds and share the validation and test courses across runs. Use the first seed's validation trend at the fixed 1.50 m/s cap to decide whether a longer run is worth trying. Save the final checkpoint for each stage. Report per-seed results and their spread, and report any longer run separately.
5. **Tune recovery PID at 1.50 m/s.** Start with a bounded search over P, I, and D on development tracks, with the saved gains included as a reference. Use separate validation tracks to choose the gains. The historical gains were tuned at 0.35 m/s and transferred to 0.50 and 1.50 m/s. Test the original and high-speed gains at both caps to measure what retuning changes. A matching search at 0.50 m/s would also let us compare two cap-specific tunings and test transfer in both directions. [PID tuning protocol](../research/PID_Gain_Tuning_Protocol.md) · [Historical gain provenance](../research/NEAT_Speed_Results.md#frozen-controllers-and-methodology)

Freeze all genomes, PID settings, and analysis choices before opening one fresh, shared final test suite. Use completion as the primary metric and report failure reasons, geometric tracking error, and measured chassis speed. Compare completion times on shared finishes, with the number of shared tracks shown.

**Paper note.** In the earlier 1.50 m/s comparison, the transferred PID finished faster on the 34 routes both controllers completed, while g98 finished all 90. Report that speed and completion trade-off together and identify the original PID tuning cap. The new matched-cap comparison will show whether retuning changes that trade-off. Use tracking error, departures, and stalls to describe observed behavior. [Historical paired results](../research/NEAT_Speed_Results.md#time-and-error-on-shared-finishes)

### Completed PID tuning at both caps on 15 November 2026

The search evaluated 208 gain triples per cap, starting with the same frozen recovery PID and holding the base command and recovery settings fixed. On 30 validation tracks, the original gains won at 0.50 m/s with 30 finishes. At 1.50 m/s, the selected gains finished 19 tracks compared with 12 for the original gains. Both configurations were frozen before evaluation on the same 90 fresh tracks. [Protocol](../runs/pid-speed-tuning-20261115/protocol.json) · [Selection record](../runs/pid-speed-tuning-20261115/selection-freeze.json)

| Wheel speed cap | Original finishes | Cap-specific selection | Selected P / I / D |
| --- | ---: | ---: | --- |
| 0.50 m/s | 90/90 | 90/90 (original gains retained) | 0.723985 / 0.168134 / 0.021582 |
| 1.50 m/s | 38/90 | 45/90 | 4.580999 / 0 / 0.038128 |

At 1.50 m/s, the new gains reduced off-route terminations from 44 to 22 and increased stalls from 8 to 23. On the 35 tracks both configurations finished, mean time went from 2.41 to 3.18 seconds. The high-speed gains transferred to 85/90 finishes at 0.50 m/s. For the paper, report the completion and time trade-off together and use a fresh shared test suite for the upcoming NEAT comparison. [Held-out summary](../runs/pid-speed-tuning-20261115/test-summary.csv) · [0.50 m/s settings](../runs/pid-speed-tuning-20261115/tuned-0.50.json) · [1.50 m/s settings](../runs/pid-speed-tuning-20261115/tuned-1.50.json)

### PID follow-up at 1.00 and 1.50 m/s on 16 November 2026

The follow-up tested 240 gain triples per cap, including the original recovery PID and the previous 1.50 m/s tune. Only P, I, and D varied. Selection on 30 validation tracks ranked completion count, mean route progress, then mean study fitness, which includes speed and command jitter. Both selections were frozen before evaluation on the same 90 fresh tracks. This suite is disjoint from the NEAT study suite. [Tuning runner](../run_pid_speed_followup.py) · [Protocol](../runs/pid-speed-followup-20261116/protocol.json) · [Selection record](../runs/pid-speed-followup-20261116/selection-freeze.json)

| Wheel speed cap | Original PID | Previous 1.50 m/s gains | New cap-specific gains | Selected P / I / D |
| --- | ---: | ---: | ---: | --- |
| 1.00 m/s | 85/90 | 81/90 | 87/90 | 0.372062 / 0.087212 / 0.056109 |
| 1.50 m/s | 38/90 | 48/90 | 67/90 | 8.632739 / 0.450732 / 0.080984 |

At 1.50 m/s, the retune cut off-route failures from 20 to 4 and improved advanced completion from 8/30 to 14/30 against the previous high-speed gains. On 44 shared finishes, its mean completion time rose from 4.44 to 6.54 seconds. Mean command jitter across all episodes rose from 0.0230 to 0.0484. Report these trade-offs with the completion gains. [Test summary](../runs/pid-speed-followup-20261116/test-summary.csv) · [Paired results](../runs/pid-speed-followup-20261116/paired-summary.csv) · [1.00 m/s gains](../runs/pid-speed-followup-20261116/tuned-1.00.json) · [1.50 m/s gains](../runs/pid-speed-followup-20261116/tuned-1.50.json)

### Run the five-seed study

Run the commands below in PowerShell from the project directory. Set `$py`, `$study`, and `$run` again when opening a new terminal.

#### Check and prepare

Preparation is safe to repeat with the same inputs. It saves the protocol, suite manifest, source snapshot, and verified PID files. [Prepared protocol](../runs/neat-speed-study/prepared/protocol.json)

```powershell
$py = ".\.venv\Scripts\python.exe"
$study = "runs/neat-speed-study"
& $py run_neat_speed_study.py preflight
& $py run_neat_speed_study.py prepare
```

#### Start or resume training

```powershell
& $py run_neat_speed_study.py train-low --seed 1
```

After the low-speed stage finishes, review its plots and start the ramp separately.

```powershell
& $py run_neat_speed_study.py train-ramp --seed 1
```

After an interruption, add `--resume` to the command for that stage, for example `& $py run_neat_speed_study.py train-ramp --seed 1 --resume`. For seeds 2–5, change the seed number in the command and use the matching folder (`seed-02` through `seed-05`).

#### Watch live or replay a saved winner

In a second terminal while training is running, attach the population viewer. Swap `low` for `ramp` when watching the high-speed stage.

```powershell
& $py -m line_follower preview-training --run "$study/seed-01/low"
```

After a stage finishes, replay its validation-selected `best.pkl` on a saved validation track.

```powershell
$run = "$study/seed-01/ramp"
& $py replay_neat_speed_study.py --model "$run/best.pkl" --track "$run/frozen-validation/002.json"
```

The replay uses the model's saved speed cap and episode rules, including the study's wider corridor and line recovery window. Track `002.json` is an advanced validation route. Press **R** to replay, **Space** to pause, **T** to show the reference route, or **C** to switch between NEAT and the viewer's built-in PID. The final assessment uses the separately frozen filtered PID configurations.

For an exploratory challenge, try advanced track seed 9004. It is a 5.75 m route with zigzags, a U-turn, gaps, a branch, and a crossing. Its geometry is outside the study's held-out suite.

```powershell
& $py replay_neat_speed_study.py --model "$study/seed-01/ramp/best.pkl" --seed 9004 --difficulty advanced
```

#### Plot saved results

```powershell
& $py plot_training.py --run "$study/seed-01/low"
& $py plot_training.py --run "$study/seed-01/ramp"
Invoke-Item "$study/seed-01/ramp/plots/learning-curves.png"
```

The plotter writes PNG, PDF, and SVG files to each stage's `plots/` directory. Start with `learning-curves` for progress over generations, `selected-policy-validation` for the selected model by difficulty, and `training-diagnostics` for failure and tracking metrics.

Both stages use a target of 100 genomes for 150 generations per seed. The low-speed stage uses the corner curriculum and the ramp uses the zigzag profile. Both run on eight CPU workers and see the same 30 validation tracks. In the ramp, generations 0–9 run at 0.65 m/s, 10–19 at 0.80, 20–29 at 1.00, 30–39 at 1.25, and 40–149 at 1.50. Review the final-cap validation trend for seed 1 before repeating both commands for seeds 2–5.

Both stages retain the ordered progress score, completion bonus, elapsed-time term, and bounded legacy term. A finished episode earns an additional `0.40 × clip(1 − elapsed / timeout, 0, 1)`. Every episode pays `0.20 × command_jitter`, where jitter is the mean of `(Δleft² + Δright²) / 8` over successive 60 Hz commands. Commands are normalized to ±1, so the metric is between 0 and 1. The first command establishes the reference. Holding a turn incurs zero command-change penalty. Validation completion remains the first selection criterion. Log jitter alongside tracking error, failures, and measured speed, since command smoothness is one measure of stability.

#### Run the final assessment

After both stages finish for all five seeds:

```powershell
& $py run_neat_speed_study.py assess
```

The command verifies the ten validation-selected models and freezes copies alongside both saved filtered PID configurations before loading the test tracks. It evaluates all twelve policies at both caps on the same 90 tracks. The primary comparison is the five ramp winners versus the 1.50 m/s PID at 1.50 m/s. The 0.50 m/s selection retains the original PID gains. Results in `runs/neat-speed-study/assessment/results/` include 2,160 raw episodes, completion by seed, spread across seeds, termination counts, geometric tracking error, measured chassis speed, and paired finish times. This study compares the resulting controllers. An estimate of the speed ramp's effect would require a matched-budget comparison.

Regenerate the paper figures from the completed assessment:

```powershell
& $py plot_neat_five_seed_assessment.py
```

The [paper figure collection](../research/artifacts/neat-five-seed-paper-figures/README.md) contains the per-seed completion results, pairwise similarity, speed and tracking comparisons, learning curves, and network input paths.

## Paper follow-ups after the current training

The next implemented experiment is documented in [Faster NEAT driving with protected tracking and smoothness](fast-neat-study.md), including its fitness, quality gates, and terminal commands.

The [19 November 2026 PID smoothness refinement at the 1.00 m/s cap](../research/PID_100_Smoothness_Results.md) records a separate local gain/filter search, frozen selection, fresh PID-only test results, and replay commands. It preserves the original NEAT study's frozen PID inputs; the newer tune must be explicitly frozen for any subsequent matched comparison.

The [frozen generation 134 versus retuned PID comparison](../research/NEAT134_vs_Tuned_PID_100_Results.md) evaluates these two fixed controllers on 150 additional fresh shared tracks at the 1.00 m/s cap. Its report records canonical frozen paths, SHA-256 hashes, paired statistics, per-group results, figures, and a ten-track viewer that switches between the exact two frozen policies.

**Cruise-speed correction:** that historical comparison retained PID base 0.75, restricting its nominal straight request to 0.75 m/s. Use the [full-cruise protocol](../research/PID_Fullspeed_100_Protocol.md) and [corrected fresh comparison](../research/NEAT134_vs_Fullspeed_PID_100_Results.md) for claims involving full-speed-capable PID. The new PID has base 1.00, verified near-1 m/s sustained speed, and separately tuned steering/filtering/slowdown. The old 9.37% time result remains labeled as a restricted-cruise baseline result.

1. **Compare network architecture before and after the ramp.** For each of the five seeds, compare the saved 0.50 m/s winner with the ramp winner descended from it. Record hidden nodes, enabled connections, and which inputs and paths actually contribute to the outputs. Plot the two networks side by side and report how consistently the architecture changes across seeds. This compares the two saved stages of each lineage.
2. **Compare specialists with one generalist at three speeds.** At wheel-speed caps of 0.50, 1.00, and 1.50 m/s, compare a PID tuned at that cap, the best NEAT genome trained entirely at that cap, and a single NEAT genome selected on validation across all three caps. Freeze the generalist once and evaluate it at all three caps without retraining or switching genomes. Use the same fresh test tracks for all controllers. Match or explicitly report training budgets, and compare completion, times on shared finishes, tracking error, and command jitter.
3. **Study when to stop evolution and how large a population to use.** Across multiple seeds, track validation performance and the cost of each improvement for several population targets. Compare fixed generation budgets with validation-based stopping rules. Set the minimum improvement and patience before the comparison, then report the quality reached, generations used, total training episodes, and wall time. Use the results to recommend practical stopping and population settings for this task.
4. **Run a short population-size experiment with 50 total generations.** Repeat the same 50-generation protocol across several population targets and independent seeds. Compare learning curves by generation and by cumulative training episodes. This will show whether a larger population improves more per generation and whether a smaller population makes better use of the same number of episodes. Record actual episode counts because the realized population can vary. Compare these runs with the current 150 + 150 runs at shared episode budgets, and include the full-run results for context.

## Shared simulation and controller interface

The competition robot is a simulated 0.55 kg differential-drive chassis with a passive caster, 175 mm wheel spacing, and 16.5 mm wheel radius. MuJoCo advances at 600 Hz and the controller receives observations at 60 Hz. Fifteen binary reflectance sensors sample the painted floor at their actual poses. The controller receives the readings and control interval. The evaluator uses simulator pose and the ordered route to measure performance. [Geometry and sensor mount](sensor-mount.md) · [Simulator](../line_follower/simulation.py)

The wheel commands are scaled by the active rim speed target cap. The speed controller uses a 50 ms actuator filter and a 0.03 N·m torque limit per wheel. Distinguish the requested wheel speed cap from measured chassis speed in all figures and tables. The corner benchmark used a 0.35 m/s cap. The warm-start protocol changes the cap on a fixed generation schedule while retaining the robot geometry, actuator limits, and episode settings. [Physical parameters](../line_follower/config.py) · [Action mapping](../line_follower/simulation.py)

NEAT g94 and its descendants use a feedforward network with 21 inputs derived from the 15 physical sensors. The six added signals represent the current and last visible line centroids, recent edge evidence, line loss duration, and the previous drive and steering commands. The controller mirrors observations to share experience across turn directions, resets its history every episode, and mixes two tanh outputs into signed wheel commands. The research PID comparators use the same physical readings. Filtered PID includes derivative filtering and line loss recovery. The supervised variants add explicit corner handling. [NEAT implementation](../line_follower/controllers.py) · [PID implementation](../line_follower/pid.py)

## Tracks, selection, and metrics

- **Training.** All genomes in a generation face the same batch. Fresh tracks come from deterministic seeds. Track geometry hashes help exclude duplicate routes, including mirrored variants, from protected splits. The speed run uses one beginner, one intermediate, and two advanced tracks per generation. One intermediate and one advanced slot guarantee zigzags. [Track generator](../line_follower/tracks.py) · [Split and batch logic](../line_follower/suites.py)
- **Validation.** Evaluate each generation's training champion on the same 30 validation courses. Rank checkpoints by validation completion, then by validation fitness. During speed training, restart that ranking at each cap and select the final model from generations 70–99. Retain the per-generation results to see changes across phases. [Training and selection loop](../line_follower/experiments.py)
- **Final test.** Freeze the selected model before opening the final test geometry. Use a fresh suite of 90 tracks with 30 per difficulty and identical track hashes across the compared conditions. The speed experiment evaluates g94 and the new model at both 0.50 and 1.50 m/s on those 90 tracks, for 360 paired records. Keep exploratory tracks and replay outcomes labeled separately from the final comparison. [Speed assessment](../run_speed_training.py)
- **Completion and progress.** The evaluator checks ordered route gates and tracks the greatest authenticated progress. Both recent NEAT runs use a 45 mm corridor half-width, a 35 mm gate radius, and termination after one second continuously outside the corridor or five seconds without 10 mm of new progress. The timeout is route length divided by 0.10 m/s, bounded to 45–120 seconds. Also record tips and technical errors separately. [Episode loop](../line_follower/episodes.py) · [Route projection and gates](../line_follower/fitness.py)
- **Training fitness.** The episode score combines authenticated progress, a completion bonus, elapsed time relative to the timeout, and a small bounded legacy term. Both recent NEAT runs combine the average score of the four tracks and the average of the two worst tracks with equal weight. [Fitness equations](../line_follower/fitness.py)
- **Reporting.** Report completion counts and denominators, progress, and failure reasons. Compare completion time on the tracks both controllers finish and state the shared denominator. Lateral RMSE and MAE summarize observed tracking error per episode. Report whether the aggregation includes failures or uses shared finishes. Keep the training seed, track, and telemetry sample distinct as units of analysis. [Episode fields](../line_follower/episodes.py) · [Assessment metrics](../line_follower/pid_experiments.py)

## Exploratory speed check

Before the warm-start run, frozen g94 was evaluated at six speed caps on the same 12 newly generated pilot tracks, four per difficulty. The pilot used `derived_seed(2026110403, "speed-pilot", index, attempt)`, indices 0–11 in beginner, intermediate, advanced rotation, skipping failed generation attempts and geometry hashes already used or protected by the corner run. The robot and episode settings came from the g94 artifact.

| Wheel target cap | Completed |
| --- | ---: |
| 0.35 m/s | 12/12 |
| 0.50 m/s | 12/12 |
| 0.65 m/s | 12/12 |
| 0.80 m/s | 6/12 |
| 1.00 m/s | 5/12 |
| 1.50 m/s | 0/12 |

On a separate 2.5 m straight, g94 reached about 1.50 m/s with the highest cap. The pilot motivated starting at 0.50 m/s and reserving the final 30 generations for 1.50 m/s. It measured transfer of frozen g94 and served as a development check for the new protocol.

### Long-course speed challenge

The custom 20.54 m course compares the five frozen ramp winners with the newer 1.50 m/s tuned recovery PID under the same 1.50 m/s wheel-speed cap. All six finished. PID crossed the opening 5 m in 4.58 s, compared with 9.68 s for every NEAT seed. Full-course times were 46.05 s for PID and 39.27–39.45 s for NEAT. The course uses 2 mm floor sampling and provides a single exploratory comparison. [Course map](../runs/long-course-speed-challenge/course-map.svg) · [Results](../runs/long-course-speed-challenge/results.csv)

To rerun the challenge in a fresh output directory:

```powershell
& .\.venv\Scripts\python.exe run_long_course_speed_challenge.py --output runs/long-course-speed-challenge-repeat
```

The lower-cap follow-up used the same saved course and the tuned PID gains for each cap. Both PID runs finished. The 1.00 m/s PID finished in 29.55 s with a mean forward speed of 0.702 m/s. The 0.50 m/s PID finished in 56.90 s with a mean of 0.360 m/s. The reference NEAT times come from the original 1.50 m/s cap runs. [Lower-cap PID results](../runs/long-course-speed-challenge/pid-lower-caps/results.csv)

To run the lower-cap PID follow-up after generating a fresh course run:

```powershell
& .\.venv\Scripts\python.exe run_long_course_speed_challenge.py --lower-pid-caps --output runs/long-course-speed-challenge-repeat
```

**Paper observations.** All tested controllers completed this course. The 1.00 m/s PID tune was fastest overall at 29.55 s. The 1.50 m/s tune reached the highest chassis speed at 1.128 m/s and was fastest over the opening straight, yet took 46.05 s over the full route. The five ramp NEAT winners made steadier progress at about 0.52 m/s and finished in 39.27–39.45 s. Each PID cap used its own gains. Report peak speed, mean speed, and full-course time separately for this exploratory course.

**Follow-up question.** Could an earlier ramp checkpoint trained at 0.65 or 1.00 m/s finish this course sooner than the final-cap winner? Select one checkpoint per seed within each speed phase using the existing validation ranking. Compare those checkpoints on the same long course at their training caps, then at a common cap to measure transfer. Report completion, finish time, and achieved speed for each seed. Review the [averaged ramp fitness curves](../research/artifacts/neat-five-seed-paper-figures/06-ramp-fitness-across-seeds.svg) first.

### Rhythm Changes long-course case study (21 November 2026)

The user selected candidate 4 from five performance-blind layouts. The completed experiment freezes 31 NEAT models (five low parents, five validation-best champions per fast training cap, and the additional formal seed-3 final winner) plus five distinct saved filtered PID configurations before testing. NEAT models run at their native cap and at a shared 1.00 m/s cap; all PID tunes run at 0.50, 0.65, 0.80, 1.00, 1.25 and 1.50 m/s. The 87 episodes use the same 45.77 m course, physics and termination rules, with a predeclared timeout of `180 / min(cap, 1.0)` seconds. No tuning or evolution takes place. This is a single-course behavior study alongside the multi-track results.

[Full results and interpretation](../research/Rhythm_Changes_Long_Course_Results.md) · [Frozen protocol and controller fingerprints](../runs/rhythm-changes-case-study-20261121/protocol.json) · [Eight PNG/SVG figures and captions](../runs/rhythm-changes-case-study-20261121/paper-figures/README.md).

The requested video shortlist contains **six clips**: one NEAT and one PID at each main evaluation cap (0.50, 1.00, 1.50 m/s). Choose the fastest successful archived run on this course; if none completed, choose greatest progress (PID at 1.50). This is a retrospective showcase, not a replacement validation selection. Open `runs/rhythm-changes-case-study-20261121/videos/selected-six/index.html` for the six-video gallery. Earlier surplus recordings remain outside that folder.

**Review correction, 23 November:** the fastest-lap shortlist omitted the existing full-speed 1 m/s PID and used base-0.75 PID examples at every cap. It also exposes genuine NEAT post-turn straight-line oscillation. The gallery is historical and requires correction before use as the final full-speed/smoothness comparison. See [the video review and measured diagnostic](../research/Rhythm_Changes_Video_Review.md); original frozen results remain preserved.

Each clip shows a real MuJoCo close-up, full-course position, controller identity, training/tuning and evaluation caps, elapsed time, measured chassis speed, wheel commands and termination. Videos play at 1× real time and 15 fps with an approximately two-second final hold. Every replay is checked against the committed result before rendering; recordings never overwrite assessment data. The manifest records file hashes, frame counts and the original episode hash.

```powershell
# Record missing videos, or verify/reuse the existing archive
& .\.venv\Scripts\python.exe research/record_rhythm_changes.py
# Restrict recording to a particular existing episode
& .\.venv\Scripts\python.exe research/record_rhythm_changes.py --episodes neat-s1-fast-100__cap-100
# Open the local gallery
Start-Process .\runs\rhythm-changes-case-study-20261121\videos\selected-six\index.html
```

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py run_rhythm_changes_case_study.py verify
# Reuse checked completed episodes; evaluate missing ones only
& $py run_rhythm_changes_case_study.py run --workers 2
# Regenerate tables and plots without simulating
& $py runs/rhythm-changes-case-study-20261121/report.py
# Replay exact frozen policies, same course and 1 m/s cap
& $py run_rhythm_changes_case_study.py replay --episode neat-s1-fast-100__cap-100
& $py run_rhythm_changes_case_study.py replay --episode pid-fullspeed-100__cap-100
```

All other episode IDs are listed in `runs/rhythm-changes-case-study-20261121/results.csv`; use any ID with `replay --episode`. The viewer prints the controller class, frozen file and cap, and does not overwrite saved results. Keep the original proposal manifest as a pretest historical record.

## Smooth-driving follow-up (23 November 2026)

The [smooth NEAT and full-speed PID workflow](neat-smooth-study.md) prepares an isolated, versioned continuation experiment: audit archived 1 m/s champions, restore/tune base-1.0 PID baselines, compare old and normalized smoothness fitness in a controlled pilot, and then confirm the selected protocol across five warm-started lineages. It uses measured straight-line yaw and steering gates alongside completion, accuracy, sustained speed and a 5% lap-time allowance. Preparation does not launch tuning or evolution. The document includes every terminal command, resume rules, stopping conditions and assessment/video requirements.

## Reminder when paper writing begins

- [ ] Create a public project website for the NEAT versus PID research paper, inspired by [LeWorldModel's project page](https://le-wm.github.io/). User requested this reminder on 22 November 2026, to revisit when manuscript writing starts. This is a writing-stage checklist item, not a scheduled notification.
- Include the paper title/authors, concise methodology, meaningful comparison figures, robot demonstration videos showing successes and failures, and links to the paper, code, frozen models and reproducibility materials. Add citation information when ready. Decide hosting and publication timing during writing; no website has been built or published yet.

## What to record for each experiment

Video review on 5 November 2026 exposed false off-route failures in the original speed evaluation. A missed corner gate could leave the ordered cursor behind a robot that was still following the route. Evaluator `progress-1.2` increases gate tolerance to cover accepted corner cuts and resets departure time whenever line sensors are active. The corrected video assessment uses a 100 mm corridor and two seconds of continuous departure with no sensor contact for every controller. Its [protocol, full telemetry, and previous-versus-current episode outcomes](../research/artifacts/neat-speed-videos/README.md) document this retrospective change. Original 90-track rankings remain historical results under the earlier evaluator.

A subsequent [540-episode archive](../research/NEAT_Speed_Results.md) applies the same corrected rules to all 90 test tracks. The current report uses 360 episodes comparing the identical speed-trained g98 genome with frozen recovery PID at both 0.50 and 1.50 m/s. The archive also retains 180 evaluations of g94 and g00 as historical references. Physical tracking error is computed by exact closest-segment distance to the full intended route. The raw local gate-relative error remains available alongside it.

1. **Before launch.** Record the research question, comparison, intended changes and rationale, population and generation budget, seeds, controller and robot settings, split manifest and selection rule, metrics and their units. Mark pilots and planned comparisons explicitly.
2. **During the run.** Keep source and dependency versions, run protocol, actual episode counts, route files and geometry hashes, checkpoints, and any interruptions or deviations. Use the saved logs for changes in speed, curriculum or evaluation backend.
3. **After the run.** Record the selected model hash, generation and validation result, then verify the final test file and matched tracks. Preserve failures, separate final results from exploratory checks, and link the report, plotting code, and figures. Keep run archives with the repository because the run directories are ignored by Git.

The reproducibility trail is `protocol.json` → `suite/suite.json` → `training/run.json`, `training/source.zip`, and the training CSVs → `assessment/selection-freeze.json` and `assessment/test-episodes.csv`. The simulation supports claims about these courses under the recorded conditions. Hardware performance, disturbance tolerance, and reproducibility across evolution seeds require their own evidence.

The [historical implementation record](research-record.md) covers the PyGame-to-MuJoCo transition, earlier defaults, and engineering validation.
