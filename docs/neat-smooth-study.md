# Smooth NEAT and full-speed PID: experiment workflow

Implemented 23 November 2026. **Preparation and implementation checks only; PID tuning and NEAT evolution have not been started.** The primary goal is reliable, smooth driving at a 1.00 m/s wheel-speed cap, allowing at most 5% longer paired lap times than the corresponding validated NEAT parent. Review that milestone before designing further 1.50 m/s NEAT evolution.

The [video review](../research/Rhythm_Changes_Video_Review.md) established two different issues: the showcase selected restricted-cruise PID configurations despite an existing full-speed 1 m/s tune, and some fast NEAT models exhibit physical steering/yaw oscillation after corners despite low tracking error. This workflow addresses controller selection and measured motion, rather than hiding oscillation in playback.

## Frozen experiment and unchanged behavior

All new data is stored under `runs/neat-smooth-study-v1/`. Preparation copies the complete runtime and new experiment code into `prepared/source/`, archives it in `prepared/source.zip`, freezes inputs and hashes every source/input/course. Later commands run that copy in a separate Python process using the same virtual environment. The old `line_follower` source, original launchers, source fingerprints, checkpoints and controllers are not edited.

MuJoCo 4.0.0, NEAT-Python 0.92, the 21 neural inputs, action mapping, mutation settings, physical robot, sensors and recovery rules are retained. Stock handling of unevaluated outputs is retained and output-evaluation status is logged. No output filter is added. The corridor is 100 mm half-width; off-route termination requires two seconds outside it with no sensor contact; the existing five-second ordered-progress stall remains. Controllers never receive route geometry or a track label.

Default runtime is two CPU workers. A worker evicts reflectance masks before allocating another that would exceed its 64 MiB retained-cache allowance, and is recycled after 128 tasks. A single oversize mask is not cached. Large individual masks/simulators still need memory; the cache setting is not a hard process-memory bound. `--workers` changes runtime parallelism, not scientific settings; use one stage at a time on this machine.

## Tracks, budgets and validation

The pilot bank contains 20 speed courses: ten isolated 5–8 m straights and ten turns/chicanes followed by a 5–8 m recovery straight. Straight lengths are sampled above 5 m to preserve a complete central 2 m measurement interval. Paint is 20 mm and floor sampling 2 mm. Every generation uses the same six tracks for every genome:

1. One isolated speed-bank straight.
2. One post-turn recovery-bank course.
3. One fresh intermediate mixed course.
4. One fresh advanced mixed course.
5. Two fresh three-turn corner drills, with opposite turn ordering.

The bank cycles in seeded shuffled order, separately by isolated/recovery class. Both pilot branches use identical batches. Fresh course seeds do not depend on fitness or execution order. Geometry hashes exclude prior suites and generation batches, including mirrored equivalents. Pilot, confirmation and PID suites are distinct. The final test suite is not generated or loaded during preparation, tuning or evolution.

Each run has **100 additional generations and a target population of 100**. The first population contains one exact parent and 99 independently mutated copies using the original mutation settings. Stock NEAT-Python can subsequently produce slightly different population counts; actual counts and episode totals are logged. The pilot branches match generation budget, initial population and track batches, not an assumed exact 60,000-episode count.

For approximately 100 genomes, six training episodes each mean approximately 600 training episodes per generation. Those scores alone drive reproduction. The training champion is then evaluated on **80 fixed validation episodes**:

- **60 at 1 m/s:** 30 mixed (10 beginner, 10 intermediate, 10 advanced), 20 speed/recovery (10 isolated, 10 post-turn), 10 corner drills.
- **20 at 0.5 m/s:** 10 mixed, six speed/recovery, four corner drills, used to check retention.

These 80 episodes select the saved winner; they do not become breeding fitness. Validation is repeatedly used selection data, not untouched test performance. References evaluate the exact seed-specific parent on these same courses at the same caps. Neither pilot branch starts automatically after the other.

The audit evaluates up to three distinct historical 1 m/s champions per seed, chosen from the original generation 100–149 validation logs by completion/fitness, completion/jitter and completion/geometric RMSE. If an existing model already meets the new gates, the pilot command stops for review; an explicit `--force-pilot` is required to evolve anyway.

## Reward and acceptance rules

The old branch uses `route-speed-1.0`. The smooth branch records reward version `route-speed-smooth-1.0`. Both calculate physical diagnostics. The smooth reward retains all existing terms, including the **0.20 command-jitter penalty**, and adds:

```text
new_score = old_score
            - 0.15 * mean C(straight_yaw_rms, 0.15 rad/s)
            - 0.15 * mean C(straight_steering_rms, 0.02)
            - 0.10 * C(route_relative_training_rmse, 0.02 m)
            - 0.10 * mean clip(1 - straight_forward_speed / 0.95 m/s, 0, 1)^2

C(x, target) = x^2 / (x^2 + target^2)
steering = (left_command - right_command) / 2
```

Straight metrics use the full 60 Hz data, without video-rate sampling or smoothing, over each eligible primitive's central 2 m. Only straight primitives at least 5 m long qualify; curves and nearby braking/turning zones do not. Physical yaw RMS is raw chassis angular velocity, not the detrended steady-curve metric from earlier PID work. All intended windows are counted. Unfinished or insufficiently sampled windows receive worst-case cost and cannot pass validation. A course without any eligible straight is marked not applicable for straight costs. Added penalties are bounded by 0.50; tests verify completion remains preferable to deliberate early failure.

The inexpensive existing ordered-route-relative RMSE is used during training. Validation and assessment independently compute geometric RMSE to the full route polyline; these names are not interchangeable. Each reward component is logged separately.

A champion qualifies only if:

- Completion count matches or exceeds its parent within each mixed/speed/corner group, at both 1 m/s and the 0.5 m/s retention cap.
- Mean paired completion time at 1 m/s is no more than 1.05 times the parent's mean, on the tracks both finish; the denominator is recorded and cannot be empty.
- Mean geometric RMSE and command jitter do not exceed the parent on those shared successful primary tracks.
- Every required primary straight window is observed, averages at least 0.95 m/s, has yaw RMS at most 0.15 rad/s, and steering RMS at most 0.02.

Among qualifiers, maximize completion. Retain candidates within 2% of the fastest qualifying paired-time ratio, then minimize straight yaw, command jitter, geometric RMSE and time. Identical pilot branch metrics prefer the old fitness; remaining ties prefer the earlier generation. `best.pkl` is therefore the best qualifying validation selection, not simply the highest training fitness. If no model qualifies, no winner is manufactured: all champions, rejection reasons and physical trade-offs remain in the logs.

## PID workflow

The saved full-speed 1 m/s PID (`base=1.0`, SHA-256 `c6c68b53342600db0078e991bbb729064d829d360c4840a720dbf5e66d94c24b`) is copied as the reference. Validate it first on the fresh PID validation suite and retain it if eligible. Retune only if that check fails. The old base-0.75 configurations are retained as historical inputs, never eligible full-speed winners.

Tune independent full-base configurations at 0.50 and 1.50 m/s. All candidates use base exactly 1.0. Variables and bounds are P 0.05–3, I 0–0.25, D 0.005–0.20, derivative filter 0.02–0.20 s and slowdown gain 0.3–10. Recovery parameters and logic stay fixed. The search starts with the established full-speed reference and clipped cap-specific gain seed, a 64-point structured grid and 64 seeded random samples, then two 24-candidate local refinements. Coarse screening uses 12 courses, expanded evaluation uses all 50 development courses, and up to ten finalists use the separate 60-course validation set. Each complete candidate evaluation is committed, so reissuing the command resumes the deterministic search.

Eligibility requires every isolated validation straight to finish and sustain at least 95% of the cap over its central 2 m, and at least 57/60 total completions. Among maximum-completion candidates within 2% of the fastest failure-penalized time, select lowest steady physical yaw, then jitter, geometric RMSE and time. Failures receive their configured timeout as tuning time cost; failure elapsed time is never a lap time. No qualifying configuration means the tuning stage reports failure and freezes no winner. Raising base alone is not evidence of a satisfactory tune.

## Commands: preparation and optional archive audit

Run from the project root in PowerShell:

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py run_neat_smooth_study.py preflight
& $py run_neat_smooth_study.py prepare
& $py run_neat_smooth_study.py verify

# Evaluates saved genomes; does not evolve anything
& $py run_neat_smooth_study.py audit --workers 2

# These commands explicitly perform validation/tuning; preparation does not
& $py run_neat_smooth_study.py tune-pid --cap 1.0 --workers 2
& $py run_neat_smooth_study.py tune-pid --cap 0.5 --workers 2
& $py run_neat_smooth_study.py tune-pid --cap 1.5 --workers 2
```

Use `--run` to choose another output directory. Existing prepared protocols are verified rather than overwritten. To use code or protocol changes, create a new version/run directory. The existing prepared source bundle stays executable even if live experiment code is later edited; changing the frozen source, dependencies, settings, inputs or courses is rejected.

## Commands: controlled pilot — starts evolution

Both branches warm-start from the validated **seed-1 generation-134, native 1 m/s model**. The transferred 0.80-phase video model is not used. Identical evolution seeds give both branches identical initial populations; fitness can subsequently change their reproduction trajectories.

```powershell
# Each command starts only its own 100-generation branch
& $py run_neat_smooth_study.py pilot --branch old --workers 2
& $py run_neat_smooth_study.py pilot --branch smooth --workers 2

# For an interrupted branch, use the same command plus --resume
& $py run_neat_smooth_study.py pilot --branch smooth --workers 2 --resume

# Requires both branches to finish; verifies matching parents, references and all 100 batches
& $py run_neat_smooth_study.py review-pilot
& $py run_neat_smooth_study.py plot
```

If the archive audit found a qualifying controller, inspect `audit/complete.json` first. If you still want the controlled pilot, append `--force-pilot` to each pilot command. This is an explicit override of an unnecessary-evolution stop, not a change to the acceptance criteria.

`pilot/review.json` records the chosen branch. If neither branch qualifies, confirmation is blocked. Do not silently extend generations or relax gates; review the trade-offs and define a new protocol. If the old-fitness continuation wins, use it for confirmation rather than assuming the new reward is superior.

## Commands: five-lineage confirmation — after pilot review

The parents are the original seed-specific validated 1 m/s winners: generations **134, 136, 131, 107 and 134**. Each gets 100 additional generations. Confirmation has a separate speed bank, validation courses and evolution seeds. The development pilot is not counted as a confirmation run.

```powershell
& $py run_neat_smooth_study.py confirm --seed 1 --workers 2
& $py run_neat_smooth_study.py confirm --seed 2 --workers 2
& $py run_neat_smooth_study.py confirm --seed 3 --workers 2
& $py run_neat_smooth_study.py confirm --seed 4 --workers 2
& $py run_neat_smooth_study.py confirm --seed 5 --workers 2

# Resume only an interrupted seed; branch selection comes from the frozen pilot review
& $py run_neat_smooth_study.py confirm --seed 3 --workers 2 --resume
```

Five warm-started lineages test repeatability of the selected protocol. A one-lineage controlled pilot does not establish an algorithm-wide causal effect of fitness. Track-bank changes are shared between pilot branches; comparison with old frozen models also includes extra evolution and the new training distribution.

## Assessment, plots and corrected videos

```powershell
# Requires five completed qualifying NEAT winners and three qualified PID tunes
& $py run_neat_smooth_study.py assess --workers 2
& $py run_neat_smooth_study.py plot

# Exactly the validation-selected NEAT/PID pair at 1 m/s
& $py run_neat_smooth_study.py record --cap 1.0

# Optional .5 m/s transfer pair only if its validation windows also qualify
& $py run_neat_smooth_study.py record --cap 0.5
```

Assessment freezes five new NEAT winners, their five original references, and three cap-specific PID tunes **before** generating the committed-seed final test suite. It contains 150 tracks: 90 mixed (30 per difficulty), 30 speed/recovery (15 isolated, 15 post-turn) and 30 corner drills. At each cap it evaluates the ten NEAT policies and only that cap's PID: 4,950 episodes total. The primary cap is 1 m/s; 0.5 and 1.5 are transfer assessments. Reissuing `assess` verifies and reuses completed episode JSONs.

The report includes every failure, completion counts, per-seed spread, paired finish times with shared-success denominators, independent geometric error, command jitter, achieved mean/peak speed and sustained straight speed/yaw. `assessment/results.csv`, `assessment/summary.json`, `report.md` and PNG/SVG files under `paper-figures/` retain the evidence. Aggregate observed-window means do not replace each-window acceptance checks. Unequal-exposure partial-episode error means are labeled as observed, not whole-lap quality.

Videos use validation-selected models and correct full-speed PID configuration, not the fastest observed test lap. The preserved Rhythm Changes course is explicitly exploratory. Each rendered replay is checked against a headless episode, uses normal-time 15 fps playback and an approximately two-second final hold, and labels measured speed and cap. A 1.50 m/s final NEAT showcase is blocked because no new full-speed 1.50 NEAT stage is qualified yet. Do not generate a misleading six-video final set; review the 1 m/s outcome before designing that stage.

## Implementation verification

```powershell
& $py -m pytest tests/test_neat_smooth_study.py -q
```

Checks cover bounded fitness and failure avoidance, raw physical metrics and windows, independent quality-gate failures, ranking ties, geometry/split separation, shared batches, actual centered PID outputs, seed-specific exact-parent warm starts, source/input integrity, stubbed evolution dispatch, incomplete-assessment rejection, and checkpoint random-state/generation recovery with reproduction stubbed. A small deterministic PID physics smoke test verifies full-speed measurement and score-component accounting; it does not tune gains. No real evolution is run by these tests.

Delivery verification on 23 November 2026: **58 tests passed** across the new workflow and existing fast-study, PID, memory and episode regression suites. Preflight and the frozen-source `verify` command passed. All 360 prepared geometries were rechecked for validity, uniqueness and separation from protected prior tracks. The preparation contains no evolution checkpoints, tuned PID selections or final test suite. Machine-readable evidence is saved in `runs/neat-smooth-study-v1/implementation-checks.json`.

## Archive audit outcome — 24 November 2026

The completed audit found **0/14 qualifying existing candidates**. Completion remained 58–60/60 at 1 m/s and 20/20 at 0.5 m/s, but every candidate violated the speed, physical yaw and steering requirements on some straight windows. Perfectly centered isolated starts were smooth; post-turn recovery exposed oscillation. Seed 1 generation 134 averaged 0.972 m/s on isolated straights and 0.830 m/s on post-turn straights. The [audit report](../runs/neat-smooth-study-v1/audit-review/report.md) saves all candidate results, the five-parent breakdown, limitations, reproducible JSON and PNG/SVG figures. The planned pilot can proceed without `--force-pilot`; no pilot was launched while reviewing these results.

## Completed paired pilot outcome

Both seed-1 branches completed their 100-generation budgets. The frozen `review-pilot` command verified shared parents, evolution seed, references and all generation batches, and recorded `successful=false`: **neither branch produced a fully qualifying champion, so confirmation is blocked.** Actual training budgets were 60,000 episodes (old) and 60,012 (smooth), with 8,000 validation episodes each. All champions retained 20/20 low-speed completions.

Straight yaw/steering gates passed for 84/100 smooth-fitness champions and 0/100 old-fitness champions. The smooth run nevertheless sacrificed speed. Its generation-83 champion completed 60/60, improved geometric RMSE and whole-course jitter against the parent, and passed the straight motion gates, but sustained about 0.713 m/s and took 12.07% longer on shared successful tracks. It is an illustrative trade-off, not a qualifying winner. The final smooth champion was slower than the target and had worse overall command jitter despite low straight yaw.

The [paired pilot report](../results/neat-smooth-study-v1/pilot-comparison/report.md) records all champion metrics, preserved model paths/hashes, three PNG/SVG figure pairs and a diagnostic rescore showing that the new validation fitness can prefer a slower smooth controller. This supports revisiting the speed/smoothness reward balance in a separate protocol rather than automatically starting confirmation. These are one-lineage repeated-validation findings, not fresh test results. Runtime worker counts changed from 2 in the old run to 4/6/8 in the smooth run, so logged wall-time differences do not isolate fitness efficiency.

Regenerate the report without simulation or evolution:

```powershell
& .\.venv\Scripts\python.exe run_neat_smooth_study.py review-pilot
& .\.venv\Scripts\python.exe research/compare_smooth_pilots.py
```
