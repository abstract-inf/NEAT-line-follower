# Faster NEAT driving with protected tracking and smoothness

This study starts five new runs from the five saved 0.50 m/s winners. It tests whether a different fitness function and more speed practice can produce faster driving while meeting validation limits for completion, geometric tracking error, and command jitter.

NEAT-Python 0.92 executes the networks using its existing feedforward evaluator. The robot, 21 sensor/history inputs, drive/steering action mapping, mutation configuration, and physics settings come from the previous experiment. Every champion records whether its drive and steering outputs are evaluated by the library.

## Implementation changes

- `run_neat_fast_study.py` prepares and verifies the experiment, launches one seed, resumes it, freezes final selections, assesses them, and generates plots.
- `line_follower/fast_study.py` builds the speed courses and six-course batches and applies the validation quality gate.
- The existing training engine accepts the explicit `fast` profile and a frozen reference. Its checkpoint also binds that reference. It saves champions even when none qualifies as a winner.
- The shared episode loop adds optional geometric RMSE and central-straight timing, plus peak forward speed. Fitness has a versioned route-speed option and named component contributions. Existing configurations retain the original fitness by default.
- `tests/test_neat_fast_study.py` checks geometry, split separation, rewards, gating, seed dispatch, checkpoint recovery, assessment freezing, and figure generation with evolution stubbed.

The installed NEAT library is fingerprinted with the experiment. Disconnected-node behavior is retained and observed through output-evaluation flags.

## Training protocol

Each seed has a target population of 100 and 300 generations, using eight CPU workers.

| Generations, zero-based | Wheel-speed cap |
| --- | --- |
| 0–49 | 0.65 m/s |
| 50–99 | 0.80 m/s |
| 100–149 | 1.00 m/s |
| 150–199 | 1.25 m/s |
| 200–299 | 1.50 m/s |

Every genome in a generation faces the same six courses. Two come from a saved bank of twenty speed courses, two are fresh mixed courses at intermediate and advanced difficulty, and two are fresh corner drills. Left and right drill directions are balanced. A seeded shuffle cycles through the twenty speed courses once every ten generations. These repeats are training practice.

The bank contains six long straights, six gentle curves, four broad S-curves, and four long straights ending in a 45° or 90° turn. Initial straight lengths range from 3 to 6 m, gentle curve radii from 1 to 3 m, and paint width is 20 mm. Each speed course is centered in an arena that fits it, using 2 mm floor sampling. Fresh mixed courses and corner drills use their existing generator settings.

Preparation freezes 60 validation courses (30 mixed, 20 speed, 10 corner) and 150 test courses (90 mixed, 30 speed, 30 corner). Their geometry is distinct from training and protected prior experiments. Fresh straight validation and test courses are at least 4 m long. A published suite hash and individual file hashes identify the exact courses.

Validation repeats the bank's 6/6/4/4 proportions with new geometry. The thirty test speed courses use 9 straights, 9 gentle curves, 6 S-curves, and 6 straight-to-turn courses. Mixed validation and test courses are balanced across beginner, intermediate, and advanced difficulty.

## Fitness

The new option is called `route-speed-1.0`. For progress p, completion indicator C, elapsed time t, timeout T, intended route length L in meters, legacy return R over N control steps, and normalized command jitter J:

```text
F = p + C - 0.1*t/T + 0.01*p*tanh(R/max(1,N))
    + 0.50*C*min(L/(t*1 m/s), 1)
    - 0.20*J
```

The speed bonus requires completion. Distance is the intended route length, so wheel spin and extra travel cannot increase it. J is the mean of (delta_left² + delta_right²)/8 over successive clipped wheel commands at 60 Hz. Each fitness term is logged separately as a mean over training or validation episodes.

Population selection retains the existing mean/CVaR aggregation. With six episodes and tail fraction 0.5, the aggregate gives equal weight to the six-course mean and the mean of the three lowest episode scores.

The evaluator keeps the 100 mm corridor half-width, two-second departure allowance with no line sensor contact, five-second stall check, and existing timeout settings. The fitness version is separate from these stopping rules. Existing episode configuration files default to the earlier fitness.

## Validation and winner selection

Preparation measures each old ramp winner on all sixty new validation courses at a 1.50 m/s cap. These five frozen records are the quality references for the corresponding five new runs.

Each generation's training champion is validated at the current cap. Selection of the final winner begins at generation 200. A candidate qualifies when its completion count matches or exceeds its reference within each course group, and its mean geometric RMSE and command jitter match or improve on the reference. The two mean metrics are checked both over all episodes and over shared successful courses. A numerical tolerance of 1e-12 handles arithmetic rounding.

Geometric RMSE uses exact distance from the chassis center to the intended route polyline. This diagnostic is independent of the evaluator's local gate cursor. It is calculated for validation and final assessment. Training retains the original local tracking diagnostic and uses the new fitness.

Among qualified candidates, select the lowest mean paired completion-time change relative to the reference on shared successful courses. Break ties by higher total completion and then lower geometric RMSE. Equal ranking retains the earliest candidate.

On dedicated straight courses, interpolate the times at which the axle's authenticated progress crosses the central 2 m interval. Sustained straight speed is 2 m divided by that interval's elapsed time. The approximately 1 m/s goal is recorded when every dedicated validation straight is completed and has sustained speed of at least 0.95 m/s. This diagnostic accompanies qualification; a quality-qualified winner can still fall short of the speed goal. Faster qualification is measured by a negative paired time change.

A run can finish with no qualifying winner. It then saves that outcome and the final-phase candidate with the fewest quality violations, followed by higher completion and lower RMSE, as a diagnostic trade-off. That candidate remains labeled as diagnostic. All generation champions remain available for inspection.

## Terminal workflow

Open PowerShell in the project directory. Set the interpreter variable in each new terminal.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py run_neat_fast_study.py preflight
& $py run_neat_fast_study.py prepare
```

Preparation generates courses, copies verified parent models and PID files, and evaluates the old reference policies. It starts no evolution. Repeating preparation verifies the frozen data.

Directory publication retries temporary Windows locks. If renaming stays locked, it copies the verified data and writes the checksum manifest last. If preparation is interrupted during publication, archive the incomplete `prepared` folder before running `prepare` again.

Start the first new seed yourself:

```powershell
& $py run_neat_fast_study.py train --seed 1
```

After an interruption with a committed checkpoint:

```powershell
& $py run_neat_fast_study.py train --seed 1 --resume
```

The first checkpoint is saved after generation 0. If interrupted before that checkpoint, archive the entire `seed-XX` folder, including `start.json`, and launch that seed again without `--resume`. The launcher reports this explicitly.

In a second terminal, watch the actual population:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m line_follower preview-training --run runs/neat-fast-study/seed-01/training
```

Replay saved champions on several validation courses and the exploratory long course:

```powershell
$py = ".\.venv\Scripts\python.exe"
$champions = "runs/neat-fast-study/seed-01/training/champions"
$validation = "runs/neat-fast-study/prepared/suite/validation"
$courses = @(
    "$validation/speed-000.json"
    "$validation/002.json"
    "$validation/corner-000.json"
    "runs/long-course-speed-challenge/track.json"
)
foreach ($generation in 134, 258) {
    $model = "$champions/generation-{0:D4}.pkl" -f $generation
    foreach ($course in $courses) {
        Write-Host "Generation $generation | $course"
        & $py replay_neat_speed_study.py --model $model --track $course
        if ($LASTEXITCODE -ne 0) { throw "Replay failed" }
    }
}
```

This opens one viewer at a time: a long straight, an advanced mixed course, a corner drill, and the exploratory long course. Close each viewer to advance. **R** restarts the episode, **Space** pauses, and **T** toggles the route overlay. The replay helper uses the model's saved robot and episode settings, so generation 134 runs at its 1.00 m/s cap and generation 258 at its 1.50 m/s cap. These checkpoints are diagnostic examples, not accepted final winners. Replace the generation numbers to inspect other saved champions. Do not substitute final test tracks while exploring or tuning.

Plot saved training and validation logs:

```powershell
& $py run_neat_fast_study.py plot
```

After reviewing seed 1, run the other four sequentially:

```powershell
foreach ($seed in 2..5) {
    & $py run_neat_fast_study.py train --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Training stopped for seed $seed" }
}
```

After all five runs finish, freeze and assess the selections:

```powershell
& $py run_neat_fast_study.py assess
& $py run_neat_fast_study.py plot
```

All paths default to `runs/neat-fast-study/`. Choose a different directory by placing `--output runs/another-name` before the subcommand. The preparation snapshot contains the executable sources, settings, documentation, and hashes needed to identify the run. Resume requires matching source, dependencies, parent, reference, and prepared courses.

## Assessment and outputs

The assessment copies the qualified new winners, five old ramp references, and three filtered PID configurations into a frozen folder before loading test courses. The PID files are the saved 0.50 m/s tune and the newer 1.00 and 1.50 m/s tunes from the PID follow-up. Every included policy runs at all three caps on the identical 150 test courses. Cap-specific PID is the primary comparator at each cap.

If some seeds have no qualifying winner, report those outcomes and assess the available winners and all references. A missing winner is part of the experiment's result.

- `prepared/protocol.json` records the study settings, seed mapping, input hashes, and prepared file hashes.
- `prepared/references/` holds the five reference measurements.
- `seed-XX/training/` contains checkpoints, all generation champions, training and validation episode logs, output evaluation flags, fitness components, and quality decisions.
- `seed-XX/training/best.pkl` exists when a final-phase champion qualifies.
- `seed-XX/training/selection.json` reports the qualified winner or the no-winner outcome and diagnostic trade-off.
- `assessment/results/` contains raw test episodes, grouped quality and speed summaries, paired comparisons, and result checksums.
- `plots/` contains SVG and 300 dpi PNG learning, fitness component, and speed/quality figures.

The plots include training fitness and validation tracking curves, consistent colors for the six fitness contributions, the primary speed/quality comparison, and paired before/after seed results at all three caps. Missing qualified winners remain missing in the comparison. The straight-speed summaries include both measured and expected segment counts so incomplete runs cannot silently pass the speed goal.

Report per-seed completion, route progress, finish time on shared successful tracks, geometric RMSE, command jitter, mean and peak forward chassis speed, and sustained straight speed. Compare mean speed with the requested wheel-speed cap using their explicit units and definitions.

Validation criteria protect selection on the validation suite. Fresh test results measure whether this protection generalizes. The earlier long course remains an exploratory benchmark. This experiment combines changes in fitness, course mixture, and episode budget; a matched-budget control is required to attribute an improvement to an individual change.

## Recorded results

The [seed 1 results summary](../research/NEAT_Fast_Study_Seed_01_Results.md) records the completed 300-generation run, representative checkpoints, quality-check outcomes, figures, and limits on interpretation. It contains validation findings; the five-seed final assessment is still pending.

The subsequent [1.00 m/s PID smoothness refinement](../research/PID_100_Smoothness_Results.md) provides a separately saved tune and replay commands. Its PID-only tests do not change this study's frozen baseline inputs. Future comparisons using the newer tune must identify that configuration explicitly.

The [historical restricted-cruise comparison](../research/NEAT134_vs_Tuned_PID_100_Results.md) used PID base 0.75 and cannot support a claim against full-speed-capable PID. The [corrected full-cruise comparison](../research/NEAT134_vs_Fullspeed_PID_100_Results.md) uses PID base 1.00 with retuned steering/filtering/slowdown, verifies sustained near-1 m/s travel, and assesses both frozen controllers on another 150 fresh shared tracks. It records frozen paths, hashes, metrics, and the updated ten-track replay command. Both comparisons are separate from the original five-seed protocol.
