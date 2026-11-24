# Continuing training with progressive tracks

This is a separately recorded continuation of the full evolved population, not
an independent evolution run. The parent run remains intact. No NEAT evolution
was started while implementing or validating these tools.

## Run the continuation

From the project directory:

```powershell
.venv\Scripts\python.exe continue_training.py --dry-run
.venv\Scripts\python.exe continue_training.py --visualize
# Or, without a window:
.venv\Scripts\python.exe continue_training.py --headless
```

The first normal invocation prepares `runs/28-10-2026-hard`; subsequent
invocations resume that phase's latest checkpoint. `--prepare-only` saves or
validates its configuration without training. Generations continue at 100 and
run through 199. The checkpoint currently contains 202 genomes: NEAT-Python's
species allocation can differ from the nominal population of 200. It resumes
all of them, rather than initializing from the selected `best.pkl`.

Edit the constants at the top of `continue_training.py` before preparing a new
phase. Existing phases always restore their saved research settings. To try
another configuration, choose a new `--output` folder; never overwrite a run.
`--visualize` and `--headless` remain runtime choices. Closing the observer
continues headlessly. The viewer does not pace training.

The fresh-session launcher, `run_training.py`, also exposes ten episodes,
aggregation and progressive training settings. The old `curriculum`, `multi`
and `single` strategies remain available for the original study baselines.

## Evidence and selected controls

The parent selected generation 38, with 16/30 validation completions. Beginner
completion was 10/10; intermediate and advanced were each 3/10. Its nine
validation routes containing zigzags failed, but route-level feature tags do not
identify the location or cause of failure. The last five training champions
completed 5/5 beginner, 3/5 intermediate and 2/10 advanced episodes.

The continuation therefore starts at adaptive stage 1, based only on parent
training data. A fresh adaptive run starts at stage 0. Validation is reused for
model selection; neither validation nor test performance controls the curriculum.

Default continuation controls:

- 100 additional generations, nominal population 200, ten distinct fresh tracks
  per genome per generation. Every genome faces the same batch. Nominal training
  allocation is 200,000 episodes; actual offspring counts are logged.
- GPU backend and batch size 200 retained. Tracks are sequential batches of
  independent worlds; ten slots do not create 2,000 simultaneous GPU worlds.
- Wheel-rim target ceiling 0.35 m/s, increased from 0.30 at the user's request.
  This permits, rather than forces, faster motion. Slowing, stopping, reversing,
  torque limit, response filtering, dimensions, sensor layout, mass, friction
  and the 600/60 Hz physics/control rates are preserved.
- Original episode fitness retained by default, with optional completion speed
  bonus disabled (`COMPLETION_SPEED_BONUS=0`). Earlier completion already
  receives a larger score through the existing elapsed-time term. Speed alone
  is not a useful fix for off-route failures.
- `AGGREGATION="mean_min"` retains the original baseline. Optional `mean_cvar`
  is available for a separate recorded comparison; it is not assumed superior.

These simultaneous exposure/speed changes define an exploratory continuation,
not an isolated test of any one intervention. Claims about which change helps
require controlled ablations and independent evolution seeds.

## Training-only adaptive curriculum

Each stage uses the following ten-slot allocation. Other positive episode
counts are supported with deterministic rotation of the same mixture.

| Stage | Beginner | Intermediate | Advanced |
|---|---:|---:|---:|
| 0 | 10 | 0 | 0 |
| 1 | 5 | 5 | 0 |
| 2 | 2 | 6 | 2 |
| 3 | 2 | 3 | 5 |
| 4 | 1 | 2 | 7 |

For each generation, take the top 10% of genomes by training fitness and pool
completion counts by difficulty. Use the last five generations, with at least
five generations in the current stage. Advanced tracks enter after intermediate competence (or the explicit plateau fallback). Stage 0 advances at 85% beginner
completion; stages 1 and 2 advance at 70% intermediate completion; stage 3
advances at 55% advanced completion. Later stages contain longer/more complex
routes, so the same difficulty label does not mean an unchanged task.

Stages never decrease, and history is cleared after advancement. This limits
oscillation caused by noisy batches. At 25 generations without competence-based
advancement, move one stage to increase exposure while retaining easier tracks.
That transition is explicitly logged as `plateau_exposure`, not mastery. It is
an anti-stall policy with a tradeoff: it can expose an unready population to
harder tasks. All thresholds, windows, minimum/maximum duration and the elite
fraction are configurable in `CurriculumConfig`.

Curriculum stage, age and rolling counts are included in every checkpoint.
Track seeds depend on the continuing generation and episode slot. Resuming
restores the exact schedule and history; it does not restart beginner training.

## Geometry and route length

The original saved training routes spanned:

| Difficulty | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Beginner | 1.21 m | 1.90 m | 2.52 m |
| Intermediate | 2.80 m | 4.08 m | 5.45 m |
| Advanced | 3.25 m | 5.09 m | 7.25 m |

The new progression extends that measured envelope gradually:

| Stage | Intermediate motifs / length | Advanced motifs / length |
|---|---|---|
| 0 | 3 / 1.0–2.8 m, unused by default | 6 / 2.8–5.5 m, if requested |
| 1 | 4 / 1.7–3.8 m | 6 / 2.8–5.5 m, unused by default |
| 2 | 6 / 2.7–5.5 m | 8 / 3.5–6.8 m |
| 3 | 8 / 3.5–6.8 m | 10 / 4.5–8.5 m |
| 4 | 8 / 3.5–6.8 m | 12 / 6.0–10.0 m |

Beginner routes retain four motifs and a 1.0–2.8 m interval throughout. Length
bounds and motif count are independent configuration parameters. Set explicit
`GeneratorConfig` bounds/counts, or change curriculum `length_scale` and
`motif_scale` independently. The exact effective parameters for every training
track are written to its generation's `batch.json`.

Sharper curves appear gradually; minimum advanced arc radius decreases from
0.16 to 0.12 m. Genuine sharp corners remain unsmoothed. Alternating difficult
slots guarantee zigzags, and later advanced focused routes require two zigzag
motifs. Up to three gaps and two crossings/branches of each kind are attempted
at the final stage, subject to the existing geometry constraints. Requested
feature exclusions remain enforced. Challenge placement can be rejected if it
cannot fit safely; the serialized geometry is the actual condition.

The arena stays 4 x 4 m with a 1 mm raster. Finite search, robot clearance,
ordered routes, intersection rules and split/hash checks remain active. Dense
routes use up to 1,600 geometry attempts and at most 100 deterministic seeds per
slot. Failed seeds/reasons are saved in `batch.json`; constraints are never
relaxed to force a track through. More slots increase total work and CPU mask
turnover, while the four-mask cache and GPU world batch remain bounded.

A geometry validation is not a guarantee that the current sensor controller
can finish a route. The inspection tool also runs a ground-truth route-following
oracle solely to demonstrate mechanical feasibility. Its results are excluded
from controller comparisons and cannot establish NEAT/PID performance. The
controller remains feedforward with 15 binary inputs; all-zero observations
still lack history about which side last saw the line.

## Time limits and aggregation

With `EPISODE_TIME_LIMIT=None`, the continuation uses:

```text
T = clip(route_length / 0.10 m/s, 45 s, 120 s)
```

The 0.10 m/s reference is a conservative route-average allowance, not a motor
command. It matches the original 0.30 m/s speed cap divided by the original
time multiplier of three. Routes below 4.5 m receive 45 seconds; 9 m receives
90 seconds. Stalling for five seconds without 10 mm new valid progress, or
persistent corridor departure, still terminates early. Every episode row
records its actual `time_limit_s`. The original uncapped formula is available
by leaving the optional bounds/reference unset.

With episode scores `F_i`, the original aggregation is:

```text
0.5 * mean(F_i) + 0.5 * min(F_i)
```

The optional alternative is:

```text
0.5 * mean(F_i) + 0.5 * lower_tail_CVaR(F_i, alpha)
alpha = 0.30
```

At ten episodes this averages the three lowest scores. Fractional tail mass
uses a weighted boundary score, rather than silently rounding to another
fraction. A single worst score has local weight 0.55 under mean/min versus
about 0.217 under the three-score tail, away from rank changes. Larger batches
increase the opportunity to encounter a poor minimum; they do not by themselves
prove the baseline is unsuitable. CVaR can soften sensitivity but also reduce
the incentive to fix one catastrophic failure. Keep both as explicit conditions.

## Versioning and evaluation limits

`benchmarks/suite/suite.json` and all existing validation/test files are
unchanged. A separately versioned progressive-training manifest reuses their
frozen identities and declares training generator 2.0. The prepared phase keeps
its own `suite.json` with those identities, generator settings and parent hash.
Parent training history is retained to reject duplicate geometry across phases.
Training never evaluates test tracks or reads their outcomes.

The original held-outs remain useful for comparison with the original run,
but do not cover the entire new long-route distribution. To freeze an
additional, separate long-route evaluation cohort before final experiments:

```powershell
.venv\Scripts\python.exe -m line_follower prepare-suite --output benchmarks/long-evaluation-v2 --seed 20261029 --generator-config configs/progressive-generator.json --protect-suite benchmarks/suite/suite.json
```

This generates version 2 beginner/intermediate/advanced held-outs using the
final-stage geometric ranges and excludes the original reserved geometries.
It does not train or evaluate controllers. Do not use final test outcomes to
tune the generator or policy. Long-route generalization claims require actual
held-out long-route evaluation, not inference from training success alone.

## Diagnostics, timing and practical cost

Existing CSV files remain available. Additional records are:

- `diagnostics.csv`: population/champion completion, failure reasons, observed
  progress and lateral RMSE by difficulty, feature presence and length bin.
  Feature presence is an association, not causal failure attribution. Population
  episode rows and repeated validation evaluations are not independent trials.
- `curriculum.csv`: current/next stage, stage age, rolling training competence,
  and the reason for advancing or staying.
- `gpu-profile.csv`: per-batch setup, network inference, physics/transfer,
  accounting/observer, compaction, maximum control-block wall duration,
  simulated work and padding. These are host wall intervals, including waits
  and scheduling, not isolated CUDA kernel timings. No extra GPU synchronization
  is introduced for profiling.
- Generation CSV timing separates track generation, training and validation.
  The generation wall time remains wall duration, not simulated duration.

Generation 27 took 7,181 s, 26.6 times the run's median wall time, while its
executed GPU work was only about 1.04 times the median. The aggregate records
cannot distinguish suspend/sleep, scheduling, contention, thermal effects,
compilation or a GPU stall. Do not claim a cause from that row alone.

A fixed-network, 32-world, two-second-capped GPU probe measured approximately
5.22 s for four slots and 13.30 s for ten. This is an engineering throughput
probe, not new trained performance or a full-generation runtime guarantee.
Scaling the original median generation time solely by 10/4 gives about
675 s/generation or 18.7 hours per 100 generations. Longer routes and different
failure patterns can change that substantially.

```powershell
# No evolution: inspect all stages, plot routes and check mechanical feasibility.
.venv\Scripts\python.exe benchmark_training.py --inspect-only --output runs/my-track-inspection
# No evolution: benchmark fixed checkpoint networks.
.venv\Scripts\python.exe benchmark_training.py --worlds 32 --seconds 2 --backend gpu --output runs/my-throughput-probe
# Plot recorded training and diagnostics after generations have completed.
.venv\Scripts\python.exe plot_training.py --run runs/28-10-2026-hard
```

## Checkpoint integrity and research records

The continuation preserves weights, topology, species membership, ancestry,
index counters, random state and seen-track hashes. Cached genome fitness,
species stagnation scores and best-model selection reset because the task has
changed. Phase physics/GPU work counters start at zero; parent cost remains
recorded separately. Parent checkpoint/source/model/metadata copies and a full
source/settings transition audit are retained. `robot.json`, `episode.json`,
`geometry-freeze.json`, source snapshots and the versioned suite support fair
PID or future PPO evaluations under the same limits.

Rerunning the same continuation command restores the last completed generation;
an interrupted generation is repeated. Source/dependency mismatches still block
ordinary resume. This deliberate scientific phase transition is separate from
`prepare_resume.py`, which only permits audited infrastructure recovery changes.


