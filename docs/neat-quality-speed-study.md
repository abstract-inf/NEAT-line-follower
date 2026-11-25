# Quality-constrained speed development

This is a separate, prospective experiment following the failed old/smooth pilots. The user authorized autonomous training and comparison with PID, with every attempt documented. Previous prepared inputs, controller code, physics, rewards and results remain preserved.

## Question and scope

Can ordinary task-fitness design with **stock NEAT-Python 0.92** evolve faster driving while protecting completion, geometric tracking, command jitter and physical straight-line motion? The primary operating cap is 1 m/s, with at least 0.95 m/s sustained on every required validation straight. Slower cornering is allowed. This development pilot is not a claim that any controller is globally optimal or that NEAT is superior to PID.

The [published methodological basis](../research/Quality_Constrained_NEAT_Design.md) separates NEAT's evolutionary machinery from task fitness. The new scalar ordering is completion-first and constraint-aware, inspired by feasibility rules. It is not Deb's tournament algorithm: stock NEAT reproduction still uses its own species allocation, crossover, mutation and stagnation mechanisms.

## Frozen controller and evaluator

The neural policy retains the original 21 sensor-history inputs, feedforward network behavior, mirrored sensor frame, drive/steering mapping, motor physics, recovery features and mutation settings. No output filter, PID imitation, manually chosen corner maneuver or route-aware neural input is added. The evaluator alone uses physical positions and route geometry to measure reward and quality.

MuJoCo 4.0.0, 600 Hz physics, 60 Hz control, 20 mm paint, 2 mm floor sampling, the 100 mm corridor and existing sensor-aware departure/stall rules are unchanged. Preparation checks controller/physics source bytes and dependencies against the preserved predecessor before copying the complete execution source into the new run.

The development warm start is disclosed: smooth-pilot seed-1 generation 83. It previously met completion, accuracy, jitter and straight-motion conditions, but was too slow. One exact genome and 99 stock independently mutated copies initialize the population. The independent new evolution seed is derived from study seed `2026112401` and `quality-speed-development`. The frozen fast native-1 m/s generation-134 parent remains the quality and final-acceptance reference.

## Training budget and task fitness

The initial budget is **50 additional generations**, target population **100**, and **six common courses per genome**: an isolated long straight, a post-turn recovery course, intermediate mixed, advanced mixed, and two opposite-order corner drills. Speed-bank and development validation courses are deliberately reused from the development study. Fresh generation courses exclude all protected prior geometries. Actual counts are logged; stock population sizes may fluctuate.

Every generation also evaluates the frozen fast parent on the exact same six courses. This adds six reference episodes per generation, avoiding incomparable track-specific quality scales. All training episodes now measure independent geometric RMSE at the full 60 Hz control rate. This is an evaluator measurement, not a policy input.

For each candidate batch, let `C` be its completed-course count and `m=6`. Quality violations are positive relative exceedances of:

- Mean geometric RMSE above the same-batch frozen-parent mean, with a **5 mm training floor**.
- Mean command jitter above the same-batch frozen-parent mean, with a **0.0005 training floor**.
- Maximum required-window raw yaw RMS above **0.15 rad/s**.
- Maximum required-window normalized steering RMS above **0.02**.
- The fraction of required windows that were not observed.

The floors avoid a perfectly centered reference imposing a numerically zero training tolerance. They do not relax the original validation rules. Sum the violations as `V`; use `Q=1/(1+V)`. All finite sub-limit quality measurements have zero excess: the controller gets no extra reward for becoming slower to make already acceptable quality still smaller.

Define a bounded speed utility:

```text
S = 0.75 * min(minimum_required_straight_speed / 0.95, 1)
    + 0.25 * mean(min(successful_route_speed / 1 m/s, 1))
```

Missing windows have speed zero; failed episodes contribute zero route-speed utility and never use early termination as lap time. The scalar breeding fitness is:

```text
incomplete batch:      10*C + 9*(0.8*mean(progress) + 0.2*Q)
all complete, V > 0:   10*m + 9*Q
all complete, V = 0:   10*m + 10 + 9*S
```

These disjoint bands preserve completion priority, grade infeasible candidates rather than discarding them, and let acceptable candidates compete on speed. The inherited legacy, elapsed and continuous jitter terms are retained only as diagnostic episode data; they do not enter this new breeding fitness. This explicitly changes both task fitness and genome-level aggregation. Its discontinuity at feasibility is a declared design choice and may create a narrow search region; a failed run will be retained rather than silently relaxed.

## Calibration, validation and selection

Before evolution, rescore the 200 stored old/smooth development champions under this objective and test synthetic counterexamples. This calibration is retrospective development, not independent evidence. The result and input hashes are saved under `results/neat-quality-speed-v1/fitness-calibration.*`; smooth g83 is the only fully completed quality-feasible batch in those stored validation records.

Each generation's **top three distinct training genotypes** receive 60 primary validation episodes at 1 m/s and 20 retention episodes at 0.5 m/s. Exact duplicate genotypes can reuse deterministic fixed-suite results. Cache identity removes only the genome identifier and its stale fitness; source, configuration and suite are frozen by the protocol. This avoids presenting repeated identical physics as new independent evidence. All unique evaluation counts are logged.

The original acceptance rules remain: match parent completion within every course group at both caps; match/improve shared-finish geometric RMSE and jitter; paired time at most 1.05 times the fast parent; every primary straight observed and at least 0.95 m/s, yaw at most 0.15, steering at most 0.02. Candidates passing those rules use the existing completion/time-band/quality ranking. Validation does not change breeding fitness. No official winner is manufactured if no candidate qualifies.

The previously consulted validation is explicitly development selection data. After protocol development, a consistent independent five-lineage confirmation procedure and new validation/test suites must be frozen. The current g83 warm-start pilot alone cannot demonstrate training independence, repeatability or test generalization.

## PID and fair comparison

The saved base-1.0 filtered PID was first validated on its previously prepared, disjoint 60-course PID validation suite. It completed **60/60**, with minimum isolated-straight speed **0.9905 m/s**, and was retained. This is a satisfactory full-speed baseline; it is not the historical base-0.75 restricted-cruise controller. Its frozen selection and gain file are copied into the new protocol.

The new `baseline` command evaluates the stable g83, fast g134 and frozen PID on the **same** development courses, caps, robot and termination rules. Failures remain rows; paired time includes only common successful tracks and reports the denominator. The auxiliary 0.5 m/s PID evaluation transfers the 1 m/s tune and is labeled accordingly, not misrepresented as a separately tuned low-speed baseline. Final claims require frozen controllers and a fresh shared test after development and confirmation; no final test suite is opened by this pilot.

## Commands

From the project root in PowerShell:

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py run_neat_quality_study.py calibrate
& $py run_neat_quality_study.py prepare
& $py run_neat_quality_study.py verify
& $py run_neat_quality_study.py baseline --workers 6
& $py run_neat_quality_study.py train --workers 6

# Resume an interrupted stage, optionally with a different worker count:
& $py run_neat_quality_study.py train --workers 6 --resume
& $py run_neat_quality_study.py status

# Read saved logs and regenerate the development report and PNG/SVG figures:
& $py research/report_quality_training.py
& $py research/analyze_quality_networks.py

# Once complete.json exists, optionally make one verified archive of the completed stage:
& $py research/report_quality_training.py --archive
```

Use another `--run` directory for a genuinely different version; do not edit the frozen source/protocol to bypass checks. The first prepared run lives under `runs/neat-quality-speed-v1/`. Later commands execute its copied source in a child process; live edits do not alter an already frozen run.

The memory-limited worker configuration retains the existing 64 MiB mask-cache allowance and 128-task recycling. This is not a hard process-memory limit. Six workers are the initial runtime choice; changes are recorded in `runtime.csv`. Small four-course validation tasks can use all workers rather than restricting validation to two large tasks.

## Reproduction and interruption record

The saved `robot` object is the common physical template. Its inherited `max_speed` field is overridden by each evaluation task: 1.00 m/s for training/primary validation and 0.50 m/s for retention. The worker constructs the simulation with that task cap and records it as `cap_m_s` in every episode. Reporting checks those recorded caps; the inherited template value must not be mistaken for the actual experiment cap.

Save `prepared/protocol.json` plus `protocol.sha256`, all prepared inputs/courses and `prepared/source/`. The run retains all generated batches and reference episodes, per-genome fitness decompositions, training episodes, candidate genomes, validation cache and decisions, generation summaries, worker history, and the last atomically committed population/RNG checkpoint. There is an initial generation-zero checkpoint, so a first-generation interruption is resumable. Partial generation logs are trimmed back to the committed state. Exact cached validation can be reused because it does not change policy or fixed episodes.

Committed episode totals exclude unknown partial work from an interruption. Record interruptions and never claim those totals are exact compute costs including aborted work. Hardware/worker differences also prevent interpreting wall time alone as an algorithmic sample-efficiency improvement.

In this frozen V1 loop, a complete validation cache saved before an interrupted generation survives resume, while that generation's uncommitted counter is trimmed. Its later cache hit correctly avoids repeated physics, but the generation-level `newly_evaluated_validation_episodes` sum can therefore omit known earlier work. The reporting helper also counts complete cached episodes for distinct reported candidate genotypes and states both scopes. Neither count includes unknown partial physics or candidates from an as-yet unlogged generation. This affects cost accounting, not training fitness or validation selection; the running source remains unchanged.

Implementation checks:

```powershell
& $py -m pytest tests/test_neat_quality_study.py -q --basetemp .cache/pytest-quality
```

The short test-temp path avoids Windows path-length limits in this desktop sandbox's long default temp root. Tests cover priority bands, the acceptable-quality plateau, speed ordering, failure avoidance, graded infeasibility, missing measurements, phenotype-cache identity and a checkpoint/resume path with reproduction stubbed. They do not evolve a real population. Standard stock-core regressions are also run before launch. The experiment ledger records progress and unsuccessful attempts.

The reporting command reads only logged generations and their cached physical episodes. It preserves all three candidates, all rejection reasons, exact unique-genotype counts, actual episode counts and reporting-input hashes. Its fixed-suite validation plots are separate from training-component plots: new six-course batches change difficulty across generations, so a change in raw training fitness alone is not a fixed-task learning improvement. Interim reports are explicitly marked while evolution is running; regenerate after `complete.json` exists for the final development report.

The frozen generation field `quality_feasible_genomes` counts zero quality violation, including any incomplete genome whose measured quality still passes. Full training feasibility additionally requires completion of all six courses: use `tier == completed_quality_feasible` in `genome-fitness.csv`, as the training-component plots do. Neither count means that a candidate passed the separate 80-episode validation rules.

The optional archive rejects an unfinished stage or an existing completed archive. It includes frozen execution source, prepared inputs, all stage logs/models, the final resumable population/RNG checkpoint, reports and a per-file SHA-256 manifest; CRCs are checked before publishing the ZIP. The external manifest records the ZIP hash. Restore to a separate clean project root with the recorded Python and dependencies, preserving directory paths; do not extract onto an active experiment. The earlier `prepared-development-bundle.zip` remains a separate preparation-era snapshot.
