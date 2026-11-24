# Repeat the training that produced fast NEAT generation 134

Recorded 20 November 2026. Launcher: `run_fast_neat_replicates.py`. This is a thin wrapper around the existing frozen `run_neat_fast_study.py` protocol, not a new fitness or curriculum. It previews by default and starts evolution only with `--start`. Seeds run sequentially, each with eight CPU workers. It skips verified completed runs and stops on errors.

## What is being repeated

The successful lineage has two relevant stages: 150 generations from scratch at 0.50 m/s, followed by a new 300-generation fast curriculum seeded from that low-speed winner. The older 150-generation short ramp is **not** the parent of the fast stage; its winner supplies a validation quality reference only.

All five low-speed stages already finished. Fast seed 1 also finished; fast seeds 2–5 have not started as of this check. The launcher reuses each corresponding frozen low-speed winner, rather than retraining those completed stages or cloning seed 1 into the other seeds. Each fast stage starts with one exact parent copy and 99 independently mutated copies. Evolution and track streams differ between seeds; validation geometry is shared.

Generation 134 means the 135th generation of fast seed 1 (zero-based indexing). It occurred in the 1.00 m/s phase. It was an exploratory checkpoint, not a final quality-gated winner. No final-phase champion in seed 1 passed every original quality gate. Repetition preserves that outcome and those rules; it cannot guarantee another generation 134 or silently promote an earlier checkpoint to a formal winner.

## Stage 1: already completed low-speed learning

150 generations at 0.50 m/s, target population 100, four training episodes per genome. Each generation shares three fresh corner drills plus one fresh procedural transfer course. The adaptive drill stages use these difficulty labels:

| Drill stage | Beginner drills | Intermediate drills | Advanced drills |
| --- | ---: | ---: | ---: |
| 0 | 3 | 0 | 0 |
| 1 | 2 | 1 | 0 |
| 2 | 1 | 2 | 0 |
| 3 | 0 | 2 | 1 |
| 4 | 0 | 1 | 2 |

The fourth course is beginner in generations 0–29, intermediate in 30–82, and advanced in 83–149. Drill geometry progresses from gentle corners to sharp alternating turns. Drill-stage changes depend on top-10%-population training completion over four generations, with a minimum six generations per stage, thresholds 0.80/0.70/0.55 by target difficulty, and a 101-generation exposure limit. Consequently there is no single fixed count of beginner/intermediate/advanced episodes across the entire stage. Its saved logs contain the realized allocation.

Each generation's training champion is evaluated on 30 fixed validation courses: ten beginner, ten intermediate, ten advanced. The saved low-speed winner is selected by validation completion, then validation fitness. Nominal budget: 150 × (100 × 4 + 30) = 64,500 episodes per lineage.

## Stage 2: the fast run you will start

300 generations, target population 100, six training episodes per genome, eight CPU workers, MuJoCo 4.0.0 and NEAT-Python 0.92. Existing network evaluation, inputs, action mapping, mutation, physics, and recovery rules are retained.

| Generations (zero-based) | Count | Wheel-speed cap |
| --- | ---: | ---: |
| 0–49 | 50 | 0.65 m/s |
| 50–99 | 50 | 0.80 m/s |
| 100–149 | 50 | 1.00 m/s |
| 150–199 | 50 | 1.25 m/s |
| 200–299 | 100 | 1.50 m/s |

The cap is an allowed wheel-speed target, not a required measured chassis speed. This reproduces the original full schedule; it does not stop at 1 m/s. All generation champions, including the 50 at 1 m/s, are saved for later inspection.

Every generation uses this shared batch:

| Training course type | Episodes per genome |
| --- | ---: |
| Speed courses from a saved bank | 2 |
| Fresh intermediate mixed course | 1 |
| Fresh advanced mixed course | 1 |
| Fresh advanced corner drills, opposite starting directions | 2 |
| **Total** | **6** |

Thus the batch has zero beginner, one intermediate, three advanced, and two separately classified speed courses. There is no difficulty-stage progression in this fast stage; the speed cap progresses. Corner sequences alternate between generations.

The 20-course speed bank contains six long straights, six gentle curves, four broad S-curves, and four straights ending in sharper turns. Initial straights are 3–6 m, gentle radii 1–3 m, line width 20 mm, and floor sampling 2 mm. A seeded shuffle visits all 20 bank entries every ten generations. The four other courses are freshly generated each generation and checked against protected geometry. Across 300 generations, this uses 600 speed-bank slots plus 1,200 fresh course geometries, rather than a fixed bank of six tracks throughout.

## What happens to each genome and why validation comes afterward

1. Every genome drives all six training courses from reset conditions. An episode ends on completion, timeout, stall, or qualifying off-route departure.
2. Each episode scores route progress, completion, elapsed time, a small legacy term, successful route-average speed, and command jitter. The speed bonus is `0.50 × completed × min(route_length / completion_time / (1 m/s), 1)`. Jitter penalty remains `0.20 × command_jitter`. Speed alone cannot earn the completion-speed bonus on a failed course. RMSE is a validation quality criterion, not a new direct fitness penalty.
3. A genome's evolutionary fitness is half its six-score mean plus half the mean of its three worst scores. This makes weak-course performance matter. These training scores drive reproduction, mutation, and species evolution.
4. **Only the single highest-training-fitness genome** is then run on the 60 fixed validation courses. The other 99 nominal genomes do not each receive 60 additional episodes.
5. Validation records generalization and controls checkpoint/winner selection. It does not replace the genomes' training fitness, train network weights on those episodes, or choose the next training batch. Because it is reused for selection, validation is not the final independent test.

The validation set contains 30 mixed courses (10 beginner, 10 intermediate, 10 advanced), 20 speed courses (6/6/4/4 types as above, new geometry), and 10 advanced corner drills. Group labels and difficulty labels are different: the corner courses are an additional advanced-labelled group.

Validation runs at the generation's current cap. Its frozen quality reference is that seed's old ramp winner evaluated on the same tracks at 1.50 m/s. Formal selection begins only at generation 200, when candidate and reference caps match. A candidate must match or exceed completion separately in mixed/speed/corner groups and match or improve mean geometric RMSE and command jitter both across all observed episodes and across shared finishes. Qualified candidates rank by mean paired finish-time change, then completion, then RMSE. The sustained-straight goal is at least 0.95 m/s over the central 2 m on every dedicated validation straight. If nobody qualifies, the outcome remains “no qualifying winner”; saved checkpoints still allow analysis of trade-offs.

At exactly 100 genomes, one generation costs 600 training plus 60 validation episodes = 660; 300 generations cost 198,000 episodes per seed, or 792,000 for the four remaining fast runs. NEAT-Python reproduction can yield slightly different actual population counts (for example 98 or 102), so these are nominal budgets; logs record actual counts. The full low+fast lineage budget is nominally 262,500 episodes over 450 generations, with the low stage already paid for.

## Terminal commands

Run from the project folder in PowerShell:

```powershell
$py = ".\.venv\Scripts\python.exe"

# Verify frozen inputs and preview the pending runs; starts no evolution
& $py run_fast_neat_replicates.py

# Start seeds 2, 3, 4, 5 sequentially
& $py run_fast_neat_replicates.py --start

# Or start only seed 2
& $py run_fast_neat_replicates.py --seeds 2 --start

# After interruption: resume started seeds, start untouched seeds, skip completed seeds
& $py run_fast_neat_replicates.py --start --resume

# Plot a run, including its validation history
& $py plot_training.py --run runs/neat-fast-study/seed-02/training
```

Use one launcher terminal at a time. Training remains a foreground process; keep it running. Resume requires a committed checkpoint (the first is saved after generation 0). If interruption happened before any checkpoint, preserve/archive that entire seed folder, including `start.json`, before restarting it; do not remove only selected files or reprepare the frozen protocol to bypass checks.

Outputs remain under `runs/neat-fast-study/seed-XX/training/`: population checkpoints, generation champions, training episodes, validation diagnostics, fitness components, quality checks, and status. The wrapper delegates saving and resume to the original engine. Its preview verifies frozen hashes and never starts evolution or assessment.

The independent prepared test set contains 150 courses: 90 mixed (30 per difficulty), 30 speed, and 30 corner. Do not use test outcomes to pick a promising 1 m/s checkpoint. The original `assess` command evaluates formal quality-gated selections and reports when none exists; exploratory 1 m/s checkpoint comparisons require a declared validation selection rule and a fresh held-out assessment. See [full fast-study protocol](fast-neat-study.md) and [generation-134 comparison](../research/NEAT134_vs_Fullspeed_PID_100_Results.md).

## Resume after the memory interruption (21 November 2026)

Seed 5 stopped during generation 173 with a raster-allocation MemoryError. The checkpoint preserves completed generations 0–172 and resumes at generation 173. Seeds 1–4 are complete. Use the unchanged launcher in a fresh process:

```powershell
& .\.venv\Scripts\python.exe run_fast_neat_replicates.py --start --resume
```

Completed seeds are skipped. This initial recovery applied no cache workaround or training-setting change. Memory exhaustion subsequently recurred at generation 198; use the optional recovery mode below for that interruption.

### Repeated memory error at generation 198

The checkpoint now preserves completed generations 0–197 and restarts generation 198. The same Pillow-to-NumPy raster allocation failed again. At inspection, Windows reported about 1.6 GiB of available physical memory on a 15.7 GiB machine, with about 4.6 GiB of remaining system commit capacity. Those readings describe the inspection time, not the exact crash time. The six generation-198 courses retain approximately 17–39 MiB each for their two grayscale rasters; temporary conversion buffers and other simulation/process allocations add to this.

The original cache keeps up to four masks per worker and evicts after allocating a replacement. The optional `--memory-safe` mode evicts before allocating, limits raster cache retention to 64 MiB per worker, and replaces CPU workers after 128 evaluation tasks. A task is one genome on its training batch or one champion validation episode. Replacement retains eight concurrent worker slots and does not reset evolution or the curriculum. Masks larger than the cache budget are constructed and used without cache retention. The budget does not bound total process memory, including the simulator's active mask and temporary buffers.

```powershell
# Verify/preview only
& .\.venv\Scripts\python.exe run_fast_neat_replicates.py --seeds 5 --resume --memory-safe

# Resume evolution from generation 198
& .\.venv\Scripts\python.exe run_fast_neat_replicates.py --seeds 5 --start --resume --memory-safe
```

The implementation lives in `fast_study_memory.py`, outside the frozen engine. Its runtime evaluator preserves the original track rasterization, observations, physics, fitness, selection, RNG handling, population, worker count, and checkpoint validation. It does not bypass mismatched fingerprints. This is an explicitly recorded execution adjustment, not a claim that every historical execution detail is unchanged. On actual launch, copies and hashes of the helper and launcher plus the cache/recycling settings are saved under `seed-05/memory-recovery/<invocation>/` before training. Cache misses and process startup can increase elapsed runtime; document the adjustment when comparing computational costs.

Verification injects a constrained-allocation failure into the original cache path, checks eviction before allocation, compares all six generation-198 masks and short real episode results, and verifies identical serial/parallel results while Windows workers are actually replaced. Launcher tests stub evolution. No evolution is started during these checks. Global memory exhaustion can still occur; freeing memory used by other applications may also be necessary.

### Further interruption at generation 240

Generation 239 completed, but generation 240 failed repeatedly even in fresh memory-safe processes. The checkpoint resumes at 240 and retains generations 0–239. At inspection Windows reported approximately 0.34 GiB of free physical RAM and 3.52 GiB of available system commit capacity. These are not measurements from the exact failing instant. Generation 240's raster sizes are in the same range as generation 198, so this is not evidence of a uniquely oversized or invalid course.

Close unneeded applications or restart Windows to release memory before retrying. Do not repeatedly retry under severe system memory pressure. For this recovery the launcher supports a recorded runtime worker override:

```powershell
# Preview only
& .\.venv\Scripts\python.exe run_fast_neat_replicates.py --seeds 5 --resume --memory-safe --evaluation-workers 2

# Resume from generation 240 using two concurrent workers
& .\.venv\Scripts\python.exe run_fast_neat_replicates.py --seeds 5 --start --resume --memory-safe --evaluation-workers 2
```

`--evaluation-workers` requires `--memory-safe`. It changes actual runtime parallelism; the saved training plan and checkpoint still retain their historical eight-worker settings. The actual override is explicitly printed and written into the recovery record, so it must be disclosed when reporting worker counts and elapsed training time. It does not change generation budget, population, curriculum, tracks, random state, scores, or selection rules. Two workers reduce concurrent image allocation and process overhead, but can run more slowly and cannot guarantee success when the machine has insufficient memory. One worker is also supported if needed.

The updated tests compare generation-240 raster pixels and short episode results, verify that an eight-worker request is actually executed with two workers under the override, and check worker replacement, runtime recording, and restoration after use. No study evolution is run by these tests.

## Completed five-seed review and PID comparison

All five fast runs completed 300 generations. The [completed results report](../research/NEAT_Fast_Five_Seed_Results.md) separates original quality-qualified winner selection from a fresh, explicitly exploratory comparison of five validation-selected 1 m/s checkpoints against the saved full-speed PID. Raw data, model hashes, and paper figures are collected under `runs/neat-fast-study/final-review-20261121/`.

```powershell
# Reproduce/reuse formal and follow-up assessments; no evolution or tuning
& .\.venv\Scripts\python.exe runs/neat-fast-study/final-review-20261121/assess.py --workers 2

# Regenerate the report and SVG/PNG paper figures from saved evidence
& .\.venv\Scripts\python.exe runs/neat-fast-study/final-review-20261121/report.py

# Verify model hashes, selection, geometry separation, and completed results
& .\.venv\Scripts\python.exe runs/neat-fast-study/final-review-20261121/verify.py
```

The review does not promote a checkpoint based on test outcomes or substitute an unqualified diagnostic controller for an original formal winner. Its assessment runtime uses two workers and bounded image retention, which is recorded separately from the original eight-worker plan.
