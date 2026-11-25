# Quality-speed confirmation design

Planning note, **24 November 2026**. QUALITY-SPEED-V1 remains an ongoing frozen development experiment; no outcome is assumed here. This note launches no experiment and changes no source, thresholds, data, or controller selection.

## Recommendation and scope

If development establishes a useful joint speed/quality controller from smooth g83, the closest confirmation is a **uniform smooth-then-speed pipeline**, applied separately to the five existing native 1 m/s parents. Use **100 smooth generations followed by 50 speed generations**, with one frozen reward and one deterministic selection rule per stage for every lineage. These budgets reproduce the development opportunity that produced the g83 starting point; they are a proposed confirmation budget, not evidence that 150 generations will suffice.

Directly training the new fitness for 50 generations from all five fast parents is cheaper and scientifically reasonable, but tests a different initialization regime. Development from a selected smooth ancestor does not validate direct training from a fast ancestor. If budget permits only direct training, describe it as a new transfer/repeatability experiment and preserve failures; do not call it confirmation of the demonstrated two-stage path. The strict objective's possible barrier outside quality feasibility makes the two-stage proposal the more realistic first confirmation of a g83-based result. This is a hypothesis, not a prediction of success.

## Parent identity and independence

Freeze the existing parents without choosing new ones from confirmation outcomes:

| Ancestral seed | Native 1 m/s generation | Prepared input | SHA-256 prefix |
|---|---:|---|---|
| 1 | 134 | `seed-01.pkl` | `6114a6859544` |
| 2 | 136 | `seed-02.pkl` | `aaefcd0c5fda` |
| 3 | 131 | `seed-03.pkl` | `441cc0f96ba3` |
| 4 | 107 | `seed-04.pkl` | `5e98a3b88a58` |
| 5 | 134 | `seed-05.pkl` | `40f4da6b38f9` |

Inputs are under `runs/neat-smooth-study-v1/prepared/inputs/`; commit full hashes in the new protocol. Their common historical selection rule was maximum validation completion, then validation fitness, then earliest generation within native-1 m/s generations 100–149. They are validation-selected descendants of distinct earlier training roots, not fresh random initial networks. [Existing five-seed results and parent-selection record](NEAT_Fast_Five_Seed_Results.md)

Five random restarts of g83 would estimate continuation variability **conditional on one selected ancestor**; they are not five independently derived ancestral lineages. Conversely, the five parents above provide distinct ancestry, but seed 1 informed method development. Report its new continuation as a **development-ancestry replication**, separately from seeds 2–5, whose ancestry was not used to produce g83. Show all five rows, but do not describe them as five untouched confirmation roots. Five entirely held-out roots would require another independently generated native parent under the already frozen historical procedure; that additional work is not proposed for launch here.

## Minimal consistent pipeline

1. **Freeze the whole procedure before any confirmation run.** Commit source/dependency/input hashes; global stage rewards; 100/50 generation ceilings; population target 100; six courses per genome; stage-specific initialization/seeds; validation budgets; failure stops; and final ranking. Preserve stock NEAT-Python 0.92 mutation, crossover, reproduction, speciation, stagnation, input/action interface, physics, and recovery. No lineage-specific reward, coefficient, budget, warm-start substitution, or restart on failure.
2. **Stage 1: smooth, 100 generations.** Start each lineage from one exact native parent plus 99 independently stock-mutated copies, as in the smooth development workflow. Use the same frozen smooth reward for all lineages, not whichever old/smooth branch retrospectively looks best. Validate the training champion once per generation on a new stage-1 suite of 60 primary and 20 retention courses. Use the lineage's original native parent as the quality reference throughout.
3. **Select the stage-2 parent prospectively.** Stage-1 eligibility protects parent completion by group at both caps, shared-success primary geometric RMSE/jitter, and every observed primary straight yaw/steering limit. Speed and paired-lap-time requirements belong to the final milestone and do not apply to this preparation-stage eligibility. Among eligible stage-1 champions, maximize primary completion, then minimize the primary paired-time ratio to the native parent, then jitter, geometric RMSE, and generation for exact ties. Require nonempty matched denominators. Freeze this complete rule in advance; never choose “the g83 equivalent” by inspection. If none is eligible, record stage-1 pipeline failure and do not silently substitute the native parent or a near miss.
4. **Stage 2: speed, 50 generations.** Initialize one exact stage-1 selection plus 99 stock-mutated copies, using a new predetermined stage seed. Apply the single speed objective settled in development to every lineage. Measure same-batch quality against each lineage's original native parent, keeping current reference floors and quality limits fixed. Validate the top three distinct training genotypes per generation, with the established deterministic cache, on a separate new 60-primary/20-retention stage-2 suite. Only its original full acceptance gates can select an official final winner.

This proposal intentionally separates stage-1 eligibility from final acceptance. It does not relax the final **every-straight 0.95 m/s**, yaw/steering, tracking/jitter, grouped completion, retention, or **1.05 paired-time** requirements. Final ranking remains the existing completion/time-band/quality rule. [Current development protocol](../docs/neat-quality-speed-study.md); [smooth workflow and acceptance rules](../docs/neat-smooth-study.md)

Nominal training is approximately **90,000 candidate episodes per full lineage**, or 450,000 across five, plus same-batch references and validation. The proposed validation ceilings are 8,000 stage-1 and 12,000 stage-2 episode slots per lineage. Actual populations, cached repeats, reference evaluations, interruptions, and early stage-1 failure stops determine committed counts. Record these separately; do not call the nominal ceiling exact compute use. No unsuccessful lineage receives additional generations.

## Fresh splits and the PID comparison

Freeze new confirmation training-bank, stage-1 validation, and stage-2 validation seeds before execution, excluding prior protected geometries and mirrored equivalents. Use common stage/generation course batches across lineages to hold tasks constant, with independently derived evolution seeds for each lineage and stage. Both validation suites are repeatedly consulted selection data, not test evidence. The distinction matters because selection can overfit finite evaluation data. [Cawley and Talbot, 2010, JMLR](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)

Commit a separate final-test seed and course generator/version before confirmation, but do not generate or inspect those courses until all outcomes, winner hashes, and PID configurations are frozen. A practical common test retains the existing 150-course composition: 90 mixed, 30 speed/recovery, and 30 corner drills. Assess primary 1 m/s and retention 0.5 m/s under the same robot and termination rules. Evaluate the original five native parents and every official new winner; reuse the frozen full-speed filtered PID at 1 m/s and a prospectively frozen cap-specific PID at 0.5 m/s. A missing qualified low-speed tune must be reported, not replaced by an unlabeled gain transfer. Count each deterministic PID evaluation once even though it supports multiple paired comparisons. Neither controller family is tuned against this final test.

## Failure reporting and permissible claims

Report stage-1 eligibility and stage-2 final qualification as `k/5` attempted ancestral pipelines, with every seed's failure reason, budget consumed, candidate history, and ancestry label. Also report seeds 2–5 separately from the development ancestry. A missing official winner is a pipeline outcome, not a missing observation to discard. If only `k` winners reach testing, their performance summary is explicitly conditional on those `k` successes; do not present it as average performance of five successful training runs. Any diagnostic fallback controller must have its own prospective rule and label before test access, and cannot become an official winner.

Show per-lineage completion, minimum required-window speed, worst-window yaw/steering, independent geometric RMSE, command jitter, and paired successful-lap times with denominators. Complement successful-lap timing with a fixed timeout-penalized time metric. Show spread/uncertainty at the lineage level and course-paired differences; hundreds of deterministic course episodes are not hundreds of independent training runs. With only four held-out ancestries, uncertainty remains substantial. General few-run evaluation cautions are supported by [Agarwal et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).

A useful V1 outcome would demonstrate that the selected objective can improve one disclosed smooth ancestor under its development conditions. It would not establish that smoothing is reproducible in other ancestries, that direct native-parent training works, that the two-stage pipeline succeeds reliably, that a new test passes, or that NEAT is generally superior to filtered PID. Those are separate questions for the frozen follow-up.

Verification for this note was read-only: existing methodology/results and the five prepared input hashes were inspected. No new checkpoints were selected or evaluated, and no experiment was launched.
