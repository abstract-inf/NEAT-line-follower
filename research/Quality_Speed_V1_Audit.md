# Quality-speed V1 independent audit

Date: 2026-11-24. Scope: scientific correctness and reproducibility of the active development pilot. Protocol SHA-256: `20c6741dbac45b07a85bb935845dbcad1a803202a40e015895f4ce344605d884`.

**No blocking scientific correctness issue found.** This audit does not assess the unfinished pilot as a success or failure. Repeated validation is development selection; the disclosed seed-1 g83 warm start is not five-lineage confirmation.

## Fitness ordering

The incomplete, completed-infeasible and completed-feasible score bands preserve the declared completion/quality/speed ordering (`neat_quality_study/fitness.py:86-95`). Four in-memory checks against the frozen fitness source passed: additional completion dominates, feasible quality beats a faster violation, feasible speed is monotone, and improving already acceptable quality does not earn extra fitness. No evolution or physics was executed.

## Data and cap fairness

All 49 inventoried source files matched both the live and frozen copies; all 158 prepared-file hashes matched. Training defaults to 1 m/s, primary validation explicitly uses 1 m/s, and retention explicitly uses 0.5 m/s (`neat_smooth_study/engine.py:52-55`; `neat_quality_study/training.py:56-64`). The shared PID baseline uses the frozen base-1.0 parameters; its low-cap transfer is disclosed.

At the read-only snapshot of **27 logged generations (0-26)**, 16,200 training episode rows matched generation totals. Recorded caps were exactly 1.0/1.0/0.5 for training/primary/retention. These are logged-generation checks, not an atomic live-process checkpoint snapshot.

## Selection integrity

All 81 saved candidate decisions recomputed exactly from their cached episodes and the frozen reference. The 53 distinct validated genotypes account for 4,240 validation episodes, matching logs at that snapshot. No candidate qualified at that snapshot. Training rank chooses the three distinct candidates; validation does not overwrite their breeding fitness (`neat_quality_study/training.py:147-190`). Completion-group, time, tracking, jitter and required-window gates are unchanged.

## Frozen execution and resume

The launcher dispatches to the verified frozen source, and checkpoint restoration includes population, Python mutation RNG, used geometry, selected-model bytes and selection metadata. Reviewed tests cover ordering, hash rejection, genotype identity and a stubbed checkpoint/resume path; the full test suite was not rerun during this read-only audit.

**P2 — Mitigated in reporting; frozen-engine behavior preserved.** `neat_quality_study/training.py:93-96` removes uncommitted generation logs/decisions, while `training.py:173-184` preserves and reuses their validation-cache files. A cache written before a crash contributes zero to `new_episodes` on resume, so summing generation totals can omit those known physical evaluations, even when their results enter committed selection. Fitness and selection remain correct. The revised report separately sums complete cached episodes for distinct reported genotypes (`research/report_quality_training.py:109-126,236-238`) and explains interruption/in-flight scope in JSON and prose (`report_quality_training.py:254-256`; `docs/neat-quality-speed-study.md:105`). This addresses the reporting risk without changing V1 execution. A future-version resume regression should interrupt after cache creation and before checkpoint commitment.

## Reporting and archival restoration

**Addressed during review:** the original completed archive omitted importable root packages although its instructions invoked the root launcher. Current `research/report_quality_training.py:69-84` materializes every protocol source entry from `prepared/source/` at the archive root and writes the corresponding mapped bytes. This resolves the identified import-path defect without substituting newer live source. Archive creation itself was not run because the pilot is active.

**P3 — Resolved report-input provenance.** The revised helper hashes the three consumed baseline JSONs (`research/report_quality_training.py:214-217`), protocol/training-episode inputs and existing selection/completion files (`report_quality_training.py:225-228`), alongside gates/caches and generation/fitness CSVs (`report_quality_training.py:243-244`). The original omitted-input finding is addressed.

## Reporting-fix recheck

The regenerated report contains 28 logged generations and 55 distinct reported genotypes. Independent read-only counting of their complete cache rows gives **4,400 episodes**, equal to the new cache-backed count and the current generation counter. All required existing report-input hashes are present; all recorded immutable-input hashes match. Mutable CSVs were not required to retain a report-time hash while evolution continues. Both live and frozen execution inventories still match the committed source hashes. No further actionable issue was found in these reporting changes.

The report clearly separates changing training batches from fixed development validation, includes rejection reasons and repeated-genotype caveats, and does not claim fresh-test generalization. No running source, protocol, input, result or process was changed by this audit; only this audit file was added.
