# Simple NEAT pilot implementation audit

25 November 2026. Bounded read-only audit of the running pilot; no physics, training, source changes or prepared-input changes performed.

## Verified implementation

- Frozen protocol SHA-256: `2ff5ecec155d293acf3c52e58802d65ff224157ace8bc0a7b379e242504cd4a5`. Every frozen source and prepared-input digest matched its manifest.
- `run_neat_simple_study.py::compatible_score` overwrites historical fitness with `neat_simple_study/fitness.py::score_batch`; `neat_quality_followup/training.py` assigns that returned number directly to `genome.fitness`. Historical reference limits, eligibility and quality counts remain diagnostics. Recomputing all 100 actual generation-zero scores from stored episode measurements gave **zero discrepancies** with `genome-fitness.csv`.
- All three arms point to the identical `initial/common.pkl.gz`, containing generation zero, 100 genomes and saved Python RNG state. Training reloads it and restores that RNG for each arm. They share frozen batches, physics and stock NEAT configuration; only the declared lambda differs. Manifest verification fixes lambda to 0/1/10, and resume fingerprints include the protocol hash, arm, initial-population hash and condition.
- The 12-episode parent/model-dispatch preflight records matching preserved measurements. This supports the unchanged evaluation path; it is not a general physics validation or hardware result.
- `reporting.py` selects from committed validation candidates using the new objective and ignores historical `best.pkl`. Its text explicitly rules out from-scratch learning and five-seed confirmation. Historical `complete.json::qualified` and `best.pkl` still mean **old-gate qualification**; downstream analysis must preserve that distinction.

Observed snapshot: lambda-0 had four committed generations, 2,400 training episodes and 880 newly evaluated validation episodes; its cache also contained 880 episode rows. Lambda-1/10 had not started. These are an interim snapshot, not final totals.

## Actionable limitations

1. **Practical comparison remains pending.** Frozen `reporting.py` contains no paired PID/prototype comparison. Its successful-time means use differing successful subsets and its all-episode error/jitter means include partial failures; those limitations are disclosed. Root is preparing separate `research/report_simple_pilot.py` with immutable inputs, shared-success paired comparisons, all-arm curves and matching saved controls. Audit that output before claiming practical improvement. A selected model below 0.95 m/s in any required straight still misses the declared speed requirement.
2. **Cost accounting is incomplete.** Committed generation logs omit the 12 preflight episodes and can omit interrupted work. Cached rows and reused-reference slots are not new simulations. Final analysis should separate committed training, unique new validation, preflight, reused slots and any unaccounted interrupted work; do not call their partial sum total physical compute.
3. **Live reports are not atomic snapshots.** `reporting.py` loads a checkpoint/log, then hashes its path separately while training may replace/append it. The hash can describe newer bytes than the interpreted snapshot. Generate final analysis from immutable copied inputs after completion, as planned, without changing this running protocol.

No blocking breeding-fitness defect was found in this bounded audit. Independent fresh-population confirmation and untouched-test evidence remain outstanding.
