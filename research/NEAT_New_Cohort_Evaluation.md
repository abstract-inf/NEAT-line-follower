# Frozen NEAT evaluation on the new PID test cohort

Evaluation and documentation: 2 November 2026. This is fixed-policy inference,
not a new NEAT training phase. The already validation-selected generation-109
model was specified before this assessment; no other genome was tried or
substituted after seeing the results.

## Exact model and protocol

Model: runs/28-10-2026-hard/best.pkl, also preserved as frozen-model.pkl in
runs/neat-frozen-new90-20261102. SHA256:
**4cf2d8af1bc4b7759f4ede52b00ff12b78c8ecec316fa7aaea623d8662540dcd**. Generation 109, evolution seed 0. The feedforward
15-binary-input/two-output controller, topology, weights, activations, motor
mapping and sensing are unchanged. No output rescaling, memory, recurrence,
training, controller tuning or model reselection occurred.

The new cohort is exactly runs/pid-gains-20261029/suite/test: 90 tracks, 30 per
difficulty. Manifest SHA256: **16399ef0aa1d6d07ac5779b2126cd597cd3a006e1b943c4c8cce9d89a0922ef7**. Every policy row uses the
same individual geometry hashes. Robot and episode settings are copied from the
PID run and verified equal to the frozen model's saved metadata: 0.35 m/s wheel
target ceiling, 0.03 N m torque limits, 50 ms actuator filter, unchanged geometry,
600/60 Hz physics/control, and the same length-dependent 45–120 s episode limits,
gates, corridor and failure rules. CPU MuJoCo is the common assessment backend;
historical GPU validation is separate. The model's training-suite hash differs
from this new evaluation suite by design.

Four persistent CPU workers evaluated one episode per test route. It took
20.055 s including setup, geometry verification, source
snapshot and integrity checks. No real-time pacing was used. Every attempted
episode is saved, including failures. Predetermined indices 0, 1, 2, 31 and 62
have full sensor/action/physics telemetry; traces are from those same episodes,
not additional independent trials. The original PID rows are reused unchanged
from their completed frozen assessment, rather than reselected or retrained.

## Common new-cohort results

| Frozen policy | Completed / 90 | Rate | RMSE (mm) | MAE (mm) | Mean progress |
|---|---:|---:|---:|---:|---:|
| Filtered PID | 51/90 | 56.67% | 30.108 | 17.712 | 0.7602 |
| PID + corner supervision, original gains | 78/90 | 86.67% | 8.606 | 5.593 | 0.9218 |
| PID + corner supervision, tuned gains | 84/90 | 93.33% | 6.654 | 4.479 | 0.9670 |
| Frozen NEAT g109 | 38/90 | 42.22% | 23.194 | 15.308 | 0.6814 |


| Frozen policy | Beginner | Intermediate | Advanced |
|---|---:|---:|---:|
| Filtered PID | 30/30 | 11/30 | 10/30 |
| PID + corner supervision, original gains | 30/30 | 29/30 | 19/30 |
| PID + corner supervision, tuned gains | 30/30 | 29/30 | 25/30 |
| Frozen NEAT g109 | 30/30 | 5/30 | 3/30 |


![Matched new-cohort assessment](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/neat-new90/common-new90.png>)

RMSE and MAE are macro-averages of per-episode geometric lateral errors over
observed duration, including failures. MAE is measured directly. Early failure
can reduce observed exposure, so a lower error alone does not establish a better
controller. Completion/progress and failures must be read alongside error.

Frozen NEAT failed 52 routes, with reasons
`{"off_route": 52}`. On the 32
routes tagged as containing zigzags, it completed 0.
Membership does not establish that a failure occurred at the zigzag; some
episodes can terminate before reaching that feature.

## Paired comparison and timing

The paired-comparison.csv artifact counts both-complete, PID-only, NEAT-only
and neither-complete outcomes on identical track hashes. It also reports time
only on the common-success subset for each PID/NEAT pair. Overall successful
time averages refer to different subsets and are not a fair direct speed ranking.
No p-value, statistical significance or algorithm-family superiority is claimed.

## Interpretation and relation to earlier results

The original common cohort remains a separate 90-track benchmark: the frozen
NEAT policy completed 35/90 there. Its 38/90 result here comes from
different routes, not training improvement. The new cohort has no geometry-hash
overlap with the original cohort. All policies have now been assessed on the new
cohort, resolving the previous missing NEAT cell for the gain-tuning comparison.

This is a descriptive comparison of these fixed implementations. The assisted
PID has explicit corner logic and temporal state. The selected NEAT model is
memoryless and its saved sigmoid outputs restrict reverse manoeuvres. Historical
training protocols and compute budgets differ, and only one NEAT lineage exists.
These results do not show that NEAT in general cannot match PID. They concern
new routes from this generator, not faults, disturbances or hardware transfer.

The new tracks were unseen by this frozen NEAT policy, but their PID outcomes
were already known when this assessment was requested. The NEAT artifact was
fixed before either test; it was not selected on these outcomes. Future tuning
must treat both evaluated cohorts as known and use a newly prespecified untouched
cohort for a fresh final claim.

## Reproducibility and preservation

Raw NEAT rows: runs/neat-frozen-new90-20261102/test-episodes.csv. Original PID
rows: runs/pid-gains-20261029/test-episodes.csv. Evaluation records include the
model copy, run.json, status.json, source.zip, tracks/test, telemetry/ and
preserved-records.json. All 760 preserved historical
files and the original model remain byte-identical. Shared package source SHA256:
**78acab6866de9a70228f5c8e6b3e84a74592c0d00290927a0c87ccdc5909da8d**. The run records Python, dependencies,
hardware, worker count, script checksum and exact metadata.

Run `.venv\Scripts\python.exe evaluate_frozen_neat.py --output runs/neat-new90-replay`
for a fixed-policy reproduction in a new directory; it is not independent
training evidence. Regenerate saved-data tables/plots with
`.venv\Scripts\python.exe research/analyze_neat_new90.py`, after the earlier
report generators. Combined 360-row analysis data, summary tables, failures and
paired comparisons are under research/artifacts/neat-new90. Raw inputs are not
overwritten. No manuscript rewrite is started by these research records.
