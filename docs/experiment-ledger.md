# Reproducible experiment ledger

This ledger records the autonomous development work requested after the old/smooth pilot comparison. Every new run must retain its protocol before evaluation or evolution, source/input hashes, dependencies, seed, exact commands, raw outcomes, model selection rule and interpretation. Unsuccessful experiments remain in the record. Development/validation data may guide the next protocol; final test data must not.

## Preserved predecessors

| ID | Protocol / evidence | Status and interpretation |
|---|---|---|
| SMOOTH-V1-AUDIT | `runs/neat-smooth-study-v1/prepared/protocol.json`; `results/neat-smooth-study-v1/audit-report.md` | 14 archived candidates audited. None met the combined speed/quality target. |
| SMOOTH-V1-OLD | `runs/neat-smooth-study-v1/pilot/seed-01/old/`; `results/neat-smooth-study-v1/pilot-old-100-generations/report.md` | 100 generations, 60,000 training + 8,000 validation episodes. No qualifier. |
| SMOOTH-V1-SMOOTH | `runs/neat-smooth-study-v1/pilot/seed-01/smooth/`; `results/neat-smooth-study-v1/pilot-comparison/report.md` | 100 generations, 60,012 training + 8,000 validation episodes. 84 champions passed straight yaw/steering, but none met the whole target. |

The original fixed inputs, outputs and frozen execution source are preserved. No retrospective edits to reward, selection thresholds or recorded outcomes are permitted.

Earlier experiments retain their original detailed records rather than being reconstructed from memory:

| Experiment family | Reproduction / interpretation record | Saved data area |
|---|---|---|
| Low-speed and short-ramp five-seed study | [Methodology and terminal commands](methodology.md) | `runs/neat-speed-study/` |
| 300-generation fast curriculum and five-lineage repetitions | [Fast-study protocol](fast-neat-study.md), [repeat and memory-recovery record](fast-neat-repeats.md) | `runs/neat-fast-study/` |
| Sensor-position geometry pilots | [Geometry protocol](geometry-pilot-protocol.md), [geometry results](geometry-pilot-results.md) | `runs/geometry-pilot-v1/`, `runs/geometry-pilot-v2/` |
| Older implementation, dynamics and evaluation changes | [Research implementation record](research-record.md) | The dated run directories referenced there |
| Long exploratory competition courses | [Course choices, geometry and commands](long-course-candidates.md) | `runs/long-course-candidates-20261121/`, `runs/rhythm-changes-case-study-20261121/` |
| Smooth versus original reward, full-speed PID protocol | [Smooth-study workflow and results](neat-smooth-study.md) | `runs/neat-smooth-study-v1/`, `results/neat-smooth-study-v1/` |

These links are an index of preserved historical records, not a claim that old protocols all used the current fitness, speed law, termination logic or reproduction guarantees. Use each run's frozen source and inputs for repetition. Replaying an old genome with today's source is a new evaluation condition and must be identified as such.

## Goal-development experiments

### PID-1M-VALID-001 — validate the established full-cruise PID

Question: does the saved base-1.0 filtered PID satisfy the frozen fresh PID validation rule? Reuse the already prepared tuning protocol and suite, not NEAT's final test tracks.

```powershell
& .\.venv\Scripts\python.exe run_neat_smooth_study.py tune-pid --cap 1.0 --workers 4
```

This existing frozen command first evaluates the established tune; if it is eligible, it retains it. If not, it executes the predeclared, deterministic PID search. Completion, every isolated-straight speed window and observed quality metrics are retained. Output: `runs/neat-smooth-study-v1/pid/cap-100/`. No new PID family or privileged controller input is introduced.

Outcome: the existing full-speed tune passed 60/60 PID validation courses, with minimum dedicated-straight speed 0.9905 m/s. It was retained without retuning. The frozen config SHA-256 is `c6c68b53342600db0078e991bbb729064d829d360c4840a720dbf5e66d94c24b`. See `runs/neat-smooth-study-v1/pid/cap-100/selection-freeze.json`. This PID-validation suite is distinct from the shared development comparison below.

### QUALITY-SPEED-V1 — calibrate and test constraint-aware task fitness

Development question: can a stock NEAT controller preserve acceptable tracking/smoothness while raising sustained straight speed to at least 0.95 m/s under the 1 m/s cap? The preceding smooth g83 is a disclosed development warm start, not a fresh independent winner.

Before evolution: document the exact scalar objective, plateau below quality limits, failure ordering, reference measurements, seeds, course splits and pilot budget. Rescore stored episodes and test synthetic counterexamples; do not choose weights against final test outcomes. Keep stock NEAT mutation/reproduction and existing controller inputs/action mapping. All physical route/error information belongs exclusively to the evaluator/fitness, not the neural inputs.

The source, inputs and new execution tree will be frozen in a distinct run directory. Existing v1 control results remain preserved. A 50-generation development pilot is the initial intended budget; all actual population and episode counts must be reported. Five-lineage confirmation and a fresh PID comparison follow only after the training/selection protocol is settled. If the pilot fails, record it and version the next experiment explicitly.

Implementation and calibration completed: 55 regression checks passed, including nine new objective/checkpoint checks. The full equations, commands, assumptions, data roles and interruption rules are in [the quality-speed methodology](neat-quality-speed-study.md). The 200 stored development champions were rescored before preparation; this is retrospective calibration, not new test evidence.

The protocol and execution source are frozen in `runs/neat-quality-speed-v1/prepared/`; `launch-record.json` records the authorized 50-generation, six-worker command and prelaunch checks. The live stage is `development/seed-01/`. Its checkpoints/logs, rather than this static ledger, determine current progress. No automatic extension or post-hoc acceptance relaxation is allowed.

Completed outcome, recorded 25 November 2026: all 50 generations committed with 30,000 candidate training episodes, 300 reference episodes and 7,840 complete cache-backed validation episodes (98 distinct genotypes, 150 candidate records). No candidate passed every original gate; all 150 missed required straight speed. The fastest candidate rejected only for speed was g40 rank 0, genotype `58815ae4f4efeab17dac70046f418488785e6199326de2423d0c28d772bf8aa9`: 60/60 primary and 20/20 retention completions, minimum straight speed 0.8841878 m/s, worst yaw RMS 0.0837191 rad/s, steering RMS 0.0114571, paired time ratio 1.0102969, shared-finish geometric RMSE 14.1514 mm and mean command jitter 0.0062963. Compared with the original g83 warm start, straight speed improved from 0.7135 m/s, with a disclosed increase in geometric RMSE. It is not promoted to an official winner. Full records and PNG/SVG figures are in `results/neat-quality-speed-v1/training/`.

### QUALITY-SPEED-V1-BASELINE — shared development comparison

Command: `& .\.venv\Scripts\python.exe run_neat_quality_study.py baseline --workers 6`.

Controllers were frozen before 240 deterministic evaluations (60 primary plus 20 retention for each of stable NEAT g83, fast NEAT g134 and the saved full-speed PID). Primary completion was 60/60, 60/60 and 59/60, respectively. The PID's one failure was off-route. This is a previously consulted development suite, not fresh final testing; it does not contradict PID's 60/60 on its separate tuning-validation suite. Raw episodes, matched finish-time denominators and frozen hashes are under `runs/neat-quality-speed-v1/baseline-development/`. The auxiliary PID 0.5 m/s run transfers its 1 m/s gains and is not called a cap-specific tune.

### PID-050-FULL-001 — cap-specific low-speed comparison baseline

Command: `& .\.venv\Scripts\python.exe run_neat_smooth_study.py tune-pid --cap 0.5 --workers 1`.

Use the existing frozen tuning protocol, development/validation splits, deterministic candidate grid and refinement schedule. All candidates have `base=1.0`; this replaces the inappropriate restricted-cruise historical tune for any later cap-specific 0.5 m/s comparison. One worker is used concurrently with the six-worker NEAT development run to limit memory pressure. Candidate results, seed, bounds, failures and final eligibility remain under `runs/neat-smooth-study-v1/pid/cap-050/`. This is baseline development, not final-test tuning. Inspect its final selection/failed record before claiming a validated tune.

Completed outcome: `refine2-169` was selected after 130 coarse candidates, 12 expanded candidates, two sets of 24 refinements and 10 validation finalists. It completed 60/60 validation tracks, with minimum dedicated-straight speed 0.4952560 m/s. Saved config `pid-fullspeed.json` SHA-256 is `7777c1c55f24102672c4689135cc6cb9d4cf95f2a63ad9fd32666f22b68c6295`. This is a validated historical baseline; the user's subsequent request to address the Rhythm Changes case motivates a separate further PID tuning version, preserving this result.

### QUALITY-NETWORK-001 — offline architecture diagnostics

Command: `& .\.venv\Scripts\python.exe research/analyze_quality_networks.py`.

Inspect both frozen predecessor networks and all candidates from logged development generations. Record node/connection counts, graph reachability, whether stock NEAT-Python actually evaluates each output, and deterministic constant-sensor probes. Each probe resets history, holds one sensor active for 180 steps at 60 Hz, then records wheel commands. No physics or evolution is executed. These commands are not measured robot speeds and are not used to select the experiment winner. Output and input-model/reporting-source hashes: `results/neat-quality-speed-v1/training/network-diagnostics.json`. Regenerate after evolution for complete coverage. This preserves the existing 0.92 handling of disconnected nodes.

The same diagnostics command records the 25 installed NEAT Python source hashes and checks them against the distribution's installation `RECORD`; all 25 matched for version 0.92. Evidence: `results/neat-quality-speed-v1/neat-runtime-source-integrity.json`. This supplements package version pinning and does not claim independent authentication of the installation manifest. A separate read-only audit of frozen fitness, source/input hashes, validation decisions, actual caps and counts is in `research/Quality_Speed_V1_Audit.md`. The preparation ZIP was extracted into a clean separate directory and its nested frozen `verify` command passed; the original ZIP was preserved.

### QUALITY-OBJECTIVE-DIAGNOSTIC-001 — offline ranking counterfactual

Command: `& .\.venv\Scripts\python.exe research/diagnose_quality_objective.py`.

Conditional proposal only: preserve the incomplete V1 score and rescore completed batches with `10*m + 9*S/(1+V)`, using exactly the already measured speed utility and quality violation. This removes the feasibility-band jump without changing thresholds or stock NEAT operators. It has not been used for evolution and does not demonstrate a better controller. The script snapshots and hashes the exact captured inputs and its own source, filters to logged generations and supports exact snapshot replay. Saved genome rankings cannot reveal the validation outcomes of unsaved alternate candidates or predict a new evolutionary trajectory. The initial interim snapshot spans generations 0–38 and is preserved under `results/neat-quality-speed-v1/objective-diagnostic/snapshots/`; refresh after completion for the complete diagnostic. Its seven synthetic ordering checks and snapshot replay passed. No V2 experiment has been registered or launched by this diagnostic.

### QUALITY-G21-PID-DEVELOPMENT-001 — compare the saved near qualifier

The training report's named fastest candidate rejected only for straight speed is generation 21, rank 1, genotype `369c307f11d669f6a8c7913c2d8f83b163466b4a34d098207c5e01200d3803d0`. Compare its saved episodes with the frozen full-speed PID and original fast g134 on the same 60 primary development geometries/groups at 1 m/s. No new simulation, training or controller selection occurs. The candidate remains unofficial: minimum required-window speed is 0.87214 m/s, below 0.95.

Outcome: g21/PID completion 60/60 versus 59/60. On 59 shared finishes, mean completion time is 6.0921 versus 7.1226 seconds, mean per-course geometric RMSE 13.988 versus 20.565 mm, and command jitter 0.0061896 versus 0.0011714. PID has substantially lower jitter. Worst straight yaw RMS is 0.0984 versus 0.0744 rad/s; the original fast NEAT reference reaches 1.3956. These reused selection data support no fresh-test or algorithm-wide superiority claim.

Evidence, fixed controller/input/source hashes, numerical results, paired PNG/SVG plots and exact replay commands are in `results/neat-quality-speed-v1/candidate-pid-development/report.md`. Immutable snapshot `cfc2659d90bb4df9be692246831547704ab0933153c0cd064d515d1fb61e4467` retains the consumed episodes, model/gain bytes, protocol/suite, training report and reporting script. Snapshot replay reproduced identical numerical JSON. The separate pre-evolution baseline report is preserved unchanged.

### QUALITY-G40-PID-DEVELOPMENT-001 — completed-budget near qualifier

After the full 50-generation budget, apply the same disclosed near-qualifier rule. G40 rank 0 reaches 0.8841878 m/s and remains unqualified. Completion is g40/PID/fast g134 = 60/59/60. On the 59 shared g40/PID successes, mean times are 6.053672 / 7.122599 s, mean per-course geometric RMSE 14.197061 / 20.565365 mm and jitter 0.00627033 / 0.001171425. PID retains substantially lower jitter. Snapshot `b070af2d238cadb8b479532d5a59e65725341f38dff34127a5f7976ab303f99a` preserves the exact inputs and generalized reporting source under `results/neat-quality-speed-v1/candidate-g40-pid-development/`. Numerical replay matched exactly; the earlier g21 snapshot/output is preserved. These are development comparisons, not new test episodes or independent training repetitions.

### QUALITY-FOLLOWUP-V1 — continuous discount and initialization comparison

Registered after the completed V1 outcome, recorded 25 November 2026. Two 50-generation, population-100 development arms use the exact same six-course batches, references, evolution seed, validation suite and unchanged final gates. `smooth` starts from g83; `fast` starts from native g134. Complete-batch reward is `10*m + 9*S/(1+V)`; incomplete reward and all measurements are unchanged. This allows speed to affect ranking during small quality violations without weakening final acceptance. Stock NEAT operators, sensor/action interface and physical source remain unchanged.

Protocol/source/input/initial-population snapshots are under `runs/neat-quality-followup-v1/prepared/`. The three saved historical V1 generation-zero genotypes match the reconstructed smooth initial population; full historical identity of all 100 cannot be directly proven because V1 overwrote its initial checkpoint. Twenty-two focused implementation checks passed before preparation, including stubbed resume/dispatch. The two arms each reuse 300 saved reference episodes and run zero new reference physics. This is development, not independent five-lineage confirmation or a fresh test; only the stated reward/initialization contrasts are supported. Workflow and commands: [follow-up methodology](neat-quality-followup.md). Actual launches, worker counts, raw episodes, caches and interruptions remain in each arm's runtime/checkpoint logs.

Before authorizing evolution, a real 12-episode replay matched exactly: smooth initial parent on six copied generation-zero courses against original V1 rows; native-fast initial parent on those six against the saved reference. Completion/termination, control steps, timing, progress, independent error, jitter and measured speed matched. Expected bytes, raw results, source hashes and replay helper are under `runs/neat-quality-followup-v1/preflight/`. Both original and new frozen verifiers passed. Evolution was then authorized at three workers per arm; PID tuning uses one concurrent worker. Local scripts handle the long computations with durable logs to minimize LLM polling.

Completed outcome: both 50-generation arms exited successfully; neither selected an official winner. Smooth used 30,000 training episodes and 8,800 new validation episodes (3,200 repeated validation slots reused cache); fast used 30,036 training episodes and 8,320 new validation episodes (3,680 cached slots). Each reused 300 reference episodes. The changed reward did not solve the combined target. Smooth's fastest quality-passing speed-only near miss was g34 rank 2 at 0.8661194 m/s; fast's was g44 rank 0 at 0.8306575 m/s. Seventeen fast-arm candidate records reached at least 0.95 m/s in every required window, but failed other gates; records include repeated genotypes and are not independent runs.

Fast g33 rank 2 completed 60/60 primary and 20/20 retention courses at minimum straight speed 0.9726349 m/s and paired-time ratio 1.0151914, but its geometric RMSE 20.2511 mm, jitter 0.0085931, worst yaw RMS 0.1932455 rad/s and steering RMS 0.0213009 failed the unchanged quality checks. It is a disclosed development candidate, not an official winner. Exact execution/count/integrity records are in `runs/neat-quality-followup-v1/launch-state/{smooth,fast}/terminal.json`.

### NEAT-JOINT-V1 — one focused joint-constraint fine-tuning attempt

After the preceding two completed unsuccessful arms, register one 50-generation run from fast g33 rank 2. The deterministic parent rule requires full primary/retention completion, paired time at most 1.05, and every required straight at least 0.95 m/s, then minimizes maximum relative quality excess. Selection audit, parent/cache bytes and native reference are frozen before evolution.

For completed batches, optimize the maximum of the original five normalized quality violations and `max(0,(0.95-minimum_straight_speed)/0.05)`. Positive joint violation has graded score `10*m+9/(1+violation)`; joint-feasible score is `10*m+10+9*S`. Incomplete fitness is exactly V1. The 0.05 m/s scale is the 1 m/s cap minus its existing 0.95 target. A slow quality-only policy is no longer joint-feasible. No validation threshold changes.

Only standard configuration powers are reduced for local refinement: bias mutation power 0.12→0.03 and weight mutation power 0.14→0.035. Rates, topology settings, operator implementations, policy inputs/output mapping and physics are unchanged. The initialization template has the original genome and exactly matching 100-step deterministic actions; restoring those two config values reproduces the original serialized configuration. This combined parent/reward/fine-tuning change does not isolate individual causal effects and is not an independent lineage.

Twenty focused checks passed. Preparation froze 57 runtime source files, 100 initial genomes/RNG, the same 50 six-course batches, 300 reused/zero new reference slots and consulted 60+20 validation. Protocol SHA-256: `b93f49b00272bf1822f53e835ab17661698fa7e33d701f9c2ef216a7ef83d0b8`. Evolution was authorized only after tests and verification, at six workers alongside one PID worker. Workflow: [joint study methodology](neat-joint-study.md). Source, actual launch, raw results and resumable population belong under `runs/neat-joint-study-v1/`; there is no automatic budget extension or extra arm. Fresh confirmation/test comparison remains pending.

Completed outcome: 50 generations, 30,030 training episodes, 10,800 new validation episodes plus 1,200 cached slots, and 300 reused references. All integrity checks passed, but no official qualifier emerged. Twenty-three training evaluations were joint-feasible; validation geometry failed all 150 candidate records and jitter failed 141. The strongest full-completion, speed/time/straight-quality-passing prototype was g49 rank 1 at 0.9501813 m/s, with geometric RMSE 16.4621 mm and jitter 0.00752684. Both still exceeded the protected reference. The saved profile diagnostic demonstrates how mixed-course deficits can be masked by extra training weight on corner gains. No proposed weighting/profile correction was implemented after the user's finalization request.

### PID-QUALITY-TUNING-V1 — address the known long-course gain issues

The user explicitly requested additional PID tuning after observing small gain issues on Rhythm Changes. That previously tested 45.7739 m course (hash `77fb2a4f7caad1dd33e3594335e1639faf2d03f339e25694b4dc3744c9205d6a`) is now disclosed development/tuning data. A later fresh shared test will compare frozen controllers; this old case cannot be called untouched final-test evidence.

Freeze a new deterministic search at caps 1.0, 1.5 and 0.5 m/s in that order, keeping base=1.0, existing physics and recovery. Search the five existing gain fields using 130 coarse candidates (including saved controls and 3% small probes), 12 expanded candidates, two sets of 24 local refinements, and up to 10 selection finalists. Use the actual existing 50 development and 60 selection tracks plus the long case; maximum planned records are 5,360 per cap, 16,080 total. Do not skip the search when a saved tune passes. Eligibility retains isolated-straight speed, completion and long-case success; ranking and all failures are recorded. No global optimum is claimed.

An initial preparation expected 60 development tracks; inspection found the existing bank has 50. No simulation occurred. The zero-evaluation partial preparation is preserved under `runs/pid-quality-tuning-v1/preparation-attempts/initial-development-count-correction/`; counts were corrected before committing the protocol. Eleven focused checks and frozen-source verification passed. Root additionally checked all three real candidate sets: 130 each and base=1.0 throughout. See [PID tuning workflow](pid-quality-tuning.md), `runs/pid-quality-tuning-v1/prepared/protocol.json`, per-cap candidate records, selection and runtime logs. Start command: `& .\.venv\Scripts\python.exe run_pid_quality_tuning.py tune --workers 1`.

Completed cap-1.0 outcome: all 5,360 planned episode records were saved; `refine2-169` completed 60/60 selection tracks plus Rhythm Changes. Minimum isolated-straight speed was 0.9905121 m/s. Frozen gains SHA-256 `d6df8d4275b568583dcaf652a131f7c3f17f4d6923998828eb1322562c78b5cc` are under `runs/pid-quality-tuning-v1/cap-100/pid-fullspeed.json`. The 1.5 m/s search also completed 5,360 records but found no candidate meeting the 95% multi-track completion gate. Its best observed `refine2-154` finished Rhythm Changes and reached isolated speed 1.4857682 m/s, yet completed only 29/60 selection tracks (plus the long case), with 28 off-route and three stall failures. No official 1.5 m/s tune was frozen. The 0.5 m/s search remains separate; its authoritative completion/selection files determine outcome. These failures are retained rather than replacing the gate or presenting a poor multi-track controller as validated.

Final PID outcome: all three searches exited normally, 16,080 total episode records and zero evaluation errors. At 0.5 m/s `refine2-160` passed 60/60 selection courses plus Rhythm Changes, minimum isolated speed 0.4952560 m/s; config hash `dabef23380b282c736540529daaca1951ba9ae8f7f4ea5f0ef002c80e80048ca`. All tuning trade-offs, exact gains and paired denominators are in `runs/pid-quality-tuning-v1/completion-summary.json` and [the tuning document](pid-quality-tuning.md). The 1.5 m/s search remains explicitly unqualified.

### FINALIZATION-FRESH-001 — frozen prototype and tuned PID on untouched shared tracks

At the user's request to finalize, stop training and freeze the validation-ineligible g49/rank1 prototype plus the validated cap-specific PID configurations before generating any fresh geometry. Evaluate 150 shared courses (90 mixed, 30 speed, 30 corner) at both 1.0 and 0.5 m/s: exactly 600 episodes. Verification checked all controller/source hashes, caps, paired geometry identities, zero evaluation errors, and no overlap with 17,108 prior protected geometries. There was no test-based controller selection or retraining. This is an exploratory assessment of one ancestry, not five-lineage confirmation or promotion to an official winner.

At 1.0 m/s NEAT/PID completion was 148/150 versus 141/150; on 140 shared finishes, mean times were 5.7292/6.7105 s, mean per-course geometric RMSE 18.8970/23.9881 mm, and jitter 0.00881205/0.00116554. At 0.5 m/s completion was 145/150 versus 148/150; 144 shared finishes had mean times 9.4734/9.3052 s. All additional speed, progress, failure and quality values remain in the numerical report. Conclusions retain PID's jitter advantage and NEAT's original failed validation gates.

Closeout: [final results and limitations](final-results-20261125.md). Numerical report, PNG/SVG plots and frozen handoff models are under `results/neat-pid-finalization-20261125/`; raw shared tracks/episodes are under `runs/neat-pid-finalization-20261125/`. Reproduce using `research/finalize_neat_pid.py` and `research/write_final_closeout.py`. No subsequent experiment was started. A future resumed study must use development data and a new untouched test suite.

### RHYTHM-CURRENT-PAIR-VIDEO-001 — two explicitly requested showcase videos

At the user's later request, record exactly two fixed-controller videos on the known 45.7739 m Rhythm Changes course, both capped at 1.00 m/s: final prototype joint g49 rank1 (model hash `e1be2276a7acc26f0b1eaaa1d716602862cb6d99721dbf664e048be7cd8be78c`) and retuned PID (`d6df8d4275b568583dcaf652a131f7c3f17f4d6923998828eb1322562c78b5cc`, base1). Both completed: NEAT 59.9833 s, PID 68.3333 s. Each separately captured replay matched its saved headless termination, steps, time, progress, mean/peak speed and jitter. Source, inputs, actor classes, frame counts and video hashes are saved under `runs/rhythm-changes-current-pair-20261125/`. No training, tuning or controller reselection occurred. This known development-course showcase does not alter the fresh-test assessment or the NEAT prototype's failed quality gates. Command: `& .\.venv\Scripts\python.exe research/record_final_pair.py`. Gallery and exactly two MP4s: the run's `videos/` folder, 1x playback at15fps with a two-second final hold.

### SIMPLE-METHOD-PILOT-V1 — compact objective sensitivity

On 25 November 2026 the user activated a new journal-methodology goal and requested six to eight evolution workers. Earlier results remain frozen. The audit is in `research/Simple_NEAT_Method_Audit.md`; primary sources are in `research/Simple_NEAT_Method_Primary_Sources.md`; commands are in `docs/neat-simple-study.md`.

New breeding fitness is `C + U/(1+U)`: C is completed-course count, U is mean progress for incomplete batches, otherwise mean normalized successful route speed divided by `1 + lambda*sqrt(command_jitter)`. Three fixed development arms use lambda 0, 1 and 10. Each has 20 generations, target 100 genomes, six shared courses/genome and top-three distinct-genotype 60+20 consulted validation. All arms reuse exactly the same native-fast g134 warm-start population/RNG and first 20 batches, with fixed original mutation powers. This is one-ancestry warm-start development, not independent confirmation or proof of the old quality target. Nominal training is 12,000 episodes/arm; maximum validation is 4,800 slots/arm, with exact duplicate caches counted separately. No automatic extension.

The latest focused checks passed 42 tests. An earlier compatibility run passed 61 checks before the additional committed-generation reporting check. Twelve real six-worker replay episodes matched the preserved parent/model and prepared-genome results. Preparation sealed 56 execution source files and 320 inputs; protocol SHA-256 is `2ff5ecec155d293acf3c52e58802d65ff224157ace8bc0a7b379e242504cd4a5`. Checks and replay evidence are under `runs/neat-simple-pilot-v1/preflight/`.

The reused trainer now accepts an explicit condition label bound to its checkpoint signature. Old runs continue to use their own frozen source. Historical quality gates remain diagnostic only; they never enter the new breeding fitness. The historical `best.pkl` must not be relabeled as a new-method winner. The new report selects by each arm's own compact validation objective, retains every candidate and excludes uncommitted generations using the population checkpoint.

Launched with six workers: `& .\.venv\Scripts\python.exe run_neat_simple_study.py pilot --all --workers 6`. Dated status: running; lambda-0 had two generations committed, with remaining arms pending. Current authority is each arm's checkpoint/log/completion file, not this snapshot. Interim outputs are under `runs/neat-simple-pilot-v1/report/`. Independent populations, fresh final assessment, final figures and the reproducibility/methods package remain outstanding; the goal is active.

### SIMPLE-METHOD-PILOT-V1-COMPLETE — matched20-generation readout

All three arms finished20generations at6workers:36,000total candidate training episodes. New validation episodes lambda0/1/10:4,400/3,520/3,200; cachedslots400/1,280/1,600. Preflight12episodes separate. All6,000actual logged genotype scores recomputed exactly; all600generation-zero genotype/course trajectories exactly reproduced across the three shared initial populations.

Each own-objective selected model completed60/60primary+20/20retention. Minimum required straight speeds0.9224/0.8786/0.9211m/s missed0.95target. Mean command-change RMS0.07527/0.06762/0.05862; maximum straight yaw RMS0.6081/0.9588/0.6075rad/s. Stronger command smoothing did not establish smooth physical motion. Paired mean time against tunedPID on59sharedfinishes:6.027/6.009/6.043s vs6.893s. These are one-ancestry development outcomes, not independent validation or a replacement for the accepted prototype.

Full paired report, verified immutable snapshot3864b04984d3709ad84f2fa3b24153b79e40230673450ade4538cdbc8446e74f and numerical recomputation audit are under results/neat-simple-pilot-v1/. Latest broad implementation check94passed; subsequent stock-source/path checks18passed.25installedNEATPythonfiles matched installationRECORD. Independent-run implementation remains prospective: full preparation, real48episode instrumentation preflight and5freshlineages have not run.

User explicitly selected50TOTALgenerations/variant. Preparing separate matched continuation of saved generation20populations/RNG, adding only20–49 with unchanged reward/NEATparameters. No automatic100generation extension. No new physical-motion reward has been wired or evolved. The research goal remains active; pilot budget completion is not goal completion.

### SIMPLE-METHOD-PILOT-50-V1 — approved exact continuation to50total

User selected50totalgenerations for each variant and requested a clear stopping boundary. Newrun runs/neat-simple-pilot-50-v1 preserves the20-generation parent unchanged. Terminal generation20populations/RNG/species/indexers were copied exactly; only signature/protocol changes declare the approved50budget. All50coursebatches come from the same frozen original schedule; first20 match byte-for-byte. Historical cache signatures are allowed only for the exactly sealed parent. The physics/policy/fitness sources remain identical; no physical-motion candidate is wired.

Twenty-two continuation/ancestry/reporting checks passed. Real boundary serialization identity was verified for all three arms. All6,000 inherited scores recomputed exactly; analysis selected the same inherited controllers and reported zero new training. The child seals65sources and1,110inputs, including sealed parent source/history for portable verification. Protocol SHA-256 a95c244174cc921b423827cde6e1789971d6e1044a71cfdca6f17cc0b8ffaac5. Boundary proof and stockNEATRECORD checks are in preflight/boundary-integrity.json. Inherited12-episode physics replay is explicitly notnewcompute.

Launched sequential continuations at6workers with run_neat_simple_continue.py train --all --workers6. Newbudget30generations/arm, nominal18,000candidateepisodes/arm and maximum7,200newvalidation slots/arm; cumulative50generations/30,000candidateepisodes perarm. Stop at50 and report all results; no automatic100extension or independent confirmation. These3 related arms are not the5 independent lineages required by the full still-active goal. Commands and interpretation are in docs/neat-simple-study.md; matched reports/audits under results/neat-simple-pilot-50-v1/.

Completed outcome: all three arms stopped at50, process exit0. Actual cumulative training episodes30,012/30,000/30,000; continuation-only54,012 total. All15,002 genome scores recomputed with zero discrepancies. New validation episodes after20:5,600/5,520/5,280; cumulative10,000/9,040/8,480. Parent/source/input verification passed after completion. Immutable analysis snapshot5c4e5e5ea6f1f2461040cdaba557279f983f78da5daa11f47a382f5f373a673a replayed report.json byte for byte. PNG/SVG learning curves and unique-genotype command/motion trade-off plots were visually inspected.

Own-objective selections: lambda0 g44/r0, lambda1 g42/r2, lambda10 g28/r1. All finish60primary+20retention; minimum straight speeds0.95483/0.95546/0.91435m/s; worst straight yaw RMS0.26934/0.35009/0.67650rad/s versus accepted prototype0.13444. Mean geometric RMSE17.231/19.360/20.220mm versus prototype16.462mm. On59shared PID finishes, times5.9797/5.9525/6.0237s versus PID6.8927s. Lambda1 g27/r0 is an exploratory near-target alternative (0.97538m/s, yaw0.15072, RMSE20.570mm), not the declared selection. Extra generations helped; command smoothing alone did not establish the full physical-quality goal. No100-generation extension, coefficient decision or five-seed confirmation started. Full results, model hashes and commands: [completed readout](neat-simple-pilot-50-results.md).

Prospective confirmation now requires explicitly identified completed50-generation evidence, matching terminal checkpoints/report, all frozen input hashes and intact snapshot payloads/archived analysis. Nineteen focused decision checks and16 related trainer/assessment/report checks passed;10 pilot-analysis checks passed. These software checks do not establish five independent trained lineages. The full goal remains active pending the next methodology decision.

### ENDURANCE-SIMPLE-LAMBDA0-20261125 — requested frozen-controller showcase

The user named the unchanged Rhythm Changes long course **Endurance Track** and requested a fitness double-check plus a replay of the selected simple-fitness model. The accepted lambda0 g44/rank0 genome4323 (file SHA2569a04af4033ce75ab745422ee1183bb089756b92c5aab1f47bba7cf1cd067208e) was fixed before evaluation. At cap1.00m/s it completed all45.7739m in59.50s, mean/peak measured chassis speed0.79791/0.99184m/s, geometric RMSE25.738mm and command jitter0.00851668. One headless evaluation and one identical checked rendering replay produced exactly one requested video, with no new evolution or tuning. The display alias preserves the original track hash77fb2a4f7caad1dd33e3594335e1639faf2d03f339e25694b4dc3744c9205d6a and old records. Outputs/source/protocol: runs/endurance-simple-lambda0-20261125/.

Fitness re-audit reproduced all15,002 three-arm genome scores; independent lambda0 inspection reproduced all5,002 records and rescored150 candidate slots. For selected genome4323, logged/manual/frozen-scorer training fitness6.409444877656034 agree exactly, and changing valid jitter/RMSE/historical-score fields leaves the selected fitness unchanged. Earlier nativeg134 ancestry did use explicit jitter and legacy shaping, confirmed in its original source and logs; native RMSE was a validation gate, not a breeding penalty. Historical episode score/gate fields remain diagnostic. Full results, naming provenance, reproduction command and limitations are in [Endurance Track](endurance-track.md). Further training still requires an explicit start instruction.

### SIMPLE-FITNESS-001 — accepted research finding recorded

At the user's request, preserved the exact chosen equation, actual-score verification, validation/retention/PID comparison, 20-to-50-generation changes, Endurance Track showcase, frozen model/source identities and suggested paper wording in [the simple-method research audit](../research/Simple_NEAT_Method_Audit.md#simple-fitness-001--successful-continuation-without-direct-tracking-or-smoothness-penalties). The supported result is successful continued evolution without direct jitter/RMSE breeding penalties. The earlier ancestor used smoothness shaping; no penalty-free from-scratch, independent-seed or untouched-test claim is made. Documentation only; no new experiments, manuscript rewrite or training launched.
