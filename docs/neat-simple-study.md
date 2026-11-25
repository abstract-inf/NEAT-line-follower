# Simple NEAT reward development

This is a new, bounded development pilot for the active journal-methodology goal. It preserves every earlier run. It is not a completed methodology, a from-scratch confirmation, or a claim that the original strict quality target was met.

The [audit](../research/Simple_NEAT_Method_Audit.md) separates mathematically motivated constants from unvalidated tolerances. The [primary-source note](../research/Simple_NEAT_Method_Primary_Sources.md) explains why simpler reward design still requires independent evaluation.

## One task coefficient

The user accepted the completed 50-generation results and explicitly selected **lambda = 0**. The paper finding, verification evidence, outcome trade-offs and suggested wording are preserved as [SIMPLE-FITNESS-001](../research/Simple_NEAT_Method_Audit.md#simple-fitness-001--successful-continuation-without-direct-tracking-or-smoothness-penalties).

For this selected version, the effective equation is

\[
F=C+\frac{U}{1+U},\qquad
U=\begin{cases}
\frac{1}{N}\sum_i p_i,&C<N,\\
\frac{1}{N}\sum_i \frac{L_i}{T_i v_{\mathrm{cap}}},&C=N.
\end{cases}
\]

Here C is completed-course count, N is batch size (six during training), p is fractional route progress, L is route length and T is completion time. An additional completion always outranks the bounded secondary utility. Completed batches compete on cap-normalized route speed. There is no effective command-jitter penalty or tuned reward coefficient in this selected variant. Tracking and physical smoothness remain measured outcomes, not guarantees of the equation. This is the accepted simpler-fitness development outcome; independent-seed generalization remains future work.

For a common batch, use completed-course count plus a bounded utility. Utility is mean progress until all courses are complete. For a completed batch it is mean normalized route speed divided by `1 + lambda * sqrt(command_jitter)`. The bounded transform is `U/(1+U)`, so an additional completion always outranks a secondary improvement.

This removes the earlier legacy term, speed/quality mixture weights, five training quality thresholds and feasibility-band bonus. It does not guarantee good tracking, sufficient straight speed or smooth physical motion. These remain measured outcomes.

## Fixed pilot budget and controls

- Three arms: `lambda-0`, `lambda-1`, `lambda-10`.
- Exactly 20 generations per arm; target 100 genomes; six shared courses/genome.
- Same native-fast g134 warm-start population/RNG, same saved 20 training batches and same consulted 60 primary plus 20 low-speed validation tracks.
- Original bias/weight mutation powers 0.12/0.14 throughout; unchanged stock NEAT-Python 0.92 operators, 21 inputs, action mapping, MuJoCo physics and recovery.
- Six workers by default; six to eight allowed. Existing 64 MiB mask-cache setting and worker recycling remain.
- These are related development arms. Five independent confirmation populations and new final test tracks are still required.

The reused trainer records its old quality gates as historical diagnostics. Those gates and reference values do not enter the new breeding fitness. In particular, an old-style `best.pkl` is not a new-method winner. Report all measured tradeoffs instead of declaring gate failures successful.

## Terminal commands

```powershell
$py = '.\.venv\Scripts\python.exe'

# Verify the implementation before any evolution.
& $py -m pytest tests/test_neat_simple_fitness.py tests/test_neat_simple_study.py -q --basetemp .cache/simple-checks
& $py run_neat_simple_study.py prepare
& $py run_neat_simple_study.py verify
& $py run_neat_simple_study.py preflight --workers 6

# This starts the three bounded arms sequentially, each using six workers.
& $py run_neat_simple_study.py pilot --all --workers 6

# Or start exactly one arm.
& $py run_neat_simple_study.py pilot --arm lambda-1 --workers 6

# Resume committed populations; completed arms are verified and skipped.
& $py run_neat_simple_study.py pilot --all --workers 6 --resume

# Read committed progress without interfering with evolution.
& $py run_neat_simple_study.py status
& $py run_neat_simple_study.py report

# Matched comparisons with the accepted prototype and tuned PID; no physics.
& $py research/report_simple_pilot.py

# Recompute actual logged breeding scores from stored episode measurements.
& $py research/audit_simple_pilot.py
```

Preparation seals protocol, runtime source, input hashes, common initial population/RNG, validation suite and batches under `runs/neat-simple-pilot-v1/prepared/`. Execution uses its frozen source. Each arm writes checkpoints, raw episodes, candidate genomes, validation caches and actual worker history under `development/<arm>/`. Source/input changes invalidate resume rather than silently changing a running experiment.

This command does not launch independent confirmation or final assessment. Their protocol will be fixed after the development evidence is examined. No journal acceptance or universal reward optimality is promised.

The paired analysis writes `results/neat-simple-pilot-v1/report.json`, `report.md`, SVG/PNG learning-and-quality figures, and immutable input snapshots. Its per-arm development choice uses each arm's own validation objective; those numerical fitness values are not compared across coefficients. Finish times and paired errors use tracks both policies finish, with denominators. All-episode quality means are labeled separately. `fitness-recomputation-audit.json` reports actual score agreement, source/input fingerprints and, once all arms have started, whether the identical initial populations produced identical stored generation-zero trajectories.

## Completed 20-generation development readout

All three arms completed at six workers. Each performed 12,000 candidate training episodes; newly evaluated validation episodes were 4,400 / 3,520 / 3,200 for lambda 0 / 1 / 10. Cached validation slots were 400 / 1,280 / 1,600, respectively, and are not additional simulations. The preflight added 12 episodes. The audit recomputed all 6,000 actual per-genome scores with zero discrepancies; all three initial generations reproduced exactly the same 600 genotype/course trajectories.

| Coefficient | Primary / retention completion | Minimum straight speed m/s | Mean command-change RMS | Maximum straight yaw RMS rad/s | Mean geometric RMSE mm |
|---:|---|---:|---:|---:|---:|
| 0 | 60/60; 20/20 | 0.9224 | 0.07527 | 0.6081 | 16.71 |
| 1 | 60/60; 20/20 | 0.8786 | 0.06762 | 0.9588 | 16.77 |
| 10 | 60/60; 20/20 | 0.9211 | 0.05862 | 0.6075 | 20.05 |

These are each arm's own-objective validation selections. On 59 shared finishes with tuned PID, their mean times were 6.027 / 6.009 / 6.043 seconds versus PID's 6.893 seconds. None of those selections met the declared 0.95 m/s minimum-straight target. Stronger command smoothing lowered command-change RMS without a corresponding reduction in worst straight physical yaw. This warns against treating actuator-command variation as interchangeable with body-motion stability. It does not prove that the reward cannot improve with longer evolution.

Full immutable readout snapshot: `3864b04984d3709ad84f2fa3b24153b79e40230673450ade4538cdbc8446e74f`, under `results/neat-simple-pilot-v1/snapshots/`. These related development trials do not establish independent-seed repeatability and do not replace the accepted prototype.

The user subsequently selected **50 total generations per variant**, with a hard pilot stop at 50 and no automatic extension to 100. A separate continuation preserves this 20-generation study and its terminal populations/RNG states. It adds only generations 20–49 with the same objective, coefficients, NEAT parameters, cap, validation suite and original frozen course schedule. The conditional physical-motion scorer in `neat_simple_study/physical_fitness.py` remains unselected and unwired; it is not part of the approved matched continuation.

## Approved matched continuation: 50 total

New run: `runs/neat-simple-pilot-50-v1/`. The parent study stays byte-for-byte unchanged. Each child begins at its exact post-generation-19 population (next generation index20), including RNG, species, node/genome indexers and inherited records. Only its protocol signature changes to declare the larger approved budget. Historical cache signatures are accepted only when they exactly match the sealed parent and evaluation inputs remain identical.

Preparation verified complete state serialization identity for all three real boundaries, not just the selected genomes. A new audit reproduced all6,000 inherited scores, and the inherited analysis selected the same three controllers with **zero new training episodes**. Runtime source/policy/physics/scorer comparisons passed and all25 installed NEATPython source files matched their installation RECORD. Twenty-two continuation/ancestry/reporting checks passed. The child seals65 execution source files and1,110 immutable inputs, including the original parent source and data; verification can use those sealed copies when the original location is unavailable.

Protocol SHA-256: `a95c244174cc921b423827cde6e1789971d6e1044a71cfdca6f17cc0b8ffaac5`. Boundary evidence: `runs/neat-simple-pilot-50-v1/preflight/boundary-integrity.json`. The old12-episode physics replay is inherited evidence, not12 new simulations. The new budget is30 generations/arm, nominally18,000 new candidate training episodes and at most7,200 new validation slots per arm; cumulative target is50 generations and30,000 candidate training episodes per arm. Logs record actual populations and exact cache reuse.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py run_neat_simple_continue.py prepare
& $py run_neat_simple_continue.py verify
& $py run_neat_simple_continue.py train --all --workers 6

# The continuation command always resumes its prepared/current checkpoint.
# Reissue after interruption; completed50-generation arms are verified/skipped.
& $py run_neat_simple_continue.py train --all --workers 6
& $py run_neat_simple_continue.py status

& $py research/report_simple_pilot.py --run runs/neat-simple-pilot-50-v1 --output results/neat-simple-pilot-50-v1
& $py research/audit_simple_pilot.py --run runs/neat-simple-pilot-50-v1 --output results/neat-simple-pilot-50-v1/fitness-recomputation-audit.json
```

All three arms run sequentially with six workers. At50, stop and examine validation completion, minimum sustained straight speed, geometric error and physical yaw as well as actuator variation. No automatic100-generation extension or independent confirmation is launched. The full research goal additionally requires a defensible successful method, five actual independent lineages, a frozen-controller fresh shared PID assessment and the reproducibility/methods handoff; merely finishing this pilot does not satisfy it.

Replay one captured analysis without accessing live training files by running its archived `analysis.py --snapshot <snapshot-directory> --output <new-directory>`. Each snapshot includes the reporting source, compact scorer, raw consumed episodes and controller bytes. The [prospective independent study](neat-simple-confirmation.md) is separate and remains inactive until its development decision and real preflight are complete.

## Completed 50-generation readout

All three arms finished at 50. The [full readout](neat-simple-pilot-50-results.md) records exact model paths/hashes, 20-to-50 changes, retained low-speed ability, paired PID comparisons, compute counts and limitations. Lambda 0 and 1 selections now exceed 0.95 m/s on every required straight and finish 60/60 primary plus 20/20 retention courses, but have higher worst straight yaw than the accepted prototype. Lambda 10 smooths commands further without achieving the speed target. No independent confirmation was launched. The accepted prototype remains unchanged.

The audit recomputed all 15,002 actual scores from 90,012 cumulative training episodes without discrepancies. Final analysis snapshot `5c4e5e5ea6f1f2461040cdaba557279f983f78da5daa11f47a382f5f373a673a` replays the numerical report byte for byte. Learning/quality curves and command-versus-physical-motion trade-off figures are saved in PNG and SVG under `results/neat-simple-pilot-50-v1/`.
