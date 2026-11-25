# Audit and simplification decision — 25 November 2026

**Latest outcome:** the user accepted the completed simple-fitness development results and selected lambda = 0. The finding and verified evidence are recorded below as **SIMPLE-FITNESS-001**. Earlier sections preserve the original development rationale. Independent replication remains future work; additional evolution requires the user's explicit start instruction.

This begins the new, active journal-methodology goal. It does not reopen or overwrite earlier experiments, change the accepted prototype, or retroactively repair its failed gates. The user authorized autonomous methodology development and requested six to eight workers for evolution. Initial new evolution will use six workers, bounded caches and worker recycling.

## What current evidence establishes

The frozen accepted prototype completed 148/150 fresh tracks at 1 m/s, versus tuned full-base PID's 141/150. It was 14.6% faster on 140 shared finishes and had 21.2% lower geometric RMSE, but 7.6 times the command jitter. It is one development ancestry. The 150-track comparison was already inspected and cannot serve as an untouched final test for a new protocol. Evidence: `docs/final-results-20261125.md`, `results/neat-pid-finalization-20261125/report.json`, and their frozen controller manifests.

The prototype failed its original native-reference RMSE and jitter gates. Good videos do not establish satisfaction of those gates or repeatability across seeds. The saved group diagnostic also shows that equally weighted speed/mixed/corner training can hide mixed-course losses when validation gives mixed courses more weight.

## Constants and their actual status

| Choice | Purpose | Evidential status |
|---|---|---|
| 10 points per completion; secondary reward at most 9 | Separate completion-count bands | Ordering is provable; particular numerical spacing and its effect on species allocation are not proven best. |
| 0.95 m/s | Define near-full-speed driving under a 1 m/s wheel cap | Declared engineering target: 95% of cap. Not supplied by a paper or established as a hardware necessity. |
| 0.15 rad/s yaw and 0.02 normalized steering | Define acceptable straight motion | Inherited project tolerances, not independently calibrated physical limits. |
| 5 mm RMSE and 0.0005 jitter training floors | Avoid zero reference denominators | Numerically motivated; exact values unvalidated. |
| 75/25 straight/route-speed weights | Favor dedicated straight speed | Project preference; no sensitivity evidence establishing necessity or superiority. |
| 80/20 partial progress/quality weights | Shape unsuccessful-batch search | Project preference, unvalidated. |
| 0.05 m/s shortfall divisor | Normalize speed deficit by cap-minus-target | Arithmetic rationale; still determines tradeoff against other normalized violations. |
| 0.20 command-jitter penalty and inherited legacy term | Earlier smoothness and compatibility shaping | Later breeding objectives replaced these; legacy arithmetic has known historical unit/sign issues and should not be retained in a new primary reward. |
| Lower mutation powers in the successful final attempt | Local refinement | Plausible mechanism; parent, objective and powers changed together, so their effects are not isolated. |
| Four objectives and stage-dependent initialization | Historical development path | Genuine experimental history, not a single prospectively confirmed recipe. |

Neither a short equation nor a citation makes numerical choices valid. The journal case needs disclosure, measured robustness and bounded claims. Published methodological context is in `research/Simple_NEAT_Method_Primary_Sources.md`.

## Chosen first development test

Test one compact scalar reward with completion-count priority and one smoothness coefficient. For `m` common episodes, let `C` be completed count. Let `U` be mean ordered-route progress if the batch is incomplete; otherwise let it be mean normalized successful route speed discounted by RMS normalized wheel-command change:

```text
U = mean(progress)                                              if C < m
U = mean((L/(completion_time*cap))/(1 + lambda*sqrt(jitter)))      if C = m
F = C + U/(1+U)
```

For finite nonnegative U, the residual is below one, proving completion priority. The bounding transform changes scalar fitness spacing and therefore can affect stock NEAT species allocation. It is an explicit design choice, not a claim of equivalence to another reward or parameter-free optimization.

The pilot compares lambda 0, 1 and 10. Zero removes smoothness pressure; at RMS normalized command change 0.1, the other choices discount speed utility by 9.1% and 50%. These bracket a weak and a strong preference on a declared development range. They are not literature-validated or claimed optimal. This differs from the research note's initially suggested narrower 0/1/2 range and is fixed before any new rescore or evolution outcome is consulted.

All three arms reuse the exact native-1 m/s g134 warm-start population and RNG from the prior fast arm, with identical first 20 six-course batches and fixed standard mutation parameters. Each arm gets 20 generations, target population 100, six courses/genome, and top-three distinct-genotype validation on the consulted 60+20 suite. Nominal training is 12,000 episodes/arm; validation has at most 4,800 slots/arm, with exact duplicate caches separately counted. No automatic extension.

This isolates the new coefficient on a useful common development initialization. It cannot establish from-scratch learning, independent seeds, or final generalization. Old threshold-based measurements remain diagnostic only. The existing trainer's `best.pkl` still means historical-gate eligibility; it must not be presented as a winner under the new objective. All candidate genomes and validation records are retained for explicit development reporting.

## Remaining requirements for the active goal

1. Run meaningful reward-ordering and resume/dispatch checks, then the bounded pilot; report all arms and failures.
2. Inspect measured speed, completion, geometric error, command-change RMS and physical yaw. Do not treat command jitter as measured chassis vibration or reward simplification as evidence of stability.
3. If the candidate is promising, freeze a from-scratch curriculum using the same reward and NEAT configuration throughout. Address training/validation mixture deliberately. If not, preserve the negative result and define the next limited hypothesis.
4. Train five genuinely independent populations with fixed budgets and selection rules. Warm-start copies of one lineage do not meet this requirement.
5. Freeze chosen controllers and PID gains before new shared tests. Existing PID development tunes and all consulted tracks remain development evidence. Report paired denominators, every failure, per-seed spread and appropriately scoped uncertainty.
6. Produce standalone reproducibility instructions, meaningful figures and a LaTeX methods/results draft that describes the actual successful protocol, including unsuccessful development attempts and limitations.

Acceptance by a journal is outside the agent's control. The goal remains active while independent validation and these deliverables are incomplete.

## SIMPLE-FITNESS-001 — successful continuation without direct tracking or smoothness penalties

Recorded 25 November 2026, following the completed 50-generation comparison and the user's acceptance of the observed performance trade-offs. This is an empirical finding from one pretrained ancestry, not a universal claim about NEAT or a from-scratch result.

### Finding

A compact fitness prioritizing completion, progress and successful route speed produced a fast, reliable controller with user-accepted driving behavior, without directly including jitter, geometric RMSE, yaw, steering or legacy reward in its breeding fitness. The lambda-0 continuation retained useful tracking and physical stability even though these quantities were measured only as outcomes.

The result suggests that explicit smoothness and tracking penalties were not necessary for acceptable behavior in this particular continuation. It does not establish that removing them caused the behavior, that smoothness emerges from scratch, or that the resulting motion is as smooth as PID or the earlier accepted prototype.

### Exact chosen equation

For N common courses, C successful completions, route progress p_i, route length L_i, completion time T_i and wheel-speed cap v_cap:

\[
F=C+\frac{U}{1+U},\qquad
U=\begin{cases}
\frac{1}{N}\sum_{i=1}^{N}p_i,& C<N,\\[4pt]
\frac{1}{N}\sum_{i=1}^{N}\frac{L_i}{T_i v_{\mathrm{cap}}},& C=N.
\end{cases}
\]

N is six for training and 60 for primary validation. For incomplete batches, speed does not enter fitness: an additional completion takes priority, then mean progress distinguishes candidates with equal completion counts. Once all courses are completed, mean per-course normalized route speed determines the secondary utility. Because U is finite and nonnegative, U/(1+U) is below one; another completion always outranks a secondary improvement. The numerical implementation also guards against floating-point rounding to the next completion band.

The selected lambda-0 version has no tuned reward weights. The bounding transform, completion-first priority, batch composition, cap and evolution settings remain explicit design choices. The 0.95 m/s straight-speed target is an evaluation target, not a term in this fitness. Normalized route speed is route length divided by completion time and cap, not measured chassis speed or commanded wheel speed.

### Evidence and performance

The three development arms used the same initial pretrained population/RNG and saved course schedule, target population 100, six courses per genome, six CPU workers and the same fixed NEAT parameters. Each arm completed 50 total generations. Selection used highest own-coefficient fitness on 60 primary validation courses, with earliest generation/rank resolving ties. The selected lambda-0 controller is generation 44, candidate rank 0, genome 4323. The 20 low-speed retention courses were measured separately.

| Outcome | Selected lambda-0 controller |
|---|---:|
| Primary completion, cap 1.00 m/s | 60/60 |
| Retention completion, cap 0.50 m/s | 20/20 |
| Minimum sustained required-straight speed at cap 1.00 | 0.954827 m/s |
| Mean primary geometric RMSE | 17.231 mm |
| Worst measured straight yaw RMS | 0.269338 rad/s |
| Mean primary command-change RMS | 0.085701 |
| Mean finish time on 59 shared PID successes | 5.9797 s versus PID 6.8927 s |
| Paired mean geometric RMSE on those 59 tracks | 16.706 mm versus PID 20.925 mm |

The paired finish-time reduction is 13.25%. PID completed 59/60 primary courses. The accepted earlier NEAT prototype completed 60/60 and had mean RMSE 16.462 mm and worst straight yaw RMS 0.134443 rad/s. Thus the simple controller's outcome includes an explicit smoothness/tracking trade-off against that prototype; acceptance does not retroactively change any historical gate.

Extending the pilot from 20 to 50 generations improved the selected lambda-0 model's minimum straight speed from 0.9224 to 0.9548 m/s and reduced worst straight yaw RMS from 0.6081 to 0.2693 rad/s. Lambda 10 produced lower command-change RMS but a slower minimum straight speed and higher worst physical yaw than lambda 0 at the final selection. Command variation and physical body motion must therefore be reported separately. These related-arm observations do not estimate a repeatable coefficient effect across independent seeds.

### Verification of the actual breeding fitness

1. Inspected the frozen scorer, launcher, compatibility wrapper and trainer assignment to `genome.fitness`. The simple result overwrites the historical diagnostic fitness before reproduction. Historical gates do not determine this own-objective development selection.
2. Recomputed all 5,002 lambda-0 genome scores across generations 0–49: zero mismatches. The complete three-arm audit reproduced all 15,002 scores without discrepancies.
3. Independently rescored all 150 lambda-0 validation candidate slots, confirming generation 44/rank 0. Its validation fitness is 60.419481856638.
4. For selected genome 4323's six training courses, C = 6 and U = 0.6933220323801613. Logged, manual and frozen-scorer fitness all equal **6.409444877656034**.
5. Changed valid command-jitter values from 0 to 1 and RMSE fields from 0 to 1,000 m, also changing historical score/legacy fields: the selected training fitness remained exactly unchanged. An independent compatibility-wrapper check likewise preserved training and validation fitness while historical diagnostics changed.
6. Checked the protocol fingerprint, 65 frozen source files, sealed parent sources/history, inherited raw data and terminal generation-50 checkpoint. Original logs and historical quality-gate failures remain intact.

Old episode-level `score`, fitness components and diagnostic gates can still contain jitter/RMSE calculations. They are not the selected breeding fitness. The zero-weight scorer still validates and logs jitter measurements; their values do not influence the selected score.

### Endurance Track showcase

The user named the unchanged historical Rhythm Changes course **Endurance Track**. The selected lambda-0 model was fixed before its requested run on this known course. It completed all 45.7739 m in **59.50 s**, at a 1.00 m/s wheel-speed cap, with mean measured chassis speed **0.79791 m/s**, peak **0.99184 m/s**, geometric RMSE **25.738 mm** and command jitter **0.00851668**. The checked rendering replay matched the saved headless result. One requested video was produced; no new tuning or evolution occurred.

This is a single deterministic development-course showcase. It does not establish across-track or across-seed robustness. Full course/results details and commands: [Endurance Track](../docs/endurance-track.md).

### Warm-start limitation and paper wording

The native generation-134 ancestor previously used an explicit `-0.20 * command_jitter` term and bounded legacy shaping. Its original generation-134 log contains mean jitter contribution -0.001541140630412865. RMSE entered that ancestor's validation gate, not a direct breeding reward. Useful tracking or smoothness could therefore have been inherited from earlier evolution. Successful completion and route-speed pressure may also indirectly discourage unstable behavior, but this mechanism has not been isolated experimentally.

Suggested paper wording:

> In a warm-start development experiment, a completion-prioritized fitness based solely on route progress and normalized successful route speed yielded a controller that completed all 60 primary and 20 low-speed retention courses. On 59 mutually completed courses at a 1 m/s wheel-speed cap, its mean completion time was 13.25% lower than that of the tuned PID baseline. Jitter and geometric tracking error were measured but did not enter the continuation's breeding fitness. The controller's ancestor had previously received smoothness shaping, so the finding establishes successful reward simplification during continued evolution rather than penalty-free learning from scratch.

Do not describe this as proof that jitter/RMSE rewards are universally unnecessary, an optimal fitness function, five-seed confirmation, superior smoothness to PID, or untouched-test generalization. From-scratch learning and independent replication remain separate future questions.

### Frozen artifacts and reproduction

- Chosen model: `results/neat-simple-pilot-50-v1/accepted-development/neat-lambda0-g44-r0.pkl`.
- Model SHA-256: `9a04af4033ce75ab745422ee1183bb089756b92c5aab1f47bba7cf1cd067208e`.
- Genotype SHA-256: `141cba6068602d96d2b699a61137016a69359a31c4cc9fd3aad4a9cf671625d4`.
- Decision/rationale: `research/simple-neat-development-decision.json`.
- Protocol SHA-256: `a95c244174cc921b423827cde6e1789971d6e1044a71cfdca6f17cc0b8ffaac5`.
- Immutable full analysis snapshot: `results/neat-simple-pilot-50-v1/snapshots/5c4e5e5ea6f1f2461040cdaba557279f983f78da5daa11f47a382f5f373a673a/`.
- Candidate metrics, paired denominators and plots: `results/neat-simple-pilot-50-v1/report.json` and the adjacent report/PNG/SVG files. Raw consumed episodes are preserved inside the immutable analysis snapshot; original training/validation records remain under `runs/neat-simple-pilot-50-v1/development/`.
- Direct chosen-genome verification: `runs/endurance-simple-lambda0-20261125/fitness-verification.json`.
- Showcase video: `runs/endurance-simple-lambda0-20261125/videos/endurance-neat-lambda0-g44.mp4`.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py research/audit_simple_pilot.py --run runs/neat-simple-pilot-50-v1 --output results/neat-simple-pilot-50-v1/fitness-recomputation-audit.json
& $py research/report_simple_pilot.py --run runs/neat-simple-pilot-50-v1 --output results/neat-simple-pilot-50-v1
& $py research/record_endurance_simple.py
```

These commands verify/recreate analysis or the fixed-controller showcase; none starts evolution. Raw `runs/` data and MP4 files are excluded from Git, so preservation requires a separate backup of those files. No Git commit, push or remote backup is claimed here.
