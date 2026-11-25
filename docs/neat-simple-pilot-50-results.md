# Completed 50-generation simple-reward pilots

## Outcome

The extension improved the evidence: two own-objective selections now meet the declared minimum sustained straight speed of 0.95 m/s. All three selections complete the 60 primary and 20 low-speed retention courses. However, none matches the accepted prototype simultaneously in minimum straight speed, mean geometric error and worst measured straight yaw. This is a useful development result, not a completed replacement methodology. No independent confirmation or additional evolution was launched.

These are three related arms sharing one warm start and the same course schedule, not three independent seeds. All comparisons below use consulted validation courses, not fresh test data.

## Selected controllers at the 1.00 m/s cap

Selection is highest own-coefficient validation fitness over all 50 generations, ties earliest generation/rank. Fitness values across coefficients are not directly comparable. Generation/rank numbers are zero based.

| Arm | Generation / rank | Completion | Minimum straight m/s | Mean geometric RMSE mm | Maximum straight yaw RMS rad/s | Mean command-change RMS |
|---|---:|---:|---:|---:|---:|---:|
| lambda-0 | 44 / 0 | 60/60 | 0.95483 | 17.231 | 0.26934 | 0.08570 |
| lambda-1 | 42 / 2 | 60/60 | 0.95546 | 19.360 | 0.35009 | 0.06472 |
| lambda-10 | 28 / 1 | 60/60 | 0.91435 | 20.220 | 0.67650 | 0.05126 |
| accepted-prototype | reference | 60/60 | 0.95018 | 16.462 | 0.13444 | 0.07628 |
| tuned-pid | reference | 59/60 | 0.93618 | 22.844 | 0.09506 | 0.03006 |

Quality columns above are all-episode means except worst-window yaw/minimum-window speed. PID has one off-route failure. Command-change RMS measures wheel-command changes per fixed control step; physical yaw measures actual simulated body rotation. They are different outcomes.

## Matched successful-track comparisons with PID

All three pairs share 59 successful tracks out of 60. Each row uses its own shared-success intersection. Lower time and RMSE are better. The denominator excludes the PID failure.

| Arm | NEAT / PID time s | Time reduction | NEAT / PID geometric RMSE mm |
|---|---:|---:|---:|
| lambda-0 | 5.9797 / 6.8927 | 13.25% | 16.706 / 20.925 |
| lambda-1 | 5.9525 / 6.8927 | 13.64% | 18.942 / 20.925 |
| lambda-10 | 6.0237 / 6.8927 | 12.61% | 19.778 / 20.925 |

## What changed between 20 and 50 generations?

| Arm | Minimum straight speed: 20 -> 50 m/s | Worst straight yaw: 20 -> 50 rad/s |
|---|---:|---:|
| lambda-0 | 0.9224 -> 0.9548 | 0.6081 -> 0.2693 |
| lambda-1 | 0.8786 -> 0.9555 | 0.9588 -> 0.3501 |
| lambda-10 | 0.9211 -> 0.9144 | 0.6075 -> 0.6765 |

The 20-generation result was insufficient to rule out improvement: lambda 0 and 1 improve substantially in both minimum straight speed and worst yaw. Lambda 10 lowers command-change RMS further but its selected model becomes slightly slower on the worst straight and has higher worst yaw. This supports measuring physical motion directly instead of treating command smoothing as a guarantee of physical stability. It does not establish a causal coefficient effect across independent lineages.

A notable non-selected candidate is lambda-1 generation 27 rank 0: 60/60 primary and 20/20 retention, minimum straight speed 0.97538 m/s, worst straight yaw RMS 0.15072 rad/s, mean geometric RMSE 20.570 mm. Its yaw is close to the accepted prototype, but tracking error is higher. It remains an exploratory candidate; selecting it manually would change the stated own-objective selection rule. No evaluated candidate simultaneously matches the prototype on completion, the 0.95 speed target, mean geometric error and worst straight yaw. This last comparison is a diagnostic of the practical goal, not a new breeding gate.

## Low-speed retention

All selections complete 20/20 separate retention courses at a 0.50 m/s cap.

| Arm | Minimum straight m/s | Mean geometric RMSE mm | Worst straight yaw RMS rad/s |
|---|---:|---:|---:|
| lambda-0 | 0.47764 | 3.390 | 0.12634 |
| lambda-1 | 0.48401 | 4.171 | 0.12112 |
| lambda-10 | 0.45716 | 4.318 | 0.34977 |

## Compute, integrity and model locations

| Arm | Cumulative training episodes | New training episodes after 20 | New validation episodes after 20 | Cumulative new validation episodes | Cached validation slots |
|---|---:|---:|---:|---:|---:|
| lambda-0 | 30012 | 18012 | 5600 | 10000 | 2000 |
| lambda-1 | 30000 | 18000 | 5520 | 9040 | 2960 |
| lambda-10 | 30000 | 18000 | 5280 | 8480 | 3520 |

The audit recomputed 15,002 actual genome scores from 90,012 cumulative training episodes with zero discrepancies. The continuation added 54,012 training episodes; inherited episodes are counted once. Target population is 100, but stock species reproduction rounding produced two extra evaluated genomes in lambda 0. The original 12 preflight episodes are inherited evidence, not new simulations. All three arms stopped at 50; process exit code was zero. Frozen source/input/ancestry verification passed after completion, including preservation of the original 20-generation run.

Nineteen focused confirmation-decision checks, sixteen related trainer/assessment/report checks and ten pilot-analysis checks passed. Those are software checks, not proof of independent-seed learning. Archived analysis replay reproduced report.json byte for byte.

Selected development model files (no models overwritten):

- `runs/neat-simple-pilot-50-v1/development/lambda-0/champions/generation-0044-rank-0.pkl`
  - File SHA-256: `9a04af4033ce75ab745422ee1183bb089756b92c5aab1f47bba7cf1cd067208e`; genotype: `141cba6068602d96d2b699a61137016a69359a31c4cc9fd3aad4a9cf671625d4`.
- `runs/neat-simple-pilot-50-v1/development/lambda-1/champions/generation-0042-rank-2.pkl`
  - File SHA-256: `3d8b028e96fdb5122491aa64d79c284daefb42e78d6e07d7c557d9755b60576c`; genotype: `0788ea348143041027b89eed8ecd111caec550c034ee4e5de595ab969684c6fc`.
- `runs/neat-simple-pilot-50-v1/development/lambda-10/champions/generation-0028-rank-1.pkl`
  - File SHA-256: `d8caabfee86a8e7de80db1a59a76c42f11cc8f8337a76b37f0b7799d09c59808`; genotype: `763bb81309b68e5e381f2866255015760806b682cdcf406834bab1355abef934`.

## Reproduction and figures

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py run_neat_simple_continue.py verify
& $py research/audit_simple_pilot.py --run runs/neat-simple-pilot-50-v1 --output results/neat-simple-pilot-50-v1/fitness-recomputation-audit.json
& $py research/report_simple_pilot.py --run runs/neat-simple-pilot-50-v1 --output results/neat-simple-pilot-50-v1
```

These commands perform verification/analysis, not evolution. Outputs under `results/neat-simple-pilot-50-v1/`: `report.json` (all candidates, failures, paired metrics and denominators), `report.md`, `learning-and-quality.{png,svg}`, `command-motion-tradeoffs.{png,svg}`, and the fitness audit. Scatter plots show distinct fully completing NEAT genotypes; reference PID is displayed separately even though it has one failure. Related candidates are not independent observations.

Final captured analysis snapshot: `5c4e5e5ea6f1f2461040cdaba557279f983f78da5daa11f47a382f5f373a673a`. Run its archived `analysis.py --snapshot <snapshot-directory> --output <new-directory>` to reproduce from sealed inputs.

## Next decision

The user subsequently accepted these results as satisfactory and chose lambda = 0. The simpler-fitness development task is accepted as successful, with the measured smoothness trade-offs disclosed. No additional reward tuning is planned. This acceptance does not alter historical quality-gate failures or establish independent-seed performance.

The chosen development controller is preserved at `results/neat-simple-pilot-50-v1/accepted-development/neat-lambda0-g44-r0.pkl`, with exact fingerprints and measured outcomes in the adjacent `manifest.json`. The coefficient decision and rationale are in `research/simple-neat-development-decision.json`. The prior prototype and all other pilot outputs remain unchanged.

No five-seed populations have been prepared or evolved. The 400-generation from-scratch curriculum in the prospective confirmation document is a draft for future review, not an approved next launch. Further training requires an explicit user start instruction. The broader saved research goal still includes independent runs and a fresh shared PID assessment; those requirements remain incomplete. Do not automatically extend the pilot, launch confirmation, or reinterpret the accepted development result as five-seed evidence.
