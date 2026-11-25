# Prospective from-scratch confirmation

Status: **draft implementation, not registered, prepared or running**. The [warm-start continuation](neat-simple-study.md) completed 50 generations in all three variants. The user accepted the development results and selected lambda = 0; the saved decision is `research/simple-neat-development-decision.json`. Further evolution requires the user's explicit start instruction. The 400-generation schedule below remains a prospective design for review, not an approved launch. A tested helper or short equation alone does not establish from-scratch learning performance.

## Intended method

Start five independent random NEAT-Python 0.92 populations, each with target population 100. No pretrained controller is used for initialization. Preserve one continuous population per seed and keep the same compact fitness and NEAT parameters throughout. There is no winner-plus-mutants restart, stage-specific reward or late mutation-power change.

Fixed NEAT settings means the mutation, speciation and reproduction configuration stays fixed; network weights and topology continue evolving normally. All 25 installed NEAT Python source files are checked against their installation RECORD and frozen by hash. Runtime source, Python/package versions and the coefficient's development decision are also bound to the protocol.

The proposed 400-generation budget is a conservative development choice based on the earlier successful ancestry, not a demonstrated minimum or an optimal stopping rule. No unsuccessful seed receives an automatic extension.

| Zero-based generations | Wheel-speed cap |
|---|---:|
| 0–149 | 0.50 m/s |
| 150–199 | 0.65 m/s |
| 200–249 | 0.80 m/s |
| 250–399 | 1.00 m/s |

Every genome in a generation receives the same six courses: three fresh mixed, one isolated straight, one post-turn straight, and one fresh corner drill. This 3:2:1 group ratio matches the primary validation and proposed final-test proportions. Speed courses cycle through a frozen bank of 20. Corner directions alternate and drill sequences cycle.

The initial mixed courses are beginner difficulty for generations 0–49. Generations 50–99 use one beginner and two intermediate courses. From generation 100 onward, the three mixed courses contain one of each difficulty. Corner difficulty follows beginner/intermediate/advanced over those same intervals. All course generation uses independent deterministic streams, separate from evolution RNG.

The one smoothness coefficient is selected on the completed development pilot and frozen before these populations are evolved. Fitness is `C + U/(1+U)`. U is mean progress for incomplete batches; otherwise it is mean cap-normalized route speed discounted by `1 + lambda * RMS(command change)`. No historical quality-reference thresholds enter reproduction.

## Validation and outcome measurement

One training champion per generation receives 60 fixed shared primary validation courses at the current cap and 30 separate retention courses at 0.5 m/s. The new validation geometries are distinct from the prior protected experiments. They remain selection data, not the final test.

Select each seed's final model by highest primary compact validation fitness during the last 150 generations at the 1 m/s cap; ties keep the earliest candidate. Retention, geometric RMSE, straight speed and physical yaw remain visible outcomes. This selection does not silently claim that an old stability gate was passed.

Training can omit expensive independent geometric-RMSE computation because it is absent from the new reward. Detailed validation and assessment still measure it at the same control rate. Before any launch, replay identical controller/course cases with detailed measurement enabled and disabled and verify identical commands, fitness-relevant outcomes and termination. This is an instrumentation optimization, not a physics change.

After all five controllers are selected and frozen, generate new shared final tracks: 90 mixed (30 per difficulty), 60 speed/recovery and 30 corner. The resulting 180-course suite has the same group proportions as the six-course profile. Assess all five seeds, the accepted old prototype as a disclosed development control, and the saved full-base PID tunes at caps 1.0 and 0.5 m/s. Do not select the best seed using test results. Report per-seed completion, failures, paired successful-track time/error/jitter with denominators, physical speed/yaw and variability across seeds.

## Preparation and training commands

These become usable only after the pilot decision is saved. It must explicitly declare `pilot_study: "neat-simple-pilot-50-v1"`, the selected coefficient and rationale, and `pilot_protocol_sha256`, `pilot_report_sha256` and `pilot_snapshot_sha256`. There is no automatic study selection or fallback to the earlier 20-generation pilot.

Preparation checks all three terminal generation-50 checkpoints against their completion records and protocol signatures. The paired report must identify the same protocol, show all three arms complete at 50, and match the decision's report hash. Every archived snapshot payload is checked against its manifest, its terminal checkpoint/completion bytes must match the study, and `analysis.py` must match its own archived analysis source. Historical frozen source is checked against its own protocol, without requiring equality to today's workspace source. Report, snapshot manifest and control copies all come from the explicitly identified 50-generation study.

The trainer, preflight and final-assessment launchers are implemented, but no full preparation, real 48-episode preflight or independent training has been run. This compatibility change does not select a coefficient or start confirmation.

```powershell
$py = '.\.venv\Scripts\python.exe'

# Decision path will contain the documented completed-pilot choice.
& $py run_neat_simple_confirm.py prepare --decision research/simple-neat-development-decision.json
& $py run_neat_simple_confirm.py verify
& $py run_neat_simple_confirm.py preflight --workers 6

# Start one independent seed; six workers by default.
& $py run_neat_simple_confirm.py train --seed 1 --workers 6

# Continue only from its committed population/RNG checkpoint.
& $py run_neat_simple_confirm.py train --seed 1 --workers 6 --resume

& $py run_neat_simple_confirm.py status

# Refused until all five seeds finish; freezes controllers before new tests.
& $py run_neat_simple_confirm.py assess --workers 6
& $py run_neat_simple_confirm.py plot
```

Preserve protocol/source snapshots, five initial populations/RNG states, complete course batches and validation suites under `runs/neat-simple-confirm-v1/prepared/`. All runtime worker changes must stay within the requested six-to-eight range and be recorded. The goal remains active until independent evolution, fresh shared assessment, reproducibility checks and the methods/results handoff are actually completed.

`plot` can show committed validation learning curves before assessment, clearly marked as interim. After the full final assessment it also writes per-seed comparisons, failures, paired denominators and SVG/PNG figures under the run's `report/`. The final `methods-reference.tex` is reference material for collaborative writing; it is not inserted into the manuscript automatically.
