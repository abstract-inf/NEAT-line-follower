# Filtered PID refinement at the 1.00 m/s cap

Recorded 19 November 2026. This refinement addresses the user's observation of jitter on gentle curves. It reuses the filtered PID with remembered-side recovery and the unchanged robot and episode loop. No NEAT evolution was performed.

## Result

The selected tune completed **149/150 fresh PID-only test courses**, compared with **143/150** for the original 1.00 m/s tune. It reduced mean command jitter by **79.4%** and measured steady-section yaw oscillation by **51.2%**. Geometric RMSE also improved on shared successful courses.

On the **143 shared finishes**, mean finish time changed from **5.960 s** to **5.994 s**, a **0.57% increase**. No course completed by the original tune was lost by the retune. One intermediate course stalled for both controllers.

## Saved parameters

| Parameter | Original | Retuned |
| --- | ---: | ---: |
| kp | 0.372062091 | 0.390665196 |
| ki | 0.087212318 | 0.087212318 |
| kd | 0.056109413 | 0.050498472 |
| derivative_filter_s | 0.019213047 | 0.080000000 |
| base | 0.750000000 | 0.750000000 |

The chosen combination increases P by 5%, decreases D by 10%, and extends the existing derivative-filter time constant from 19.2 ms to 80 ms. I, base command, slowdown, recovery parameters, and controller logic are unchanged. The wheel-speed cap remains 1.00 m/s. With base command 0.75, this does not force 1 m/s chassis travel; peak forward speed on the long course was approximately 0.748 m/s for both controllers.

The user's lower-P/higher-D proposal was tested as a hypothesis. On the original diagnostic curve, raising D alone by 5–10% increased yaw oscillation. Lowering P alone also failed to improve that diagnostic. A longer derivative filter gave a much larger improvement. This is consistent with excessive response to discrete sensor-error changes; it is evidence from the measured loop, rather than a universal claim about PID tuning.

## Search and selection

The deterministic probe uses a 2 m-radius curve and excludes entry/exit transitions. Physical yaw oscillation is the RMS residual after subtracting a centered 15-sample moving average from measured chassis yaw rate at 60 Hz. This separates rapid oscillation from the intended steady turn. The broader speed-course measurements average the annotated constant-radius and long-straight windows.

The local search evaluated 100 new combinations of P, D, and the existing derivative filter, plus the unchanged baseline. Candidates first faced 12 development cases; 13 policies advanced to 51 development courses, followed by 6 policies on 60 separate validation courses. Selection required:

- Retain every baseline-successful course and match or improve completion in each group.
- Mean paired finish time no more than 3% longer than baseline.
- Shared-finish geometric RMSE no more than 5% above baseline.
- Overall command jitter no higher than baseline.
- Steady-section yaw oscillation at least 20% lower than baseline.

These are explicitly declared engineering targets for this PID refinement. They do not modify the original NEAT study's stricter winner-selection criteria. Among eligible candidates, ranking favored completion, then lower yaw oscillation, lower command jitter, and shorter paired time.

Candidate **local-088** completed **60/60 validation courses** versus **59/60** for baseline and was frozen before opening the independent 150-course PID-only test split. The test suite contains 90 mixed, 30 speed, and 30 corner courses. Geometry is disjoint across these splits and excluded from the frozen NEAT study geometry. No NEAT final-test course was evaluated or used for tuning.

## Fresh test results

| Measure | Original | Retuned |
| --- | ---: | ---: |
| Completed, all courses | 143 | 149 |
| Mixed completed, out of 90 | 86 | 89 |
| Speed completed, out of 30 | 30 | 30 |
| Corner completed, out of 30 | 27 | 30 |
| Mean geometric RMSE, all episodes (mm) | 33.419 | 23.318 |
| Mean geometric RMSE, shared finishes (mm) | 24.845 | 22.742 |
| Mean command jitter | 0.004711 | 0.000969 |
| Steady-section yaw oscillation RMS (rad/s) | 0.065357 | 0.031893 |

The original tune's failures were four mixed-course stalls and three corner-course off-route terminations. The retune recovered six of those courses and retained one mixed-course stall. These are results from one deterministic simulation suite, not hardware measurements or a NEAT/PID comparison.

## Original diagnostic and exploratory long course

On the original 2 m-radius diagnostic curve, the retune reduced yaw oscillation from **0.245637** to **0.123123 rad/s** (49.9% lower). Command jitter fell by **80.9%**. Mean steady-section forward speed was **0.7136 → 0.7146 m/s**. The original reproducible smoothness check passed for the retune.

Both policies completed the existing exploratory long course. Finish time was **29.550 → 29.317 s**, while command jitter fell by **77.8%**. The long course was checked after selection and was not a selection criterion.

## Files and terminal commands

The new gains are [tuned-1.00.json](../runs/pid-100-smoothness-20261119/tuned-1.00.json). The original tune and all prepared NEAT inputs remain unchanged. The current frozen NEAT protocol therefore still references its original PID file. Any future NEAT comparison using this stronger tune must explicitly identify the new configuration and freeze it in that comparison's manifest.

Open PowerShell in the project directory:

```powershell
$py = ".\.venv\Scripts\python.exe"
$pid = "runs/pid-100-smoothness-20261119/tuned-1.00.json"

# Replay the gentle diagnostic curve with the retuned recovery PID
& $py run_pid_smoothness_tuning.py view --config $pid

# Replay our long course at the same 1.00 m/s cap
& $py run_pid_smoothness_tuning.py view --config $pid --track runs/long-course-speed-challenge/track.json

# Replay any saved validation track; change the path as desired
& $py run_pid_smoothness_tuning.py view --config $pid --track runs/neat-fast-study/prepared/suite/validation/002.json

# Recheck the deterministic smoothness target
& $py run_pid_smoothness_tuning.py probe --config $pid

# Reproduce the complete search in a separate directory
& $py run_pid_smoothness_tuning.py run --output runs/pid-100-smoothness-reproduction
```

The viewer starts with the supplied filtered recovery PID and the study's recorded episode rules: 100 mm corridor, two seconds outside without sensor contact, and the five-second stall check. Press R to reset, Space to pause, or T for the intended route. The generic viewer's C toggle selects its built-in controller rather than another saved research tune; compare saved tunes by launching them with their explicit configuration paths.

- [Gentle-curve physical and command traces, PNG](../runs/pid-100-smoothness-20261119/plots/01-gentle-curve-smoothness.png), [SVG](../runs/pid-100-smoothness-20261119/plots/01-gentle-curve-smoothness.svg).
- [Fresh-test comparison, PNG](../runs/pid-100-smoothness-20261119/plots/02-fresh-test-comparison.png), [SVG](../runs/pid-100-smoothness-20261119/plots/02-fresh-test-comparison.svg).
- [Protocol](../runs/pid-100-smoothness-20261119/protocol.json), [selection freeze](../runs/pid-100-smoothness-20261119/selection-freeze.json), and [raw test episodes](../runs/pid-100-smoothness-20261119/test-episodes.csv).
- [One-factor probes](../runs/pid-100-smoothness-20261119/one-factor-screen.csv), [validation summaries](../runs/pid-100-smoothness-20261119/validation-summary.csv), and [test summaries](../runs/pid-100-smoothness-20261119/test-summary.csv).
