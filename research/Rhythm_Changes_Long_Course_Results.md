# Rhythm Changes: frozen NEAT and PID long-course case study

Date: 2026-11-21. Chosen course: candidate 4. All controller choices were frozen before its first evaluation.

## Findings to carry into the paper

**At the shared 1.00 m/s cap, all five preselected 1 m/s NEAT champions and all five frozen PID configurations completed the course.** The NEAT lap times were 60.62–62.58 s (mean 61.99 s, sample SD across controllers 0.85 s). They were 12.00–14.76% shorter than the full-speed PID's 71.12 s, and 5.58–8.55% shorter than the fastest saved PID configuration on this course at that same cap, the original 0.50 tune applied at 1.00 m/s (66.28 s). These are five model comparisons on one shared track, not five independent test tracks. The fastest-tune designation is observed after testing and does not replace the predefined full-speed PID baseline.

**The time advantage comes mainly from curves and turn combinations.** Relative to the full-speed PID, all five 1 m/s NEAT champions took longer on the opening straight (2–15%; mean +8.21%). Their mean section times were 24.70% shorter on the flowing S, 34.93% shorter on the tighter curved approach, 15.51% shorter on the wide chicane, and 12.88% shorter on the dense final chicane. All 19 sections have five shared successful comparisons; there are 95 paired section cells. These are descriptive section differences, not independent statistical replicates. Figure 05 shows every pair, and figure 06 shows the measured speed, error and jitter profiles.

**NEAT did not dominate tracking and smoothness.** The five 1 m/s champions had geometric RMSE of 17.58–21.05 mm versus 20.78 mm for the full-speed PID: four improved, while seed 5 was slightly worse. Each had more command jitter (0.004216–0.013563 versus 0.001144). The fastest saved PID tune at this cap also had lower error and jitter than all five: 13.65 mm and 0.000948. The smoothness PID had the lowest jitter among the five saved PIDs (0.000439), while the aggressive 1.50 tune, transferred to a 1.00 cap, achieved the lowest geometric RMSE (5.76 mm) but the slowest PID lap (77.53 s) and the highest jitter (0.035556). A smaller geometric tracking error is therefore not interchangeable with smoother wheel commands, and faster NEAT does not establish an overall stability victory.

**The higher-cap checkpoints were not the fastest NEAT controllers on this course.** All five native-cap champions completed at 0.50, 0.65, 0.80 and 1.00 m/s; four of five completed at 1.25, and three of five phase-selected champions completed at 1.50. Every successful native 1.25/1.50 NEAT lap was slower than every native 1.00 NEAT lap. This is an observation about the resulting selected controllers after different curriculum phases, not a causal estimate of the cap or extra generations. The separate formal seed-3 g216 winner completed at 1.50 in 88.10 s with 3.02 mm error and 0.000647 jitter, illustrating its conservative behavior. Its native mean/peak chassis speeds were 0.518/0.535 m/s; its drive output is not evaluated under the retained stock NEAT-Python behavior. The phase-selected seed-3 g294 drive output is also unevaluated; the other four phase-selected 1.50 controllers have evaluated drive outputs.

**PID produced the fastest successful lap anywhere in this grid.** The older 1.00 m/s tune, tested at the 1.25 m/s wheel-speed cap, completed in 56.32 s (mean/peak chassis speed 0.839/0.938 m/s, geometric RMSE 28.43 mm, jitter 0.003177). Under that same 1.25 cap, four of five phase-selected NEAT champions completed in 63.28–71.08 s with lower error of 14.64–23.32 mm. The NEAT laps were 12.37–26.22% longer than that PID lap. This PID result is a useful transfer finding, not a newly tuned baseline, and the lower error versus higher pace is an explicit trade-off. The full-speed PID failed at this cap, reinforcing that the PID settings matter. Do not compare the 56.32 s PID lap directly with the 1.00-cap NEAT laps as an equal-cap race.

**At 1.50 m/s, none of the five saved PID tunes completed.** Four failed in the first wide chicane at 28.1–28.5% route progress. The saved 1.50 tune reached the late four-turn chicane before stalling at 83.7% (38.31 m). Three phase-selected NEAT champions finished in 64.87, 71.93 and 86.47 s; seed 4 stalled in the first chicane at 31.0%, and seed 5 reached the dense final chicane at 92.8%. The additional formal g216 control also completed. Thus the result is three of five independent phase-selected NEAT controllers plus one extra controller from seed 3, not four of six independent seeds. These completion counts concern a single fixed geometry.

The successful phase-selected NEAT controllers at the 1.50 cap reached peak chassis speeds of only 0.840, 0.811 and 0.570 m/s. The full-speed PID reached 1.506 m/s transiently before failing. Both controller families receive the same wheel-speed cap, but choose different wheel commands and achieve different physical speeds; the 1.50-cap completion result therefore does not demonstrate reliable NEAT travel at 1.50 m/s.

**A reported stall is not a stopped motor.** The native-cap NEAT failures still traveled 2.84–3.35 m during their last five seconds but ended near earlier route segments than their authenticated progress. The 1.50 PID tune traveled 1.75 m in that interval with sensor contact throughout; its final closest route position was 36.67 m, behind its authenticated 38.31 m. These indicate failure to continue the ordered course, even while a painted line is detected. The smooth 1.00 PID transferred to 1.25 m/s ended near route distance 19.23 m despite authenticated progress of only 15.45 m: it reacquired a later route section without validating the intervening route. Preserve both the ordered-progress result and the physical trajectory. Do not describe such failures as the chassis coming to a standstill. `failure-diagnostics.csv` records final nearest-route position, last-five-second path length and sensor-contact fraction for all 25 incomplete episodes.

**Raising an early checkpoint's cap does not reliably reproduce the learned 1 m/s policy.** At the common 1.00 cap, all five low-speed parents and all five 0.65-phase champions failed, and only one of five 0.80-phase champions finished. The seed-3 g97 0.80-phase model was the fastest NEAT finisher observed at the 1.00 evaluation cap (60.02 s), although it had a higher 23.35 mm geometric RMSE. This is an interesting post-test transfer outlier; the four other 0.80 checkpoints failed, so it cannot justify a general claim that the earlier phase is better. In contrast, all five validation-selected 1.00-phase champions completed here.

**The cap still differs from achieved straight speed.** On the opening straight's central 2 m, the five 1 m/s NEAT champions averaged 0.858–0.972 m/s chassis speed; the full-speed PID averaged 0.991 m/s. On the central 2 m of the approximately 4 m row-2 back straight, NEAT averaged only 0.807–0.876 m/s and PID 0.924 m/s. Therefore none of those NEAT champions met the 0.95 m/s target on both eligible straights of this course. Seed 1's good isolated-straight performance does not imply it always accelerates back to that speed after a chicane. Figure 08 reports these measurements explicitly; no inference from the nominal cap is needed.

The course supports a narrower conclusion: the selected 1 m/s NEAT models repeatedly handled its bends quickly, whereas the saved PID controllers offer several different speed/error/jitter trade-offs, including the fastest lap at 1.25 m/s. Use this case study to explain behavior alongside the existing multi-track assessment. It does not establish universal superiority of either controller family or prove that extra ramp generations caused the later conservative behavior.


## Scope and reproducibility

This is a descriptive case study on one deterministic 45.7739 m course, not a new generalization benchmark. It has 20 sharp turns, three 0.8 m-radius turnarounds, broad/precision S-curves and late chicanes, a 20 mm line and 2 mm floor sampling in an 11 × 11 m arena. It contains no branches or gaps. The primary comparison uses a common 1.00 m/s wheel-speed cap.

Thirty NEAT policies comprise five frozen 0.50 m/s parents and the validation-best champion from each of five fast phases in each seed. The additional seed-3 g216 policy is the formal quality-gated final winner. Phase winners maximize validation completion, then fitness, with earliest-generation ties. Later phases inherit earlier caps; they are not fixed-speed-only training controls. All five distinct saved filtered PID configurations are retained, including the full-speed base-1.00 tune; no gains or genomes were changed.

Each NEAT policy is tested at its native cap and at 1.00 m/s (deduplicated); each PID configuration is tested at all six caps. This produces 36 frozen controllers and 87 episodes. A tune used away from its original cap is a transfer test. The same robot and MuJoCo 4.0 / NEAT-Python 0.92 semantics are used throughout. No outputs are patched, including disconnected drive outputs.

The timeout is fixed in advance at `180 / min(cap, 1.0)` seconds, so 180 seconds at caps ≥1 m/s and 360 seconds at 0.5. This accommodates long-course low-speed travel. The 100 mm corridor half-width, two seconds outside it with no sensor contact, and five-second ordered-progress stall check are unchanged. A stalled result means insufficient authenticated route progress, not necessarily a stationary chassis. All policies at a given cap share the same episode rules. Controllers receive sensors/time only.

## Native-cap NEAT results

| Seed | 0.50 | 0.65 | 0.80 | 1.00 | 1.25 | 1.50 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 100.23 s | 78.62 s | 67.35 s | 61.70 s | 65.25 s | 64.87 s |
| 2 | 100.75 s | 78.62 s | 67.35 s | 62.50 s | 65.70 s | 71.93 s |
| 3 | 99.47 s | 78.75 s | 68.07 s | 62.58 s | 71.08 s | 86.47 s |
| 4 | 99.78 s | 78.35 s | 66.82 s | 62.53 s | 61.5% (stalled) | 31.0% (stalled) |
| 5 | 99.37 s | 78.72 s | 66.98 s | 60.62 s | 63.28 s | 92.8% (stalled) |

Cells show finish time only when completed. Otherwise they show route progress and the termination reason.

## Primary 1 m/s phase champions versus full-speed PID

The five 1 m/s phase champions were chosen from the existing validation logs, rather than from this course. The PID's base=1.00 permits a centered straight target of 1 m/s. Other PID tunes with base=.75 have a centered straight target of .75 × cap; the measured speeds below make this distinction explicit.

| Controller | Finished | Progress | Lap time (s) | Mean / peak chassis speed (m/s) | Geometric RMSE (mm) | Command jitter |
|---|---|---:|---:|---:|---:|---:|
| S1 g134 · trained 1.00 | yes | 100.0% | 61.70 | 0.759 / 0.983 | 17.58 | 0.005758 |
| S2 g136 · trained 1.00 | yes | 100.0% | 62.50 | 0.749 / 0.865 | 18.00 | 0.013563 |
| S3 g131 · trained 1.00 | yes | 100.0% | 62.58 | 0.749 / 0.924 | 17.92 | 0.013268 |
| S4 g107 · trained 1.00 | yes | 100.0% | 62.53 | 0.749 / 0.916 | 17.74 | 0.004216 |
| S5 g134 · trained 1.00 | yes | 100.0% | 60.62 | 0.777 / 0.933 | 21.05 | 0.004575 |
| PID 0.50 tune (base .75) | yes | 100.0% | 66.28 | 0.699 / 0.748 | 13.65 | 0.000948 |
| PID original 1.00 (base .75) | yes | 100.0% | 68.30 | 0.680 / 0.748 | 21.54 | 0.002433 |
| PID smooth 1.00 (base .75) | yes | 100.0% | 67.62 | 0.687 / 0.748 | 18.92 | 0.000439 |
| PID full-speed 1.00 (base 1.00) | yes | 100.0% | 71.12 | 0.657 / 1.003 | 20.78 | 0.001144 |
| PID 1.50 tune (base .75) | yes | 100.0% | 77.53 | 0.590 / 0.748 | 5.76 | 0.035556 |

A failed episode has no lap time; its elapsed duration is retained separately in the complete table. Means/RMSE/jitter from partial episodes cover different distances and must not be read as equal-exposure full-lap quality.

## Frozen models and raw evidence

All copied genomes and PID JSON files are in `runs/rhythm-changes-case-study-20261121/frozen/`. `frozen-controller-inventory.csv` identifies every source, generation, training cap, output-evaluation status, and SHA-256. `protocol.json` commits the policy selection, task grid, robot, episode rules and input/software fingerprints; `source-snapshot.zip` preserves evaluation source. The original training and earlier comparison artifacts remain unchanged.

`results.csv` contains the full headless results. `results-with-sections.csv` adds the final course section. Every episode has its own committed JSON and gzip CSV trace with 60 Hz sensors, actions and physical state. `section-metrics.csv` provides traversal times and observed quality by section. `paired-section-times-1m.csv` includes a row for each of 19 × 5 potential comparisons and marks unavailable pairs. `straight-windows.csv` records both measured chassis speed and 2 m / crossing-time speed. `verification.json` records trace-to-summary checks. Eight figures are saved in both PNG and SVG under `paper-figures/`, with captions and limitations in its README.

## Commands

From the project root in PowerShell:

```powershell
$py = ".\.venv\Scripts\python.exe"

# Verify the already frozen course and all controllers
& $py run_rhythm_changes_case_study.py verify

# Evaluate missing episodes only; completed results are checked and reused
& $py run_rhythm_changes_case_study.py run --workers 2

# Regenerate tables and figures, without simulation or training
& $py runs/rhythm-changes-case-study-20261121/report.py

# Replay the preselected seed-1 1 m/s champion on the chosen course
& $py run_rhythm_changes_case_study.py replay --episode neat-s1-fast-100__cap-100

# Replay the full-speed filtered PID under exactly the same 1 m/s cap
& $py run_rhythm_changes_case_study.py replay --episode pid-fullspeed-100__cap-100
```

For any other frozen episode, copy its `id` from results.csv into `--episode`. Replays print the actual controller class, source file and cap. Space pauses, R resets, T toggles the route, and B toggles chassis transparency. Replays do not overwrite the study results.

## Every evaluated episode

| Episode ID | Result | Progress | Lap / elapsed (s) | Mean / peak (m/s) | RMSE (mm) | Jitter | Final section |
|---|---|---:|---:|---:|---:|---:|---|
| neat-s1-low-050__cap-050 | completed | 100.0% | 100.23 / 100.23 | 0.453 / 0.495 | 5.89 | 0.001605 | Finish straight |
| neat-s2-low-050__cap-050 | completed | 100.0% | 100.75 / 100.75 | 0.451 / 0.492 | 5.59 | 0.001377 | Finish straight |
| neat-s3-low-050__cap-050 | completed | 100.0% | 99.47 / 99.47 | 0.455 / 0.495 | 3.70 | 0.001583 | Finish straight |
| neat-s4-low-050__cap-050 | completed | 100.0% | 99.78 / 99.78 | 0.454 / 0.495 | 6.25 | 0.001431 | Finish straight |
| neat-s5-low-050__cap-050 | completed | 100.0% | 99.37 / 99.37 | 0.456 / 0.496 | 6.68 | 0.002567 | Finish straight |
| pid-followup-100__cap-050 | completed | 100.0% | 135.97 / 135.97 | 0.340 / 0.372 | 18.19 | 0.001911 | Finish straight |
| pid-followup-150__cap-050 | completed | 100.0% | 129.17 / 129.17 | 0.348 / 0.372 | 5.53 | 0.017746 | Finish straight |
| pid-fullspeed-100__cap-050 | completed | 100.0% | 139.77 / 139.77 | 0.328 / 0.496 | 12.18 | 0.000781 | Finish straight |
| pid-low-050__cap-050 | completed | 100.0% | 129.47 / 129.47 | 0.353 / 0.372 | 8.18 | 0.000615 | Finish straight |
| pid-smooth-100__cap-050 | completed | 100.0% | 135.20 / 135.20 | 0.341 / 0.372 | 16.87 | 0.000260 | Finish straight |
| neat-s1-fast-065__cap-065 | completed | 100.0% | 78.62 / 78.62 | 0.582 / 0.645 | 4.02 | 0.001386 | Finish straight |
| neat-s2-fast-065__cap-065 | completed | 100.0% | 78.62 / 78.62 | 0.581 / 0.645 | 1.98 | 0.001596 | Finish straight |
| neat-s3-fast-065__cap-065 | completed | 100.0% | 78.75 / 78.75 | 0.580 / 0.636 | 2.16 | 0.002039 | Finish straight |
| neat-s4-fast-065__cap-065 | completed | 100.0% | 78.35 / 78.35 | 0.584 / 0.645 | 4.93 | 0.001836 | Finish straight |
| neat-s5-fast-065__cap-065 | completed | 100.0% | 78.72 / 78.72 | 0.581 / 0.645 | 3.15 | 0.001979 | Finish straight |
| pid-followup-100__cap-065 | completed | 100.0% | 104.65 / 104.65 | 0.441 / 0.483 | 18.70 | 0.002009 | Finish straight |
| pid-followup-150__cap-065 | completed | 100.0% | 101.77 / 101.77 | 0.444 / 0.483 | 4.78 | 0.022652 | Finish straight |
| pid-fullspeed-100__cap-065 | completed | 100.0% | 107.83 / 107.83 | 0.427 / 0.645 | 12.97 | 0.000791 | Finish straight |
| pid-low-050__cap-065 | completed | 100.0% | 99.93 / 99.93 | 0.458 / 0.483 | 8.61 | 0.000653 | Finish straight |
| pid-smooth-100__cap-065 | completed | 100.0% | 104.00 / 104.00 | 0.444 / 0.483 | 17.02 | 0.000299 | Finish straight |
| neat-s1-fast-080__cap-080 | completed | 100.0% | 67.35 / 67.35 | 0.686 / 0.796 | 11.15 | 0.002005 | Finish straight |
| neat-s2-fast-080__cap-080 | completed | 100.0% | 67.35 / 67.35 | 0.689 / 0.798 | 13.02 | 0.003175 | Finish straight |
| neat-s3-fast-080__cap-080 | completed | 100.0% | 68.07 / 68.07 | 0.678 / 0.796 | 9.51 | 0.002816 | Finish straight |
| neat-s4-fast-080__cap-080 | completed | 100.0% | 66.82 / 66.82 | 0.692 / 0.798 | 11.71 | 0.002220 | Finish straight |
| neat-s5-fast-080__cap-080 | completed | 100.0% | 66.98 / 66.98 | 0.691 / 0.798 | 12.16 | 0.001642 | Finish straight |
| pid-followup-100__cap-080 | completed | 100.0% | 85.10 / 85.10 | 0.543 / 0.594 | 19.33 | 0.002055 | Finish straight |
| pid-followup-150__cap-080 | completed | 100.0% | 86.42 / 86.42 | 0.526 / 0.594 | 2.84 | 0.025626 | Finish straight |
| pid-fullspeed-100__cap-080 | completed | 100.0% | 87.55 / 87.55 | 0.528 / 0.798 | 14.82 | 0.000845 | Finish straight |
| pid-low-050__cap-080 | completed | 100.0% | 81.55 / 81.55 | 0.563 / 0.594 | 9.29 | 0.000705 | Finish straight |
| pid-smooth-100__cap-080 | completed | 100.0% | 84.45 / 84.45 | 0.548 / 0.594 | 17.43 | 0.000321 | Finish straight |
| neat-s1-fast-065__cap-100 | stalled | 30.8% | — / 21.30 | 0.784 / 1.002 | 34.63 | 0.004461 | Wide six-turn chicane |
| neat-s1-fast-080__cap-100 | stalled | 33.6% | — / 32.58 | 0.768 / 1.000 | 29.14 | 0.004719 | Wide six-turn chicane |
| neat-s1-fast-100__cap-100 | completed | 100.0% | 61.70 / 61.70 | 0.759 / 0.983 | 17.58 | 0.005758 | Finish straight |
| neat-s1-fast-125__cap-100 | completed | 100.0% | 69.72 / 69.72 | 0.663 / 0.751 | 10.03 | 0.014185 | Finish straight |
| neat-s1-fast-150__cap-100 | completed | 100.0% | 84.57 / 84.57 | 0.539 / 0.574 | 3.07 | 0.002410 | Finish straight |
| neat-s1-low-050__cap-100 | stalled | 30.9% | — / 21.87 | 0.809 / 1.003 | 28.24 | 0.005785 | Wide six-turn chicane |
| neat-s2-fast-065__cap-100 | stalled | 30.9% | — / 21.47 | 0.819 / 1.002 | 26.24 | 0.004402 | Wide six-turn chicane |
| neat-s2-fast-080__cap-100 | stalled | 28.4% | — / 19.43 | 0.876 / 1.002 | 15.51 | 0.003511 | Wide six-turn chicane |
| neat-s2-fast-100__cap-100 | completed | 100.0% | 62.50 / 62.50 | 0.749 / 0.865 | 18.00 | 0.013563 | Finish straight |
| neat-s2-fast-125__cap-100 | completed | 100.0% | 73.77 / 73.77 | 0.626 / 0.724 | 8.29 | 0.031287 | Finish straight |
| neat-s2-fast-150__cap-100 | completed | 100.0% | 107.52 / 107.52 | 0.422 / 0.530 | 2.12 | 0.011288 | Finish straight |
| neat-s2-low-050__cap-100 | stalled | 28.4% | — / 19.95 | 0.860 / 0.995 | 27.85 | 0.003050 | Wide six-turn chicane |
| neat-s3-fast-065__cap-100 | stalled | 28.3% | — / 19.52 | 0.832 / 0.988 | 31.60 | 0.004309 | Wide six-turn chicane |
| neat-s3-fast-080__cap-100 | completed | 100.0% | 60.02 / 60.02 | 0.788 / 1.000 | 23.35 | 0.004532 | Finish straight |
| neat-s3-fast-100__cap-100 | completed | 100.0% | 62.58 / 62.58 | 0.749 / 0.924 | 17.92 | 0.013268 | Finish straight |
| neat-s3-fast-125__cap-100 | completed | 100.0% | 80.32 / 80.32 | 0.574 / 0.646 | 6.40 | 0.021653 | Finish straight |
| neat-s3-fast-150__cap-100 | completed | 100.0% | 129.72 / 129.72 | 0.348 / 0.368 | 3.60 | 0.004921 | Finish straight |
| neat-s3-formal-150__cap-100 | completed | 100.0% | 130.45 / 130.45 | 0.346 / 0.360 | 2.97 | 0.000404 | Finish straight |
| neat-s3-low-050__cap-100 | stalled | 30.8% | — / 21.48 | 0.805 / 1.003 | 32.60 | 0.006063 | Wide six-turn chicane |
| neat-s4-fast-065__cap-100 | stalled | 30.9% | — / 25.57 | 0.788 / 1.003 | 32.14 | 0.007742 | Wide six-turn chicane |
| neat-s4-fast-080__cap-100 | stalled | 30.9% | — / 21.28 | 0.831 / 1.003 | 27.19 | 0.004050 | Wide six-turn chicane |
| neat-s4-fast-100__cap-100 | completed | 100.0% | 62.53 / 62.53 | 0.749 / 0.916 | 17.74 | 0.004216 | Finish straight |
| neat-s4-fast-125__cap-100 | completed | 100.0% | 70.30 / 70.30 | 0.657 / 0.742 | 10.10 | 0.012627 | Finish straight |
| neat-s4-fast-150__cap-100 | completed | 100.0% | 103.22 / 103.22 | 0.452 / 0.587 | 8.18 | 0.043691 | Finish straight |
| neat-s4-low-050__cap-100 | stalled | 61.3% | — / 40.57 | 0.803 / 1.003 | 19.53 | 0.006117 | Right-angle transition |
| neat-s5-fast-065__cap-100 | stalled | 30.8% | — / 21.12 | 0.823 / 1.003 | 33.19 | 0.005098 | Wide six-turn chicane |
| neat-s5-fast-080__cap-100 | stalled | 31.0% | — / 21.32 | 0.843 / 1.003 | 19.72 | 0.002235 | Wide six-turn chicane |
| neat-s5-fast-100__cap-100 | completed | 100.0% | 60.62 / 60.62 | 0.777 / 0.933 | 21.05 | 0.004575 | Finish straight |
| neat-s5-fast-125__cap-100 | completed | 100.0% | 72.53 / 72.53 | 0.633 / 0.667 | 6.66 | 0.002998 | Finish straight |
| neat-s5-fast-150__cap-100 | completed | 100.0% | 93.80 / 93.80 | 0.484 / 0.532 | 6.11 | 0.017428 | Finish straight |
| neat-s5-low-050__cap-100 | stalled | 33.7% | — / 24.02 | 0.789 / 1.003 | 29.18 | 0.007133 | Wide six-turn chicane |
| pid-followup-100__cap-100 | completed | 100.0% | 68.30 / 68.30 | 0.680 / 0.748 | 21.54 | 0.002433 | Finish straight |
| pid-followup-150__cap-100 | completed | 100.0% | 77.53 / 77.53 | 0.590 / 0.748 | 5.76 | 0.035556 | Finish straight |
| pid-fullspeed-100__cap-100 | completed | 100.0% | 71.12 / 71.12 | 0.657 / 1.003 | 20.78 | 0.001144 | Finish straight |
| pid-low-050__cap-100 | completed | 100.0% | 66.28 / 66.28 | 0.699 / 0.748 | 13.65 | 0.000948 | Finish straight |
| pid-smooth-100__cap-100 | completed | 100.0% | 67.62 / 67.62 | 0.687 / 0.748 | 18.92 | 0.000439 | Finish straight |
| neat-s1-fast-125__cap-125 | completed | 100.0% | 65.25 / 65.25 | 0.730 / 0.907 | 20.82 | 0.013924 | Finish straight |
| neat-s2-fast-125__cap-125 | completed | 100.0% | 65.70 / 65.70 | 0.731 / 0.901 | 23.32 | 0.023893 | Finish straight |
| neat-s3-fast-125__cap-125 | completed | 100.0% | 71.08 / 71.08 | 0.665 / 0.795 | 14.64 | 0.017300 | Finish straight |
| neat-s4-fast-125__cap-125 | stalled | 61.5% | — / 44.32 | 0.724 / 0.890 | 14.70 | 0.010810 | Right-angle transition |
| neat-s5-fast-125__cap-125 | completed | 100.0% | 63.28 / 63.28 | 0.747 / 0.837 | 19.61 | 0.005015 | Finish straight |
| pid-followup-100__cap-125 | completed | 100.0% | 56.32 / 56.32 | 0.839 / 0.938 | 28.43 | 0.003177 | Finish straight |
| pid-followup-150__cap-125 | completed | 100.0% | 91.88 / 91.88 | 0.503 / 0.938 | 6.17 | 0.048038 | Finish straight |
| pid-fullspeed-100__cap-125 | off_route | 30.9% | — / 20.83 | 0.787 / 1.257 | 53.21 | 0.001525 | Wide six-turn chicane |
| pid-low-050__cap-125 | off_route | 36.2% | — / 22.97 | 0.802 / 0.938 | 86.28 | 0.001839 | Wide six-turn chicane |
| pid-smooth-100__cap-125 | stalled | 33.8% | — / 23.98 | 0.839 / 0.938 | 32.26 | 0.000730 | Wide six-turn chicane |
| neat-s1-fast-150__cap-150 | completed | 100.0% | 64.87 / 64.87 | 0.728 / 0.840 | 17.16 | 0.007995 | Finish straight |
| neat-s2-fast-150__cap-150 | completed | 100.0% | 71.93 / 71.93 | 0.638 / 0.811 | 6.77 | 0.013524 | Finish straight |
| neat-s3-fast-150__cap-150 | completed | 100.0% | 86.47 / 86.47 | 0.527 / 0.570 | 3.62 | 0.004983 | Finish straight |
| neat-s3-formal-150__cap-150 | completed | 100.0% | 88.10 / 88.10 | 0.518 / 0.535 | 3.02 | 0.000647 | Finish straight |
| neat-s4-fast-150__cap-150 | stalled | 31.0% | — / 29.10 | 0.620 / 0.765 | 22.79 | 0.028117 | Wide six-turn chicane |
| neat-s5-fast-150__cap-150 | stalled | 92.8% | — / 70.42 | 0.663 / 0.804 | 21.23 | 0.020875 | Dense final chicane |
| pid-followup-100__cap-150 | off_route | 28.1% | — / 14.40 | 0.928 / 1.128 | 69.22 | 0.001921 | Wide six-turn chicane |
| pid-followup-150__cap-150 | stalled | 83.7% | — / 102.38 | 0.406 / 1.128 | 9.20 | 0.046267 | Four-turn chicane |
| pid-fullspeed-100__cap-150 | stalled | 28.4% | — / 17.88 | 0.981 / 1.506 | 31.64 | 0.001573 | Wide six-turn chicane |
| pid-low-050__cap-150 | stalled | 28.5% | — / 17.65 | 1.033 / 1.128 | 24.66 | 0.001649 | Wide six-turn chicane |
| pid-smooth-100__cap-150 | off_route | 28.3% | — / 14.92 | 0.908 / 1.128 | 68.66 | 0.000567 | Wide six-turn chicane |

## Interpretation limits

The fastest finisher or best tune on this course is an observed course-specific outcome, not a newly validated winner. Do not silently replace the frozen champions or retune on this track and call the result held out. Completion counts across seeds here describe five different controllers on one geometry; they are not an estimate of completion probability across tracks. Use the earlier shared multi-track assessment for generalization claims. Isolating training-cap or curriculum effects would require matched-budget independent controls.
