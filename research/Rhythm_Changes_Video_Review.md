# Review of the six Rhythm Changes videos

Reviewed 23 November 2026 after the user identified limited PID straight speed and NEAT steering jitter. Relevant opening-straight, turning and post-turn straight sections of all six MP4 clips were inspected in VLC, alongside their original 60 Hz telemetry. This is a diagnostic addendum; frozen controllers, physics and assessment results have not been changed.

## PID selection error and speed feasibility

The shortlist selected fastest successful laps on this course, or greatest progress for incomplete runs. That did **not** satisfy the intended request for the appropriate cap-specific, full-speed PID baselines. In particular, it replaced the previously established full-speed 1 m/s PID with an older base-0.75 configuration. It also used a 1.50-tuned configuration at 0.50 m/s simply because that configuration completed this particular course fastest among the saved PIDs.

All three selected PID clips use `base = 0.75`. With centered sensors and settled history, the actual controller returns `[0.75, 0.75]`. Its straight wheel-speed target is therefore `0.75 × evaluation cap`:

| Video evaluation cap | Nominal straight target | Measured central opening speed |
|---|---:|---:|
| 0.50 m/s | 0.375 m/s | 0.371 m/s |
| 1.00 m/s | 0.750 m/s | 0.743 m/s |
| 1.50 m/s | 1.125 m/s | 1.114 m/s |

These values explain the speed plateaus in the videos. The cap labels are the simulator's real wheel-speed caps, but the selected controllers do not command those full speeds on a centered straight. The target is not a hard bound on every transient chassis velocity, but it precludes sustained full-cap centered driving in this setup.

The correct existing full-speed 1 m/s configuration remains at `runs/rhythm-changes-case-study-20261121/frozen/pid-fullspeed-100.json` (`base = 1.0`, SHA-256 `c6c68b53342600db0078e991bbb729064d829d360c4840a720dbf5e66d94c24b`). It commands `[1, 1]` for centered sensors and measured 0.991 m/s over the central opening straight. It completed this course in 71.12 s under the 1 m/s cap. Its prior multi-track assessment remains separate from the incorrectly selected video example.

The saved 1.50 PID is an actual previously tuned configuration, but it was tuned with base 0.75 and is not a satisfactory full-speed 1.50 baseline. On this course's central row-2 back straight it had normalized steering RMS 0.899 and physical yaw RMS 6.024 rad/s, with wheel commands reaching opposing saturation. Calling it a well-tuned full-speed controller would be unsupported. Simply raising base to 1.0 is not sufficient evidence of good tuning; a new full-speed tuning/validation run is required, retaining the old results as historical evidence.

## NEAT steering oscillation is real

Opening straights are steady for all three selected NEAT models: wheel commands are equal and recorded yaw is zero in the central opening window. However, later straights following turns expose oscillations, especially in the 1.00 and 1.50-cap clips. Thus the issue is not uniform over every straight or equally severe at every speed.

The following measurements use the same geometric section (row-2 back straight), excluding 0.8 m at either end. All original 60 Hz samples are included, with no smoothing. Steering is `(left_command - right_command)/2`; reversals count sign changes after excluding steering magnitudes ≤0.02. Durations differ because achieved speeds differ.

| Controller / evaluation cap | Video window | Mean chassis speed | Steering reversals | Physical yaw RMS | Geometric RMSE |
|---|---|---:|---:|---:|---:|
| NEAT seed 5 low g91 / 0.50 | 38.48–43.35 s | 0.493 m/s | 2 | 0.096 rad/s | 6.20 mm |
| NEAT seed 3 g97, trained at 0.80 / 1.00 | 22.30–24.90 s | 0.920 m/s | 20 | 0.683 rad/s | 0.90 mm |
| NEAT seed 1 g262 / 1.50 | 24.82–27.72 s | 0.824 m/s | 25 | 0.683 rad/s | 1.89 mm |
| Existing full-speed PID / 1.00 | 27.02–29.58 s | 0.927 m/s | 1 | 0.100 rad/s | 1.94 mm |

The 1 m/s NEAT example stays very close to the line while repeatedly reversing steering. At nearly the same measured straight speed as the existing full-speed PID, it has approximately 6.8 times the yaw RMS in this window. This confirms physical oscillation, not solely 15 fps playback or camera artifacts. The low-speed model instead makes a few recovery corrections and then settles for most of the window.

Low geometric RMSE does not imply low yaw motion or smooth commands. Successful completion and short lap time do not establish a smoothness advantage. The fastest-lap shortlist was not selected through the formal stability gate; the 1 m/s clip is even a transferred 0.80-phase champion. It should not be presented as the best balanced, speed-specific validated controller.

The current fitness retains a command-jitter penalty, but its existence alone does not establish that the resulting controller is smooth. Future selection needs explicit straight and post-turn settling checks, including yaw/steering oscillation, alongside completion, speed and geometric error. This audit identifies the behavior; it does not prove which neural feature or mutation caused the oscillation.

## Consequences and next work

1. Restore the existing base-1.0, validated 1 m/s PID as the primary 1 m/s baseline. Select controllers by the declared validation rule and tuning cap rather than fastest observed lap on this test course.
2. Establish a base-1.0 PID tune at 1.50 m/s, and review the 0.50 baseline's speed requirement too. Tune on training/validation courses; keep this already observed course as an exploratory diagnostic, not a fresh held-out selection set.
3. Add explicit straight and post-turn oscillation/settling criteria when selecting or retraining NEAT. Do not claim the current fast examples meet the original smooth-driving goal.
4. Freeze the revised controllers and verify measured sustained straight speed before replacing the six showcase videos. New PID tunes, neural training and replacement video rendering have not been started by this review.

The gallery is marked as a historical shortlist requiring correction. The original MP4 files and all assessment evidence remain preserved.

## Reproduce the diagnostic

```powershell
& .\.venv\Scripts\python.exe research/review_rhythm_video.py
# Expected to exit 1 for the historical shortlist: all three selected PIDs have base .75
& .\.venv\Scripts\python.exe research/review_rhythm_video.py --require-full-speed-pids
```

Evidence: `runs/rhythm-changes-case-study-20261121/video-review-20261123/audit.json`, `straight-metrics.csv`, and the PNG/SVG `straight-command-and-yaw-oscillation` figures. The diagnostic checks actual controller outputs for centered sensors, exact geometric tracking error and full-rate wheel/yaw traces. No training or evaluation source was edited.
