# Smooth-fitness pilot: generation-44 progress snapshot

Checked 2026-11-24T12:26:06.455232+00:00. **45/100 generations completed at this snapshot; no qualifying champion yet.**

Latest champion: 60/60 primary validation completions and 20/20 retention completions. Straight yaw and steering acceptance passed, as in the preceding eleven generations. Remaining failures: straight speed, paired lap time and whole-course command jitter.

| Metric | Original parent | Generation 44 |
|---|---:|---:|
| Isolated-straight mean speed | 0.972 m/s | 0.697 m/s |
| Post-turn straight mean speed | 0.830 m/s | 0.705 m/s |
| Post-turn straight mean yaw RMS | 1.319 rad/s | 0.023 rad/s |
| Shared-finish geometric RMSE | 14.49 mm | 9.72 mm |
| Whole-course command jitter | 0.006530 | 0.007450 |

Paired laps were 12.30% slower, above the allowed 5%. All 20 intended primary straight windows were observed. Maximum post-turn yaw was 0.0381 rad/s (limit 0.15); maximum steering RMS was 0.00502 (limit 0.02). Mean straight speed remained about 0.70 m/s, below the required 0.95 m/s.

The emerging trade-off is lower physical straight oscillation and smaller geometric error with lower achieved speed. Straight steering smoothness does not establish lower command jitter over whole courses. The speed/smoothness combination remains unsolved in this partial run.

These are repeated validation observations from an unfinished one-seed pilot, not final test evidence or an approved selected winner. Continue the predeclared run without post-hoc changes to its fitness or thresholds. The source validation JSON hash is recorded in the adjacent snapshot JSON.


## Following checkpoint: generation 45

46/100 generations are now completed. The generation-45 champion completed 60/60 primary and 20/20 retention tracks. Straight yaw/steering gates passed for the thirteenth consecutive generation (33-45).

Isolated and post-turn mean straight speeds were approximately 0.750 and 0.751 m/s. Post-turn mean yaw RMS was 0.0257 rad/s, geometric RMSE was 8.49 mm, paired lap time was 8.98% slower, and whole-course command jitter was 0.007195. Remaining rejections were speed, time and jitter; no champion qualified.
