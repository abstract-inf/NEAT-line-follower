# Top-down NEAT and PID replays

Each MP4 shows the same course and clock in three panels. From left to right, the panels show frozen NEAT g98, filtered PID with remembered-direction recovery, and plain PID. Both sets use the same five tracks, controllers, and per-track camera framing. The values 1.50 and 0.50 m/s are wheel speed caps. The speed shown in each panel is the measured forward chassis speed.

| Track | Course features | 1.50 m/s cap | 0.50 m/s cap |
| --- | --- | --- | --- |
| 000 · Beginner | Gentle curve and straight | [Watch](01-1.50-mps/track-000-beginner.mp4) | [Watch](02-0.50-mps/track-000-beginner.mp4) |
| 001 · Intermediate | Corner and S-curve | [Watch](01-1.50-mps/track-001-intermediate.mp4) | [Watch](02-0.50-mps/track-001-intermediate.mp4) |
| 031 · Intermediate | Zigzag and corners | [Watch](01-1.50-mps/track-031-intermediate.mp4) | [Watch](02-0.50-mps/track-031-intermediate.mp4) |
| 032 · Advanced | Hairpin, gap, and crossing | [Watch](01-1.50-mps/track-032-advanced.mp4) | [Watch](02-0.50-mps/track-032-advanced.mp4) |
| 017 · Advanced | Gap, crossing, U-turn, and zigzag | [Watch](01-1.50-mps/track-017-advanced.mp4) | [Watch](02-0.50-mps/track-017-advanced.mp4) |

The videos use corrected evaluation version `progress-1.2`. Ordered gate tolerance now covers valid corner cuts within the accepted corridor. Off-route termination requires both being outside the 100 mm tracking corridor and having no active line sensors continuously for two seconds. The wider corridor accommodates the 90 mm forward sensor reach during turns. The recovery allowance covers the filtered PID's 0.738 second hold and a full one-second search sweep. All three controllers use these same rules at both caps. Five seconds without new authenticated progress still ends an episode as stalled.

Each panel freezes at the end of its episode, with the final status, progress, time, and speed on screen. Track 017 adds a difficult gap and zigzag example. These are retrospective replays on the previously assessed cohort.

[The corrected screenshot case](track-001-0.50-corrected.png) shows all three controllers still running at 7.43 seconds on track 001. All three finish that course at the 0.50 m/s cap.

[Episode results and previous outcomes](episode-summary.csv) · [Evaluation protocol](evaluation-protocol.json) · [Full telemetry](telemetry/) · [Video manifest and checksums](manifest.json) · [Recorder source](../../record_speed_comparisons.py)

To record these again on Windows, run the command below from the repository root. The recorder uses VLC for H.264 encoding. Pass `--vlc` if VLC is installed outside its standard location.

```powershell
.\.venv\Scripts\python.exe research/record_speed_comparisons.py --output runs/video-replay
```
