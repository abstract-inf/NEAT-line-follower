# DEVELOPMENT / EXPLORATORY: saved g21 candidate versus PID

This uses the same 60 saved primary development courses at a verified 1 m/s cap. The candidate is generation 21, rank 1, chosen by the saved training report. It remains unqualified: its minimum required-straight speed is below 0.95 m/s. No official winner is promoted; these reused selection data are not a fresh test.

| Controller | Completion | Failures | Observed mean episode geometric RMSE | Observed mean command jitter |
|---|---:|---|---:|---:|
| NEAT g21 rank 1 | 60/60 | none | 13.973 mm | 0.0062373 |
| Full-speed filtered PID | 59/60 | {'off_route': 1} | 22.102 mm | 0.0011875 |
| Fast NEAT g134 | 60/60 | none | 14.493 mm | 0.0065299 |

Observed means include all 60 episodes, including partial exposure on failures. They do not replace the matched shared-success comparisons below.

| Left / right controller | Shared successful tracks | Mean finish time left / right | Mean episode geometric RMSE left / right | Mean jitter left / right |
|---|---:|---:|---:|---:|
| NEAT g21 rank 1 / Full-speed filtered PID | 59/60 | 6.092 / 7.123 s | 13.988 / 20.565 mm | 0.0061896 / 0.0011714 |
| Fast NEAT g134 / Full-speed filtered PID | 59/60 | 5.993 / 7.123 s | 14.605 / 20.565 mm | 0.0065187 / 0.0011714 |
| NEAT g21 rank 1 / Fast NEAT g134 | 60/60 | 6.115 / 6.016 s | 13.973 / 14.493 mm | 0.0062373 / 0.0065299 |

| Controller | Required / missing windows | Minimum required-window speed | Maximum raw yaw RMS | Maximum steering RMS |
|---|---:|---:|---:|---:|
| NEAT g21 rank 1 | 20 / 0 | 0.8721 m/s | 0.0984 rad/s | 0.01475 |
| Full-speed filtered PID | 20 / 0 | 0.9269 m/s | 0.0744 rad/s | 0.01576 |
| Fast NEAT g134 | 20 / 0 | 0.8298 m/s | 1.3956 rad/s | 0.16766 |

Missing windows contribute zero required-window speed; yaw/steering maxima describe observed windows. Acceptance limits are speed ≥0.95 m/s, raw yaw RMS ≤0.15 rad/s and steering RMS ≤0.02.

![Matched finish times and geometric errors](matched-course-comparison.png)

Each point is one shared successful course, colored by course group. Below the diagonal means a smaller candidate value. Failed courses remain in the all-episode summary and exact captured JSON, and are excluded from shared-success finish-time plots. Lower command jitter and lower geometric error measure different behavior; no blanket superiority claim follows.

Input snapshot: `cfc2659d90bb4df9be692246831547704ab0933153c0cd064d515d1fb61e4467`. Exact consumed report, episode caches, frozen model/parameter bytes, suite/protocol, manifest and reporting source are preserved under `snapshots/`. `report.json` records input/model/source hashes, failure rows and paired course identities. Existing snapshots are verified and never overwritten.

Replay from the project root with the recorded Python/dependencies (this reads the snapshot only):

```powershell
& .\.venv\Scripts\python.exe 'C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\results\neat-quality-speed-v1\candidate-pid-development\snapshots\cfc2659d90bb4df9be692246831547704ab0933153c0cd064d515d1fb61e4467\report-source.py' --snapshot 'C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\results\neat-quality-speed-v1\candidate-pid-development\snapshots\cfc2659d90bb4df9be692246831547704ab0933153c0cd064d515d1fb61e4467' --out 'C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\results\neat-quality-speed-v1\candidate-pid-development'
```
