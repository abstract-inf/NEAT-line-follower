# Compact NEAT reward — matched development comparison

Warm-start development sensitivity only. All validation data are consulted; one ancestry, no independent-seed or final-test claims.

**Completed pilot.**

| Arm | Generations | Primary / retention | Min straight m/s | Paired time vs PID | Shared finishes | RMS command change | RMSE mm |
|---|---:|---|---:|---|---:|---:|---:|
| lambda-0 | 50 | 60/60; 20/20 | 0.9548 | 5.980 / 6.893 s | 59/60 | 0.08570 | 17.23 |
| lambda-1 | 50 | 60/60; 20/20 | 0.9555 | 5.953 / 6.893 s | 59/60 | 0.06472 | 19.36 |
| lambda-10 | 50 | 60/60; 20/20 | 0.9144 | 6.024 / 6.893 s | 59/60 | 0.05126 | 20.22 |

Highest own-coefficient fitness on 60 primary validation tracks; ties earliest generation/rank. This is development selection, not a new final winner.

Time comparisons use each pair's shared successful tracks; different rows can have different denominators. Error/RMS columns above are observed all-episode means. Full paired error/jitter, failures, physical yaw, per-group completion and all candidate records are in `report.json`.

The frozen accepted prototype and tuned full-base PID are development controls, not independent training trials. No threshold is relaxed and no rejected historical model is promoted.

Immutable analysis snapshot: `snapshots/5c4e5e5ea6f1f2461040cdaba557279f983f78da5daa11f47a382f5f373a673a/`. Its `analysis.py` can replay all captured numerical inputs with `--snapshot <snapshot> --output <new-output>`.
