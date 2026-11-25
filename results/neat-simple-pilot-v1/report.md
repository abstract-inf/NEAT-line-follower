# Compact NEAT reward — matched development comparison

Warm-start development sensitivity only. All validation data are consulted; one ancestry, no independent-seed or final-test claims.

**Completed pilot.**

| Arm | Generations | Primary / retention | Min straight m/s | Paired time vs PID | Shared finishes | RMS command change | RMSE mm |
|---|---:|---|---:|---|---:|---:|---:|
| lambda-0 | 20 | 60/60; 20/20 | 0.9224 | 6.027 / 6.893 s | 59/60 | 0.07527 | 16.71 |
| lambda-1 | 20 | 60/60; 20/20 | 0.8786 | 6.009 / 6.893 s | 59/60 | 0.06762 | 16.77 |
| lambda-10 | 20 | 60/60; 20/20 | 0.9211 | 6.043 / 6.893 s | 59/60 | 0.05862 | 20.05 |

Highest own-coefficient fitness on 60 primary validation tracks; ties earliest generation/rank. This is development selection, not a new final winner.

Time comparisons use each pair's shared successful tracks; different rows can have different denominators. Error/RMS columns above are observed all-episode means. Full paired error/jitter, failures, physical yaw, per-group completion and all candidate records are in `report.json`.

The frozen accepted prototype and tuned full-base PID are development controls, not independent training trials. No threshold is relaxed and no rejected historical model is promoted.

Immutable analysis snapshot: `snapshots/3864b04984d3709ad84f2fa3b24153b79e40230673450ade4538cdbc8446e74f/`. Its `analysis.py` can replay all captured numerical inputs with `--snapshot <snapshot> --output <new-output>`.
