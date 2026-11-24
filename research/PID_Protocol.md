# Frozen PID development and final-assessment protocol

Prepared 29 October 2026 after NEAT preservation and before PID tuning. No NEAT training will be performed. Robot geometry, 0.35 m/s rim-speed ceiling, 600/60 Hz clocks, shared reflectance field and route/termination rules remain unchanged.

## Controller families

1. `pid_filtered`: normalized sensor-centroid PID, filtered derivative, bounded integral, error-dependent forward command, and last-seen-direction line-loss recovery.
2. `pid_corner_supervised`: the same PID plus an explicitly labelled sensor-only corner supervisor. Broad unilateral sensor patterns indicate a possible sharp corner; a short command-odometry approach is followed by counter-rotation until the central line is reacquired. A continuing central line cancels the turn at branches. No route, position, yaw, progress or track label is available to either controller. This is an assisted PID baseline, not a claim about unmodified textbook PID.

Both receive the existing 15 binary readings and dt, and share physical motor bounds with NEAT. Their temporal state and supervisor distinguish them from the preserved feedforward NEAT policies; this difference will be disclosed.

## Bounded optimization

- Fixed NumPy tuning seed 20261029, four persistent Windows-spawn CPU workers.
- At most 96 initial candidates: seeded historical-gain adapters plus randomized bounded configurations, balanced between the two families.
- Coarse evaluation on short training-only straight, mirrored 90-degree corners, the diagnostic zigzag and its mirror, and a curve.
- Keep the best six of each family; evaluate on a larger frozen training-only suite with 45-degree entries, tighter alternating turns, curves and separate generated beginner/intermediate/advanced tracks.
- Up to 24 local refinements around the best two configurations per family, using the same larger training suite. This is a fixed maximum of 120 candidates, not an indefinite search.
- Every attempted candidate and every episode is recorded, including failures and technical exceptions. Candidate ranking is lexicographic: completion fraction, mean authenticated progress, negative mean observed lateral RMSE, then negative mean successful completion time. Completion dominates error and speed.
- Bounds: normalized proportional gain 0.25-3.5, integral gain 0-0.25, derivative gain 0-0.12, forward command 0.25-0.75, minimum command 0.02-0.25, derivative-filter time 0.015-0.10 s, slowdown gain 0-5, line-loss hold 0.10-0.8 s, search command 0.2-0.65, corner approach command 0.2-0.5, corner lead distance 0-0.025 m, pivot command 0.25-0.6.

## Validation, freezing and final test

Training tracks drive all parameter exploration and refinement. Validate the top three larger-suite candidates per family on the same original 30 frozen validation tracks. Select one policy per family by the same lexicographic criteria; choose the overall PID policy from those validation results. Save exact configuration, hashes, sources, split hashes and selection criteria before loading any final-test track geometry.

After freezing, evaluate the selected PID policies and the already frozen generation-109 NEAT model on the same 90 held-out test tracks, CPU backend, identical robot/episode settings. No tuning, architecture changes or reselection follows test outcomes. Preserve failures. One evolution lineage and deterministic PID do not support population-level algorithm superiority or significance claims. Report descriptive paired-track results and identify the additional PID memory/supervision.

## Metrics and artifacts

Use the existing EpisodeResult columns plus per-episode mean absolute lateral error, policy ID, condition, seed, backend and wall time. RMSE and MAE use 60 Hz error to the authenticated local route window over observed duration; macro-averages weight tracks equally. Successful completion times condition on success. Export config/candidate/episode/validation/test CSVs and representative 60 Hz sensor, error, derivative, integral, motor-command, torque and trajectory traces. Archive a source snapshot and hardware/dependency record. PPO remains unimplemented with empty result cells.
