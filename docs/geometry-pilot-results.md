# Sensor-offset pilot recorded 27 October 2026

The protocol was saved before evaluation in `geometry-pilot-protocol.md`.
Inputs/outcomes, 60 Hz telemetry and source snapshot are in
`runs/geometry-pilot-v2/`. Version 2 reran the same deterministic pilot with
the final shared episode tools and source snapshot: its evaluation CSV is
byte-identical to the earlier pilot, not additional independent evidence.

One PID gain set was selected on separate pilot tuning tracks, balanced across
all four offsets: Kp=0.12, Ki=0.0001, Kd=0.001. It was evaluated on the same
12 distinct pilot courses at nominal straight targets 0.10/0.18/0.26 m/s.
Base command was target/0.30; physical speed/torque limits remained unchanged.
The original default PID configuration was not overwritten.

| Center offset (m) | Outer-sensor X (m) | Completed / 36 | Mean observed lateral RMSE (m) | Minimum PCB clearance (m) | Maximum tilt (deg) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.090 | 0.052598 | 22 | 0.020050 | 0.007252 | 0.272 |
| 0.120 | 0.082598 | 20 | 0.026391 | 0.007129 | 0.259 |
| 0.140 | 0.102598 | 18 | 0.030793 | 0.007069 | 0.242 |
| 0.160 | 0.122598 | 16 | 0.032117 | 0.007025 | 0.223 |

There were no all-zero-sensor events and no tips in these trials. Nevertheless,
unsuccessful episodes departed the permitted ordered-route corridor. A forward
array can continue seeing paint while the axle cuts the corner, so line-loss
count alone is not a suitable success measure.

At 0.18 m/s target, 90 mm completed 12/12 courses and 140 mm completed 6/12.
At 0.26 m/s, 120 mm completed 8/12 versus 4/12 for 90 mm. The gain set was tuned
at 0.18 m/s; this is evidence of speed/controller dependence, not a universal
claim that a short offset is always better. Longer preview alone did not solve
sharp-corner tracking with this baseline. Actual observed speeds and featurewise
sharp-turn completion counts are preserved in the raw/summary CSVs.

The prespecified pooled completion/progress/error ranking recommends keeping
**90 mm as the starting configuration**, especially for the 0.18 m/s pilot
operating point. It is not adequate evidence to finalize geometry for every
speed/controller. The 140 mm configuration remains editable and available in
`configs/robot-offset-140mm.json`. No main geometry freeze was created; that
selection remains the user's explicit action before evolution.

RMSE and stability metrics from failed episodes cover only their observed
duration. These deterministic engineering results provide no confidence
interval, significance, NEAT-performance or hardware-validity claim. No main
study test outcomes were used. Main NEAT evolution remains user-operated.

Recreate raw trials with `geometry-pilot --config configs/geometry-pilot.json`
and an empty output directory. Recreate PNG/PDF/SVG figures from saved data with
`geometry-report --run runs/geometry-pilot-v2`. The generated report includes the
completion-by-speed table and the axle trajectories at a representative 90° turn.
