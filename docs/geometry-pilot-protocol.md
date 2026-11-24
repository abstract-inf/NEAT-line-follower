# Sensor-offset pilot protocol

This engineering pilot selects a starting geometry. It is separate from the
NEAT/PID research comparison and never reads that study's validation/test data.
No NEAT evolution is run. Deterministic episodes are not independent trials.

Before this pilot, the integrated PCB robot has a 0.090 m center sensor offset,
0.110 m arc radius, and 1.70 rad span. Sensor indices 0 and 14 are 0.052598146 m
ahead of the axle and 0.082640845 m to either side. Their planar axle distances
are 0.097959554 m; those are different from longitudinal preview distances.

Compare center offsets 0.090, 0.120, 0.140, and 0.160 m. Translate all 15 sites,
packages, and the curved PCB together. Only the integral neck length changes;
the main body, motors, wheels, caster, total mass, effective inertia, sensor
spacing, width, reflectance processing, and motor limits remain fixed. The
lumped body inertia approximation intentionally excludes the small inertia
change of a real longer PCB. No claim about mount flexibility is made.

Use the existing sensor-only PID calculation, with one shared gain set for
every offset and evaluation speed. Tune once on four dedicated pilot tracks
(straight, 90-degree corner, zigzag, circular curve), balanced across all four
offsets at a 0.18 m/s nominal straight target. Search Kp in {0.06, 0.12, 0.20},
Kd in {0.001, 0.004, 0.010}, and fixed Ki=0.0001. Include the initial repository
gains {2, 0.0001, 0.5} as a tuning reference. Select gains lexicographically by
completion fraction, mean valid progress, then lower observed lateral RMSE.
Do not tune separately for a favored offset.

Evaluate the selected gains on 12 different pilot tracks: three straights,
three sharp-corner courses, three zigzag courses, and three curve courses.
Include left/right turns, different dimensions and line widths. Check geometry
hashes to prevent tuning/evaluation overlap. Evaluate nominal straight target
speeds of 0.10, 0.18, and 0.26 m/s by changing the same PID's base command. The
physical wheel-rim command ceiling remains 0.30 m/s in every trial. Nominal
target is not guaranteed achieved speed, especially through turns.

Use the existing episode stopping rules, corridor, ordered gates, and time
limits unchanged. Save every failure, completion, valid progress, lateral
RMSE over the observed duration, measured speed, line-loss count/duration,
completed sharp turns, and maximum chassis tilt/minimum PCB clearance sampled
at 60 Hz. A sharp turn counts only after valid ordered progress reaches 50 mm
beyond its vertex. Save 60 Hz evaluation telemetry and all input track/config
files, hardware/dependency/source records, and CSV summaries.

Recommend a starting offset by completion fraction across the fixed evaluation
grid, then mean valid progress, then observed RMSE. Report results by feature
and speed so the pooled ranking does not hide failures on 90-degree corners.
Errors from incomplete episodes describe only their observed duration. Do not
claim statistical significance, universal optimality, or NEAT performance.
If the shared PID fails to complete the relevant corners, do not declare the
geometry finalized on the strength of partial progress or small early RMSE.

Preserve the 90 mm design and save any recommended configuration separately.
Final research geometry must be explicitly selected and its configuration
hash recorded before the user starts main evolution runs.
