# Research implementation record

For current protocols and completed experiments, see the [methodology record](methodology.md). This document preserves the earlier implementation history and defaults.

## Scope and provenance

The scientific question is whether a NEAT neural controller provides meaningful
advantages over PID for line following. MuJoCo supplies a more realistic
methodology; migration between engines is not the research contribution.
The future manuscript will be one coherent study in Overleaf-compatible LaTeX.
No manuscript rewrite or main NEAT experiment is authorized at this stage.

The historical manuscript supplied in this workspace and source repository
`abstract-inf/NEAT-line-follower`, commit
`a883c3a7ce4daed75836cf58688b748151d04248`, establish the original interface,
PID calculation, NEAT settings and legacy arithmetic. Selected original sources
and the MIT notice are archived in `docs/reference/original/`.

Original observations are 15 binary reflectance values, with two independent
motor commands. NEAT remains feedforward with evolving topology/weights.
NEAT-Python 0.92 is the explicit compatibility baseline. Native artifact
roundtrips are tested; an original historical genome has not been supplied and
its successful physical transfer has not been established. Pickle artifacts
must come from trusted sources.

## Historical discrepancies retained explicitly

| Topic | Historical implementation / interpretation |
| --- | --- |
| Nominal dimensions | 0.55 kg, 175 mm wheel spacing, 16.5 mm wheel radius; no identification or uncertainty measurements supplied |
| Pixel conversion | 1000 pixels/m was used by the source; this is bookkeeping, not physical calibration |
| Velocity | Source `get_velocity` multiplies already linear wheel values by radius again; legacy reward preserves that numerical error through an explicit adapter |
| Angular velocity | Historical derived value mixes pixel wheel-spacing and radius; legacy units are intentionally not labeled rad/s |
| PID equation | Runnable positional P+I+D calculation takes precedence over discrepant manuscript descriptions; integral clearing order is retained |
| Motor mixing | Source screen +Y is downward; MuJoCo +Y is left. Physical PID wheel mixing is frame-corrected to preserve intended steering direction |
| Sensor layout | Paper semicircle and XLINE ordering/offset differ; they remain distinct presets, not silently reordered inputs |
| Fitness variants | Headless source completion term is 10000, while manuscript/other variants differ; headless arithmetic is the named legacy reference |
| Speed sign | Legacy negative speed-penalty signs become speed rewards; preserved and labeled, not silently corrected |
| Reported error | Historical sensor-index error is separate from new geometric lateral RMSE in meters |

The redesigned integrated PCB robot is a new geometry revision. See
`sensor-mount.md` for mechanical dimensions, mass distribution, sensor equations,
caster relocation and implications. Historical PyGame results and interim
development smoke outputs are not MuJoCo research findings.

## Dynamics, observations and controller boundary

Physics runs at 600 Hz and control/sensing at 60 Hz independently of rendering.
Wheel-speed targets are filtered with a nominal 50 ms time constant and limited
to ±0.30 m/s rim target speed and ±0.03 N m per wheel. This is a speed ceiling,
not imposed constant forward velocity. The free chassis, two hinge wheels and
ball-jointed passive caster move through contact forces. Nominal wheel/caster
sliding friction coefficients are 0.75 / 0.05; explicit pairs prevent a floor
coefficient from overriding the caster through contact mixing. Solver/contact
parameters are visible in `line_follower/assets/robot.xml` and preserved in the
generated MJCF hash.

Each sensor uses its actual site pose to intersect a downward ray with z=0.
Invalid/upward/overrange intersections return background. The floor mask is
binary black=1 and white=0, using a shared approximately 1 mm raster. Its PNG
and MuJoCo UVs use the same world orientation; overlays do not change the mask.
There is no camera-image or display-color-dependent sensing/termination.

Controllers receive only the 15 readings and dt. Pose, exact route, progress,
planned gaps, branch labels and finish information are evaluator diagnostics.
The privileged slow waypoint driver is only a physical feasibility check and
is excluded from controller comparisons.

The PID source calculation starts at Kp=2, Ki=0.0001, Kd=0.5, base=1. The
engineering pilot explicitly tunes a different saved configuration once,
balanced across offsets. This does not establish the final study tuning
methodology. NEAT outputs retain their configured activation semantics; sigmoid
values are not rescaled to force reverse commands. Negative outputs remain
possible for applicable evolved activations. Both controllers face the same
bounded physical interface.

## Tracks, fitness and termination

Explicit straight/arc/corner primitives retain genuine angular corners. Motifs
include S-curves, zigzags, hairpins and U-turns. Visible gaps remove paint while
the ordered evaluation route remains continuous. Declared perpendicular
crossings/distracting branches do not change the intended straight-through
route. Validation rejects invalid bounds, widths, too-small radii, unintended
nearby geometry and invalid challenge placements using bounded retry/backtrack
generation. Track JSON saves parameters, seeds, feature labels and version.
Geometry hashes normalize rigid pose/reflection and group duplicates across
splits. A configurable painted finish runout is evaluation-independent paint,
allowing forward sensors to remain on line until the axle reaches its finish.

Ordered segment projection and gates prevent jumping to nearby future route
segments or collecting progress repeatedly through reversals. Valid progress
must remain inside the configured route corridor. The default score is

```text
F = p + I_completed - 0.1*t/T
    + 0.01*p*tanh(R_legacy / max(1, control_steps))
```

Here p is greatest valid normalized route progress. A stationary centered robot
gets no positive secondary reward. Equal progress completed faster scores
better; oscillation cannot accumulate fresh progress. Legacy arithmetic is
separate, explicitly versioned, and bounded in this score. Fitness aggregation
across four episode slots is `0.5*mean(F)+0.5*min(F)`.

Default limit is `max(45 s, 3*route_length/speed_limit)`. Terminate on completion,
invalid physics/technical error, tip, persistent corridor departure, timeout or
five seconds without 10 mm fresh progress. All-zero sensor intervals are logged
and do not immediately terminate an episode. Full configuration records define
the corridor, gate tolerance and departure duration. Incomplete-run metrics
cover only observed duration; their missing completion times are not imputed.

## Experimental tools and planned comparisons

Default settings are population 200, 500 generations, four episode slots per
genome, and five independent evolution seeds per condition. These are editable
study settings, not completed runs. Within each generation all genomes see the
same fresh-track batch. Separate deterministic namespaces isolate evolution,
track-generation and evaluation randomness.

* Single-track NEAT repeats one fixed training benchmark in its four budget
  slots. Deterministic repetitions do not create independent evidence.
* Multi-track NEAT uses one beginner, one intermediate and two advanced tracks
  throughout training.
* Curriculum NEAT uses four beginner tracks in the first 20%, two beginner/two
  intermediate in the next 40%, and the final mixture in the last 40%.
* PID is evaluated on the same held-out courses, with tuning/provenance recorded.

Generation champions are evaluated on 30 fixed validation tracks; saved best
models maximize validation completion rate, then aggregate score. Ninety
stratified final-test tracks remain unavailable for tuning. Test evaluation
requires a deliberate CLI flag. Training sees only test geometry hashes to
exclude overlap, not test files/outcomes. Seen-track diagnostics remain separate
from held-out scores. Strong seen performance plus poor transfer supports
track-specific overfitting; poor performance everywhere does not demonstrate
memorization.

CPU evaluation uses persistent spawn workers with worker-local state. The
optional GPU adapter batches physics with Warp, retaining the same CPU sensor
lookup, controller interface and episode metrics. It is experimental, uses
float32, and records its diagnostic-flag difference, GPU padding and backend.
Small contact differences can change later binary observations. A common CPU
final evaluator permits a shared reference evaluation, with both backends
recorded; do not conceal a training/evaluation backend change in a manuscript.

## Evidence, records and limitations

The PID geometry pilot is an engineering comparison on separate manually
defined courses. Its protocol is `geometry-pilot-protocol.md`; no main study
test tracks were used. It does not finalize geometry automatically. Frozen
configuration/MJCF hashes are required before the user starts main training.

Runs export episode CSVs and optional 60 Hz telemetry, including sensor input
time and next-step readings, commands, measured speeds, torques, progress,
lateral error, line loss and explicit failures. Run metadata preserve physical,
controller and generator parameters, dependency/source hashes, seeds, hardware,
worker count and runtime. The pilot also snapshots source in a ZIP. Preserve
run folders, source and lockfiles together; they are excluded from default Git
tracking to avoid silently committing large datasets.

The statistical tools summarize per evolution run, pair shared held-out tracks,
and bootstrap run/track variation. They do not infer independent samples from
telemetry rows or repeated deterministic trials. Prespecify the final analysis,
sample sizes, metrics and failure rules before main experiments. Report failures
and unfavorable NEAT results accurately. Fault, disturbance and motor-degradation
claims require those explicit experiments; unseen-track transfer alone does
not establish robustness to them. No hardware calibration, main-study NEAT
performance, statistical significance or publication conclusion is claimed.

The manuscript rewrite remains pending the user's original PDF/LaTeX sources,
completed experiments and explicit authorization. Keep useful foundations and
verified references, replace obsolete statements, and derive equations from the
implemented methods and actual evidence.

## Population observer and execution checks (2026-10-28)

The observer now displays every genome using throttled actual evaluator poses.
Grid translations and diagnostic colors exist only in the rendering process;
they never enter physics, sensor rays, controller inputs or route scoring.
Waiting and terminated individuals remain visible with distinct status counts.
CPU concurrency remains the configured worker count. CUDA can start an entire
population on one track when its batch capacity equals the population size.
The launcher now sets that capacity to `POPULATION_SIZE`. This scheduling and
source version must be preserved within a reproducible run.

Track-major GPU task ordering preserves each genome's original score ordering.
Reflectance rays are vectorized across worlds using the same float64 ray/floor
calculation as the scalar CPU path. Finished-world compaction preserves qpos,
qvel, actuator filter state, solver warm starts and logical episode counters.
CUDA batch-shape changes can still introduce float32 rounding differences;
short fixed-policy regression tolerances are 10 micrometers for route progress,
1e-5 for fitness, and 1e-4 m/s for average speed. These checks do not establish
long-horizon bitwise equivalence. Geometry, controllers, fitness and timesteps
were not changed for this work.

Engineering artifacts are in `runs/validation/population-view-20261028/`.
`probe.py` constructs 200 constant-output network fixtures without population
evolution. A primed CUDA pass advanced each world for 0.8 simulated seconds in
2.306 wall seconds with live snapshot exports: approximately **69.4 simulated
world-seconds/s aggregate throughput**, but only **0.347x individual-world
speed**. The GUI was closed during timing and opened separately for inspection.
This is not a training benchmark, does not include evolution/validation, and
does not establish a 10x per-world speedup. Rendering FPS is not simulation speed.

A separate 64-world mixed-duration check reduced executed physics steps from
30,720 to 16,320 by compaction, eliminating 14,400 padded steps. Its single-pass
timings include graph setup for the batch shapes; use the saved step counts as
evidence of removed work, not as a controlled full-training speedup estimate.
These fixtures are software checks, not NEAT experimental results.

## GPU buffer-capacity correction (2026-10-28)

The user-operated run encountered broadphase overflow with 32 allocated contact
slots per world (128 pooled slots in a four-world batch). The adapter correctly
aborted before returning fitness for the invalid batch. The generation remained
uncommitted. Contact-heavy regression poses reproduce the old broadphase,
narrowphase and solver-row overflow without creating or evolving a population.

The defaults are now 512 allocated contact slots per world and 2,048 solver
rows per world. The contact pool is shared, while solver capacity applies to
each world. Only these two allocation constants changed in the GPU adapter;
physics, observation processing, controller algorithms, fitness, control rates
and compaction behavior remain unchanged. Capacity increases are allowed by an
explicit source-transition audit only if every other GPU stepping statement
matches the original archive. Preserve the transition records and source ZIP.

Twelve CUDA stress cases (lowered PCB, tilt, side and inverted poses, each with
1, 4 and 200 worlds) passed with finite positions and no overflow. The lowered
pose generated up to 84 candidate pairs, 237 contacts and 715 solver rows per
world at the sampled control-block endpoints. A separate CPU calculation at
the initial lowered pose produced 323 contacts and 973 rows. The CUDA checks
include 30 physics steps per pose; they do not bound every intermediate count
or establish coverage of arbitrary future states. Raw capacity/count records
are saved as CSV and JSON in `runs/validation/gpu-buffer-capacity-20261028/`.

## Adaptive continuation tools (2026-10-29)

The completed `28-10-2026` run is preserved. The revised tools resume its full
checkpoint population in a separately audited phase, with ten fresh shared
tracks, training-only rolling curriculum decisions, independent length/motif
controls, route-based 45-120 s limits, an unchanged default episode fitness and
mean/min aggregation, optional exact empirical lower-tail CVaR, and descriptive
failure/feature/length diagnostics. The requested 0.35 m/s target ceiling is a
new recorded operating condition; chassis, sensing geometry, torque and control
rates remain unchanged. The new phase must not be treated as an independent seed.

See `docs/continuation-training.md` for equations, stage allocations, advancement
thresholds and plateau semantics. Generator 2.0 manifests are separate from the
original frozen benchmark. Existing held-out identities are preserved; long-route
generalization still needs a separately frozen long-route evaluation cohort.

Validation used fixed networks and an evaluator-only ground-truth geometry
probe, not NEAT evolution. The complete suite passed 81 tests (one optional GUI
smoke check skipped), including CUDA checks, checkpoint/RNG retention, synthetic
resume/diagnostic I/O, curriculum transitions, standard-generator regression,
time bounds, CVaR and versioned split protection. A transient Windows checkpoint
replacement lock was reproduced during testing; complete checkpoint files now
use unique temporary names and bounded replacement retries, keeping the old
checkpoint on persistent failure.

Generation 27's 7,181 s wall duration was 26.6 times the run median, while GPU
work was about 1.04 times the median. Historical aggregates cannot identify the
cause. New batch and phase host-wall timing separates setup, inference,
physics/transfer, accounting/observer and compaction; it does not claim isolated
GPU kernel measurements. A matched 50/50 beginner/intermediate fixed-network
probe measured 5.22 s for four slots and 13.30 s for ten (32 worlds, two-second
caps). Raw data, track specifications and inspection figures are under
`runs/validation/adaptive-20261029/`. No new NEAT performance is claimed.
