# MuJoCo line-following research environment

A contact-driven differential-drive robot, procedural tracks, sensor-only PID
and NEAT interfaces, reproducible evaluation, and user-operated training tools.
All physical quantities use SI units. The robot is a nominal simulation model,
not an identified or manufactured hardware replica.

**Current follow-up:** the [smooth-driving NEAT workflow](docs/neat-smooth-study.md)
compares continued old-fitness training with a physical-smoothness reward at a
1 m/s cap. The [curated audit and frozen inputs](results/neat-smooth-study-v1/README.md)
document the starting models and measured recovery weaknesses. See the
[GitHub workflow](docs/github-workflow.md) for branch, ignore and data-archive rules.

For the current experiment, see [five-seed study commands](docs/methodology.md#run-the-five-seed-study) for training, resuming, live viewing, replaying a winner, plotting, and final assessment.

The follow-up experiment uses six training courses per genome and validates faster candidates against the previous ramp policies. See [the fast NEAT study guide](docs/fast-neat-study.md) for its methods and terminal commands.

For the corrected 1.00 m/s comparison, use the [full-cruise PID versus frozen NEAT report](research/NEAT134_vs_Fullspeed_PID_100_Results.md). Both controllers can request full straight-line speed; the report includes frozen paths, measured speeds, and the ten-track replay command.

**Corner training results, 4 November 2026.** The generation 94 controller
finished 88 of 90 fresh test tracks. The self-contained notebook includes the
four published figures, their data, and the plotting code.
[Results notebook](research/NEAT_Corner_Plots.ipynb) · [Experiment report](research/NEAT_Corner_Results.md)

To continue from g94 with a rising wheel speed cap, see the [speed training guide](docs/speed-training.md).

The completed speed run is documented in the [corrected speed results report](research/NEAT_Speed_Results.md) and [corrected matched PID notebook](research/NEAT_Speed_PID_Comparison.ipynb). The [historical checkpoint notebook](research/NEAT_Speed_Checkpoints.ipynb) preserves the original evaluator's results.
See the [experiment methodology](docs/methodology.md) for the methods record, code map, and evidence for the paper.

**Earlier experiments, 29 October 2026.** The project preserved the models from the
original 100-generation phase and 74 completed continuation generations.
The recorded robot geometry was frozen for those experiments. PID is the
active development baseline; a bounded 120-configuration study is complete.
See `research/NEAT_Experimental_Results.md`, `research/PID_Experimental_Results.md`
and `research/Comparative_Analysis.md`. Historical training commands below
remain available for reproducibility.

## Install and inspect

Use Python 3.11 or later. The tested local interpreter is Python 3.12.14.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,plots]"
.venv\Scripts\python.exe -m line_follower --help
```

The project uses MuJoCo 4.0 and MuJoCo Warp 4.0 for CUDA. If you have a 3.x
environment, create a fresh virtual environment for 4.0 because the Python
distribution was split into separate core and Studio packages.
`requirements-lock.txt` records the pinned environment, including optional GPU,
plot, and test packages.
For an exact reproduction, install its pinned versions in a fresh environment;
the editable local project itself is installed separately.
Historical MuJoCo 3.17 training checkpoints require the matching source and
dependency versions for exact continuation.

```powershell
.venv\Scripts\python.exe -m line_follower view --pid-config configs/pid-pilot.json
.venv\Scripts\python.exe -m line_follower render --camera top --output runs/previews/top.png
.venv\Scripts\python.exe -m line_follower render --camera front --output runs/previews/front.png
.venv\Scripts\python.exe -m line_follower render --camera robot --output runs/previews/perspective.png
```

Viewer keys: **Space** pause, **R** reset, **N/P** track, **C** PID/saved model,
**T** route overlay, **B** transparency. `--seed 10 --difficulty advanced --count 6`
offers a sequence of tracks. Rendering/overlays never change reflectance data.

To inspect the robot and track in a browser, export a self-contained model and
drop it onto [MuJoCo Live](https://live.mujoco.org/).

```powershell
.venv\Scripts\python.exe -m line_follower export-studio --seed 10 --difficulty advanced
```

The export is saved to `runs/previews/line-follower.mjb`. Use the desktop `view`
command for PID and NEAT runs. The MuJoCo Studio 4.0.0 Windows wheel lacks the
`headless_ui` extension required to stream the Python simulation to a browser.

## Editable settings

| File | What to edit |
| --- | --- |
| `run_training.py` | Top-level settings: HEADLESS, WORKERS, BACKEND, population, generations, EPISODES_PER_GENOME, EPISODE_TIME_LIMIT and paths |
| `configs/robot-competition.json` | Complete versioned robot parameters; `sensor_offset` measures axle to center sensor |
| `configs/robot-offset-140mm.json` | Same chassis and sensor spacing with a 0.140 m center offset |
| `line_follower/config.py` | Documented dataclass defaults for physical parameters and episode rules |
| `line_follower/robot_design.py` | Named PCB/mount construction constants and connected mechanical geometry |
| `configs/geometry-pilot.json` | PID-only pilot offsets, speeds, shared tuning grid, worker count |
| `configs/pid-pilot.json` | Shared pilot gains; base 0.6 corresponds to a 0.18 m/s nominal straight target |
| `line_follower/configs/neat.txt` | Historical NEAT-Python 0.92 evolutionary settings |

There is no training at import time. Mutable process state is kept inside
objects, except the explicitly documented process-local multiprocessing worker
handle. Workers never share a MuJoCo data instance. Prefer editing a copied JSON
configuration to changing defaults midway through an experiment.

## Sensor-offset pilot

Current center and outer sensor longitudinal distances are 0.090000 m and
0.052598 m. Sensor radius/span are 0.110 m / 1.70 rad; all 15 sites preserve
left-to-right ordering. See `docs/sensor-mount.md` for coordinates and mechanics.

```powershell
.venv\Scripts\python.exe -m line_follower geometry-pilot --config configs/geometry-pilot.json --output runs/my-geometry-pilot
.venv\Scripts\python.exe -m line_follower geometry-report --run runs/my-geometry-pilot
.venv\Scripts\python.exe -m line_follower view --robot-config configs/robot-offset-140mm.json --track runs/my-geometry-pilot/tracks/evaluation/evaluation-03-sharp.json --pid-config runs/my-geometry-pilot/pid-selected.json
```

This tunes **one common PID gain set**, then compares 9/12/14/16 cm offsets on
the same separate pilot evaluation tracks at three nominal speeds. It never
evolves NEAT or reads the main study test set. Complete inputs, source snapshot,
CSV outcomes, failure reasons, 60 Hz telemetry, and plots are preserved. Read
`docs/geometry-pilot-protocol.md` before changing its selection criteria.

## Select geometry and prepare the study

After reviewing the pilot and choosing dimensions yourself:

```powershell
.venv\Scripts\python.exe -m line_follower freeze-robot --robot-config configs/robot-competition.json
```

The freeze records the full configuration and generated MJCF hashes. It is
never created automatically. A later geometry change needs a new versioned
freeze file and reevaluation. The supplied default suite contains one fixed single-track
training benchmark, 30 validation tracks and 90 stratified test tracks. Generation
rejects duplicate/mirrored geometry across splits. Preserve this suite unchanged.

For a separate fresh suite, use `prepare-suite --output benchmarks/new-suite`
and point `TRACK_SUITE` to its manifest. Existing suite directories are never
overwritten.

## Start training yourself

Edit the top-level variables in `run_training.py`, then inspect without running:

```powershell
.venv\Scripts\python.exe run_training.py --dry-run
```

When you intend to begin evolution:

```powershell
.venv\Scripts\python.exe run_training.py
```

Fresh launcher sessions reserve an unused output folder automatically: the
configured `OUTPUT_DIRECTORY`, then `-02`, `-03`, etc. Existing folders are
preserved, including empty folders. The actual path prints before opening the
observer and is also shown by `--dry-run` without creating anything. Resuming
keeps the configured original directory; CLI `train` still requires an empty
explicit output directory.

`HEADLESS=True` uses no rendering or live-pose exports. `HEADLESS=False` opens
an independent MuJoCo window immediately and displays **every genome** from
actual evaluation snapshots. `PREVIEW_POPULATION_GRID=True` starts with separate
arenas for every individual; zoom to inspect or press **F** to follow one robot.
**G** toggles an overlay on one shared track. Their physics worlds remain
independent in both views. In overlay mode the initial poses can look like one
robot because every genome starts at exactly the same position.
For responsiveness, distant robots show the real PCB, wheels, caster and 15
sensor packages; the selected genome also shows the small mechanical details.
These display-only simplifications, opaque diagnostic colors and disabled
shadows/reflections do not affect physical geometry or observations. Native
geometry batching avoids copying thousands of objects in Python while holding
the UI lock. `POPULATION_DISPLAY_HZ=5` in `line_follower/population_view.py`
limits pose refreshes while mouse camera interaction runs independently.
Unused native physics panels are hidden in the observer; use the keys below.
The display reports running, waiting and finished individuals for the selected
track. Finished robots retain their last pose. **N** changes the observed track.
By default the observer follows the track with the most running genomes and
selects a robot still evaluating. **N** selects a track manually; **[ / ]**
selects a robot manually. **A** restores automatic live selection. A static
finished track is not a paused training simulation. The title explicitly labels
paused display, completed training, and exports older than five seconds. The
observer cannot restart training or advance stopped evaluator worlds.

With `BACKEND="gpu"` and `GPU_BATCH_SIZE=POPULATION_SIZE`, all 200 genomes start
the same track together, followed by the next track. A smaller GPU batch
evaluates subsets; CPU concurrency is `WORKERS`, with the rest waiting. Seeing
all genomes does not mean that 200 CPU workers exist. Live poses are sampled at
`SNAPSHOT_HZ=10` in `line_follower/population_view.py`, not recorded every physics
step. Validation retains the last training-population view and reports validation
episode progress. **SPACE** freezes the display only; closing it leaves training running.

**F** frames and follows the selected robot's translation while preserving
subsequent mouse orbit, pan offset and zoom. **[ / ]** select another genome.
**C** switches overview/robot-follow; **U** disables following for a free camera.

Attach a window to a run already in progress, from another terminal, without
restarting training:

```powershell
.venv\Scripts\python.exe -m line_follower preview-training --run runs/my-curriculum-run
```

An older running process cannot acquire live-population exports retroactively;
its observer uses a clearly labeled single-policy replay. `PREVIEW_REPLAY_SPEED=10`
advances ten unchanged control steps per frame for these older runs (CLI:
`preview-training --replay-speed 10`). This setting never paces live evaluation.
Current processes retain their loaded code. Existing checkpoints
still require the source version preserved in that run's `source.zip` to resume;
the strict source check is not bypassed by this viewer update.

An **episode** is one genome attempting one track. The current launchers use
`EPISODES_PER_GENOME=10`, giving `POPULATION_SIZE * 10` training episodes each
generation, plus 30 champion-validation episodes for the supplied suite. Set
`EPISODE_TIME_LIMIT=None` for the route-based timeout or supply an explicit
limit in **simulated seconds**. There is no fixed wall-clock duration per
generation and faster-than-real-time execution is not guaranteed. CPU sensing,
network inference, metric processing, GPU transfers and setup also cost time.
The viewer's rendering FPS is unrelated to evaluation speed. Physics always
uses 600 Hz and control 60 Hz; speed comes from doing those steps faster, not
changing dt. Console output, the observer, and `progress.json` report measured
GPU batch world speed (`maximum simulated episode time / batch wall time`) and
aggregate throughput (`sum of simulated episode times / batch wall time`).
For example, 200 worlds each advancing one simulated second in two wall seconds
means 0.5x world speed and 100 simulated world-seconds/s aggregate throughput.
There is no artificial real-time cap to remove, and 10x per-world speed with an
entire population cannot be guaranteed on a particular machine. Completed generation times are saved in
`generations.csv`.

After training episodes finish, each generation prints the population's average
and best aggregated fitness, plus signed changes from the previous generation.
It also prints average/best episode scores across genomes for each track slot,
including the winning genome ID. These per-slot summaries are saved in
`episode-fitness.csv`; individual scores remain in `training-episodes.csv`.
Tracks and curriculum difficulty can change across generations, so training
fitness changes alone do not establish improvement on a fixed evaluation set.

The original protocol used four episodes and the CLI retains that baseline
default. Four episodes preserve the original research mixtures exactly. Other positive
counts are supported for deliberate configuration changes: larger batches
repeat the mixture, and partial four-slot batches rotate deterministically
across generations. A smaller episode count changes the protocol/compute
budget; use the same declared setting across the compared conditions.

CLI for the original generation-based curriculum baseline:

```powershell
.venv\Scripts\python.exe -m line_follower train --strategy curriculum --seed 0 --workers 4 --headless --robot-config configs/robot-competition.json --output runs/my-run
# Replace --headless with --visualize for the live population observer.
```

Each genome sees the same four tracks within a generation. Persistent Windows
spawn workers parallelize independent evaluations. Start with four workers and
measure runtime before increasing the count; available CPU/memory and viewer
load matter. The production objective is one policy that transfers across tracks.
`single` and non-curriculum `multi` conditions support the memorization and
curriculum comparisons. No study is launched unless you run its command.

Checkpoints are saved after reproduction/generation advancement. Resume in the
same run directory with the same code, dependency environment, suite, geometry,
and settings. In the launcher set `RESUME_CHECKPOINT`; for CLI use `--resume`.
Saved scheduling/settings take precedence on both launcher and CLI resume.
The launcher prints any research settings it overrides and includes them in
`--dry-run` output. Edited globals remain available for the next fresh session;
`HEADLESS` and camera preferences still apply to the current launcher session.
Source/dependency
changes are rejected rather than silently changing an ongoing experiment.
Resume removes uncommitted generation log rows and restores the checkpoint's
best model. Keep a backup of the run directory before recovering from a hard crash.

On Windows, a reader or synchronization tool can briefly lock `progress.json`.
Progress-display writes are disposable and no longer abort evaluations when
locked. Essential JSON metadata retries transient access failures and still
raises persistent errors. The observer tolerates a temporarily unreadable
progress record and labels stale exports explicitly.

`prepare_resume.py --run RUN_DIRECTORY` prepares an audited **copy** of a
checkpoint after observer/logging-only fixes; it never runs evolution. It
verifies the original source archive against the checkpoint, requires every
scientific module to remain byte-identical, and compares the experiment module
AST after excluding only `write_json` and `_progress_reporter`. Physics,
controller, fitness, scheduling, evaluation and dependency changes are rejected.
With `--gpu-capacity-fix`, the recovery audit additionally permits increases
to `CONTACTS_PER_WORLD` and `CONSTRAINTS_PER_WORLD` only; it rejects changes to
GPU stepping code or smaller capacities. The capacity change is explicitly
recorded in the transition manifest.
The derived checkpoint, source-transition manifest and updated source archive
are saved alongside the unchanged original checkpoint/source. Resume using the
printed derived checkpoint path. An interrupted generation reruns from its
preceding completed checkpoint; partial episode evaluations are not resumed.

To continue a **completed** population under a harder training distribution,
use `continue_training.py`. It prepares a separate phase with the full saved
population, records deliberate speed/exposure changes, and leaves the
parent run intact. For `runs/28-10-2026`, the defaults add 100 generations with
ten shared fresh tracks, training-based curriculum advancement, longer routes,
a 0.35 m/s speed ceiling, bounded route-based time limits and optional CVaR
aggregation. The original episode score and mean/min baseline are retained.
See [the continuation protocol](docs/continuation-training.md) before
interpreting its results as research evidence.

```powershell
.venv\Scripts\python.exe continue_training.py --dry-run
.venv\Scripts\python.exe continue_training.py --visualize
# The same command resumes the prepared phase after an interruption.
```

## Optional GPU backend

The RTX 3050 Laptop GPU was detected and CUDA physics was tested with MuJoCo
Warp 4.0.0 / Warp 1.18.0 on this Windows machine. Install optional dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[gpu]"
.venv\Scripts\python.exe -m line_follower gpu-benchmark --worlds 200 --workers 4 --output runs/gpu-probe.json
```

The launcher now uses `BACKEND="gpu"` and `GPU_BATCH_SIZE=POPULATION_SIZE`,
and the train CLI defaults its GPU batch size to the requested population.
Reduce it if device memory is insufficient (e.g. `--gpu-batch-size 128`).
CPU worker count applies to the CPU
backend; GPU evaluation batches worlds in one process. NEAT inference, sensor
sampling and route accounting still run on CPU. The first GPU use compiles CUDA
kernels and can take minutes. The GPU backend is opt-in and experimental.

Editable GPU capacities are in `line_follower/gpu_backend.py`:
`CONTACTS_PER_WORLD=512` and `CONSTRAINTS_PER_WORLD=2048`. The shared contact
pool is `512 * active batch worlds`: 102,400 slots at 200 worlds and 2,048 at
four worlds. Solver rows remain a separate per-world allocation. These defaults
replace 32/128 after a collision-buffer overflow in a small batch. Overflow
checks remain enabled: a technical overflow must never be scored as policy
failure. Larger buffers reserve more device memory; 1-, 4-, and 200-world
contact-heavy stress cases passed on this machine. These finite stress checks
are not a guarantee for every future geometry or state.

GPU tasks are scheduled track-first and scattered back to the original per-genome
track order before scoring. Sensor rays are sampled together using the same
full-pose calculation as CPU. When at least half of a batch has finished, batches
larger than 16 worlds are compacted, preserving pose, velocity, actuator filters,
solver warm starts and episode state. This avoids repeatedly stepping finished
worlds. Device buffer retention is bounded. Mixed-track champion validation
still batches tracks together. As with other float32 CUDA operations, changing
batch shape can produce small numerical differences; source and batch size must
be held fixed for reproducibility. No controller, fitness, geometry or timestep
was changed for these optimizations.

The measured 200-world open-loop physics probe took 0.318 s on GPU versus
0.999 s on four CPU workers; 32 worlds favored CPU (0.146 s vs GPU 0.238 s).
These single diagnostic measurements exclude first-use compilation, controller
inference and sensing. They do not establish full-training speedups. Repeat the
benchmark on your intended workload.

GPU state uses float32. Small trajectory differences can change later binary
observations, so bitwise CPU/GPU equivalence is not claimed. The unsupported
AUTORESET diagnostic flag is omitted only from the GPU model; finite-state and
overflow checks remain explicit. Per-generation logs record backend, logical
physics steps, GPU executed/padding world steps, and time. Model/run metadata
record the backend. Keep a consistent backend for a declared experiment and
validate it on full episodes before relying on confirmatory results.

Primary documentation: [MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html).
The JAX/MJX route has different native-Windows GPU constraints; the optional
adapter here uses Warp directly, without JAX.

## Evaluate and export

Generate learning curves and selected-model validation plots directly from a
completed run's saved CSVs, without running training or fresh evaluations:

```powershell
.venv\Scripts\python.exe plot_training.py --run runs/my-curriculum-run
.venv\Scripts\python.exe -m line_follower view --model runs/my-curriculum-run/best.pkl --track runs/my-curriculum-run/frozen-validation/000.json
```

The first command writes PNG/PDF/SVG figures, per-difficulty CSV summaries and
a JSON report under the run's `plots/` folder. The second is a CPU visual replay;
it can differ from saved float32 GPU validation trajectories. `best.pkl` is
selected by validation, and may come from an earlier generation. Training
fitness spans changing tracks/difficulty; validation is reused for selection
and must not be presented as untouched test performance. One evolution run
does not establish training-run variability or superiority over PID.

For separate standardized evaluations and multi-controller comparisons:

```powershell
.venv\Scripts\python.exe -m line_follower evaluate --suite benchmarks/suite/suite.json --split validation --model runs/my-run/best.pkl --telemetry --output runs/model-validation
.venv\Scripts\python.exe -m line_follower evaluate --suite benchmarks/suite/suite.json --split validation --pid-config configs/pid-pilot.json --telemetry --output runs/pid-validation
.venv\Scripts\python.exe -m line_follower evaluate --seen-run runs/my-run --model runs/my-run/best.pkl --output runs/model-seen
.venv\Scripts\python.exe -m line_follower analyze 'runs/*-validation/episodes.csv' --output runs/analysis
```

The common final evaluation command uses CPU MuJoCo for all controllers and
records both trained and evaluation backends. Test tracks require explicit
`--split test --final-test`, after controller/geometry/protocol selection. Do not
tune on their outcomes. Pilot-selected PID gains are an engineering starting
point, not the finalized main-study PID tuning protocol.

All incomplete episodes remain in CSV. Lateral RMSE is in meters; historical
sensor-index error is separate. Completion time is absent on failures. Telemetry
records pre-action observations and post-step observations separately. Repeated
deterministic runs are not independent statistical evidence. Paired analysis
accounts for held-out track pairing and evolution-run variation; prespecify
final sample sizes/analysis before the main experiments.

## Code map and checks

`simulation.py` owns contacts, actuation and physical sensing; `tracks.py` owns
geometry/raster generation; `controllers.py` owns sensor-only policies;
`fitness.py` owns ordered progress and the labeled legacy scorer; `episodes.py`
owns shared stopping/metrics; `experiments.py` owns workers/scheduling/checkpoints;
`viewer.py` owns visualization; `gpu_backend.py` is the optional batched adapter.

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .cache/pytest
$env:RUN_MUJOCO_GPU_TEST='1'
$env:RUN_MUJOCO_GUI_TEST='1'
.venv\Scripts\python.exe -m pytest tests/test_experiment_tools.py -q --basetemp .cache/pytest-tools
```

Tests use fixed networks and physical episodes without starting evolutionary
training. GPU/window checks are opt-in. Testing an actual evolution/checkpoint
continuation run is left to the user while the no-training instruction applies.

Research assumptions, units and historical discrepancies are in
`docs/research-record.md`. Retain `THIRD_PARTY_NOTICES.md` when publishing code.
The manuscript rewrite remains a later explicitly authorized task.

## PID research tools and preserved records

For the complete definitions of filtered PID and PID with corner supervision,
including equations, the state diagram, decision thresholds, configuration
provenance and research-paper interpretation, see
[PID controller methodology](research/PID_Controller_Methodology.md).

The existing `controllers.py` PID retains its historical semantics. New research
variants are in `line_follower/pid.py`: filtered PID and an explicitly labelled
PID with corner supervision. They use the real 15 binary sensors and dt; the
supervisor estimates short approach motion from prior commands and never reads
route geometry or robot pose. Its extra memory and geometric prior must be
acknowledged when comparing it with feedforward NEAT.

`run_pid_optimization.py` exposes the seed, worker count and bounded search
budget as editable top-level constants. A fresh exploratory run is:

```powershell
.venv\Scripts\python.exe run_pid_optimization.py --output runs/pid-new-run
```

This runs headlessly with persistent CPU workers and records every candidate,
training episode, validation selection and frozen final assessment. Existing
output folders are rejected. The completed study is `runs/pid-research-20261029`;
`best.json` is the selected supervised configuration, while
`selected-pid_filtered.json` preserves the separate filtered baseline. These
JSON files use `ResearchPIDConfig`, not the legacy `--pid-config` schema.

To visualize the selected controller, use the existing `view_tracks` Python API:

```python
import json
from pathlib import Path
from line_follower.config import RobotConfig, EpisodeConfig
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.tracks import TrackSpec
from line_follower.viewer import view_tracks

record = json.loads(Path("runs/pid-research-20261029/run.json").read_text(encoding="utf-8"))
robot = RobotConfig(**record["robot"])
episode = EpisodeConfig(**record["episode"])
controller = ResearchPIDController(
    ResearchPIDConfig.load("runs/pid-research-20261029/best.json"), robot)
track = TrackSpec.load("runs/pid-research-20261029/tracks/training/003.json")
view_tracks([track], robot, controller, episode)
```
The 90 test tracks were opened only after current policies were frozen. Their
outcomes are now known: future optimization must not tune against them, and a
new untouched cohort is needed for a fresh confirmatory study. PPO has not been
implemented. NEAT raw data/source/checkpoints are archived with SHA256 manifests
under `research/archive/neat-20261029`; original CSVs are unchanged. Rebuild
research tables and plots with `research/analyze_neat.py`,
`research/write_neat_draft.py`, and `research/write_pid_results.py`.

<!-- PID-GAINS-FOLLOWUP -->
## PID gain-only tuning follow-up

The completed follow-up is `runs/pid-gains-20261029`. It changes only P/I/D of each frozen PID variant; its selected JSON files and `best.json` use `ResearchPIDConfig`. Original runs are preserved. See [results](research/PID_Gain_Tuning_Results.md) and [prespecified protocol](research/PID_Gain_Tuning_Protocol.md).

Run a fresh study with `.venv\Scripts\python.exe run_pid_gain_tuning.py --output runs/pid-gains-new-run`. Seed, workers and bounded search controls are editable globals. Rebuild the original reports first, then run `research/analyze_pid_gains.py` to restore follow-up sections and figures. Existing test cohorts are now known; fresh confirmatory work needs a new prespecified suite seed.

<!-- FROZEN-NEAT-NEW90 -->
## Frozen NEAT on the new PID cohort

The unchanged generation-109 model has now been assessed on the same new 90 tracks as the gain-tuned PID. See [matched results](research/NEAT_New_Cohort_Evaluation.md). `evaluate_frozen_neat.py` performs evaluation only; existing outputs are protected. Run `research/analyze_neat_new90.py` last when rebuilding research documents.
