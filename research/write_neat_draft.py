"""Generate editable scientific drafts from frozen NEAT records and diagnostics."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
ARTIFACTS = RESEARCH / "artifacts/neat"


def main() -> None:
    data = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    runs = data["runs"]
    parent, continuation = (runs[k] for k in ("28-10-2026", "28-10-2026-hard"))
    diagnostic = (ROOT / "runs/diagnostics/zigzag-20261029/report.md").read_text(encoding="utf-8")
    references = diagnostic[diagnostic.index("[provenance]:"):]
    observations = diagnostic.split("## 1. Complete observation-to-action pipeline\n", 1)[1].split("### Confirmed learned-policy failure", 1)[0]
    fitness = diagnostic.split("## 3. Exact fitness and episode termination\n", 1)[1].split("## 4. Dynamics", 1)[0]
    neat = diagnostic.split("## 5. NEAT configuration and measured diversity\n", 1)[1].split("## 6. Curriculum", 1)[0]
    with (ARTIFACTS / "posthoc-cpu-validation.csv").open(encoding="utf-8") as stream:
        replays = list(csv.DictReader(stream))
    figure_prefix = "C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/neat"
    table = ["| Phase / difficulty | Completed / n | Mean observed RMSE (mm) | Mean progress | Failed progress | Successful completion time (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, record in (("Original, g38", parent), ("Continuation, g109", continuation)):
        for difficulty, metrics in record["validation"].items():
            failure_progress = "n/a" if metrics["mean_failed_progress"] is None else f"{metrics['mean_failed_progress']:.4f}"
            table.append(f"| {label}: {difficulty} | {metrics['completed']}/{metrics['episodes']} | "
                         f"{1000*metrics['mean_lateral_rmse_m']:.3f} | {metrics['mean_progress']:.4f} | "
                         f"{failure_progress} | {metrics['mean_success_completion_s']:.3f} |")
    replay_table = []
    for name in runs:
        subset = [r for r in replays if r["run"] == name]
        mean = lambda key: sum(float(r[key]) for r in subset)/len(subset)
        replay_table.append(f"| {name} | {sum(r['completed']=='True' for r in subset)}/30 | "
                            f"{1000*mean('lateral_rmse_m'):.3f} | {1000*mean('mean_absolute_lateral_error_m'):.3f} |")
    trajectory = "C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/trajectories.png"
    body = f"""# NEAT experimental setup, results and research interpretation

Scientific research draft. Prepared 29 October 2026 (Asia/Amman) from preserved records. Editable Markdown with LaTeX equations; no journal manuscript rewrite or claims of statistical significance are made here.

## 1. Experimental setup

### Research objective and scope

The study asks whether an evolved neural line-following controller offers useful performance relative to classical PID, and eventually possibly PPO, under a reproducible physical simulation. MuJoCo is a methodological choice, not the scientific contribution. The two main NEAT phases form **one evolution lineage, seed 0**, with a changed protocol at generation 100; they are not independent replications. Historical PyGame outcomes are not included as MuJoCo results.

The simulated competition-style robot uses a free chassis, two contact-driven actuated wheels and a passive rolling ball caster on a flat floor. Nominal mass is 0.55 kg, wheel-centre spacing 0.175 m, wheel radius 0.0165 m and width 0.012 m. The main PCB body is 0.090 by 0.130 by 0.0025 m; an explicit lumped mass/inertia models electronics and power hardware. These are simulation assumptions rather than measured hardware identification. The visible integrated PCB and its convex collision pieces share the sensor geometry. The caster is 0.045 m ahead of the axle; body +X is forward.

Wheel targets are filtered with a 0.050 s time constant and converted to torque-limited speed-servo actuation (gain 0.005, limit 0.03 N m per wheel). Wheel/caster sliding friction is 0.75/0.05. Physics runs at 600 Hz and controllers/sensors at 60 Hz, independently of rendering. Contact dynamics produce motion; no custom kinematic position update bypasses contacts. Ideal no-slip relationships are v=(uL+uR)/2 and yaw_rate=(uR-uL)/0.175. Signed motor commands permit pivots, although the selected sigmoid policies do not command them.

The sensor arc radius is 0.110 m and span 1.70 rad. The middle sensor is 0.090 m ahead of the wheel axle, outer sensors 0.052598 m ahead, lateral span 0.165282 m, nominal optical height 0.006 m. The separate 0.140 m-offset preset was **not** the geometry of these runs. Rays intersect the floor from actual 3-D robot poses. Reflectance and visible floor use one 1 mm raster over a 4 by 4 m arena. Observations are ideal binary point samples, with no tested noise, faults or hardware calibration.

### Observations, motor semantics and neural architecture

{observations}

### Evolutionary configuration

{neat}

### Changes between phases

| Setting | Original phase | Hard-track continuation |
|---|---|---|
| Directory | runs/28-10-2026 | runs/28-10-2026-hard |
| Recorded generations | 0-99 (100) | 100-173 (74 completed; 174 unfinished) |
| Nominal population | 200 | 200; actual library allocation varies, final saved population 199 |
| Episode slots per genome | 4 | 10 |
| Motor rim-speed ceiling | 0.30 m/s | 0.35 m/s |
| Training generator | v1.0 | progressive v2.0, longer motifs/routes |
| Curriculum | Generation-fraction schedule | Training-competence schedule plus 25-generation forced progression |
| Timeout | Explicit 60 s | clip(L/0.10,45,120) s |
| Fitness version | progress-1.0 | progress-1.1, optional beta=0 (same default arithmetic) |
| Backend / GPU batch ceiling | GPU / 200 | GPU / 200 |
| Evolution seed | 0 | same continuing lineage, 0 |
| Package source SHA256 | {parent['configuration']['environment']['source_sha256']} | {continuation['configuration']['environment']['source_sha256']} |

The geometry, torque limits and sensor ordering were retained. Generation counts and fitness trajectories from these phases must not be merged into a single unchanged-protocol learning curve. Source ZIPs accompany each run; no Git repository/commit was available. Nominal population 200 is not an assertion that every generation contained exactly 200 genomes: NEAT-Python species spawn rounding produced nearby values.

Hardware: Windows 11, Intel Family 6 Model 154 CPU with 12 logical processors; NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB. Python 3.12.14, MuJoCo/MuJoCo-Warp 3.17.0, Warp 1.18.0, NEAT-Python 0.92, NumPy 2.5.4, Pillow 12.3.0. GPU accelerates physics; inference, sensing and route accounting stay on CPU. Four CPU workers were configured for CPU-mode jobs; GPU mode uses batched worlds instead of that CPU pool. Rendering and later diagnostic resource contention are not excluded from recorded wall time.

## 2. Training methodology

Each genome controls two motors from the same 15 physical sensors. All genomes in a generation receive the same freshly generated track batch. Per-episode fitness is aggregated as half the mean plus half the minimum, so a bad route cannot simply be concealed by excellent easy-route scores. Selection, crossover, mutation, speciation and reproduction use the saved NEAT-Python configuration. The training-fittest genome is evaluated on 30 fixed validation tracks each generation. The saved model is selected lexicographically by validation completion fraction, then aggregate validation score. Validation is therefore reused for model selection, not an untouched performance estimate. Test outcomes do not affect evolution.

Original curriculum: generations 0-19 have four beginner tracks; 20-59 have two beginner/two intermediate; 60-99 have one beginner/one intermediate/two advanced. Intermediate generation already samples exact 90-degree corners and zigzags. The continuation uses 10-slot mixtures: 10 beginner; 5B/5I; 2B/6I/2A; 2B/3I/5A; 1B/2I/7A. It starts at stage 1 based on parent training observations. Rolling five-generation completion in the top 10% of training genomes controls progression after at least five generations, with B/I/A thresholds 0.85/0.70/0.55. A 25-generation plateau forces a stage increase without competence. Actual stage transitions were at generation 124 (intermediate completion 1.188%) and 149 (7.605%); stage 3 continued through 173. These rates are training-only, not validation results.

Tracks combine straights, arcs, S-curves, exact corners, alternating zigzags, hairpins/U-turns, gaps, perpendicular crossings and branches. The intended ordered route is evaluator-only. Gaps remove paint but retain route geometry; declared junctions preserve straight-through continuation. Geometry is checked for bounds, clearance, intersections and minimum arc radii, with finite retries and saved failed seeds. Zero-length corner primitives retain sharp angles. Progressive odd non-beginner slots require zigzags; later focused advanced routes require two motifs. A zigzag has a 45-degree entry/exit and three alternating 90-degree corners separated by 0.20-0.30 m straights. At stage 3 intermediate routes target 3.5-6.8 m and advanced 4.5-8.5 m; easier routes remain mixed in.

Stateless SHA256-derived seed streams separate evolution, training track generation and evaluation. Frozen manifests contain 30 validation and 90 test tracks, stratified equally by difficulty. Duplicate/mirrored geometry is grouped by canonical hash; protected hashes exclude held-out geometry from training. Adaptive training does not repeatedly use the single training-track manifest entry. The final 90 test tracks remained untouched at NEAT phase closure; a later jointly frozen PID/NEAT assessment will be labelled separately in Comparative_Analysis.md.

Checkpoint continuation restores the population, random state, species/reproduction information, curriculum state and seen hashes. The continuation's new phase resets phase-specific best-model/stagnation records and retains audited ancestry/weights from the parent. The phase file documents this methodological change. The original run is not overwritten.

## 3. Experimental results

### Training fitness and completion history

| Measure | Original 0-99 | Continuation 100-173 |
|---|---:|---:|
| Initial generation mean / best training fitness | {parent['initial_generation']['training_mean']} / {parent['initial_generation']['training_best']} | {continuation['initial_generation']['training_mean']} / {continuation['initial_generation']['training_best']} |
| Final generation mean / best training fitness | {parent['final_generation']['training_mean']} / {parent['final_generation']['training_best']} | {continuation['final_generation']['training_mean']} / {continuation['final_generation']['training_best']} |
| Population-mean fitness range | {parent['training_mean_range']} | {continuation['training_mean_range']} |
| Population-best fitness range | {parent['training_best_range']} | {continuation['training_best_range']} |
| Selected generation / genome | {parent['selected_generation']} / {parent['selected_genome']} | {continuation['selected_generation']} / {continuation['selected_genome']} |
| Selected validation completion | 16/30 (53.33%) | 16/30 (53.33%) |
| Last generation validation completion | 15/30 | 12/30 |
| Sum of retained generation wall times | {parent['recorded_wall_s']:.3f} s ({parent['recorded_wall_s']/3600:.3f} h) | {continuation['recorded_wall_s']:.3f} s ({continuation['recorded_wall_s']/3600:.3f} h) |
| Logical physics steps in retained generation records | {parent['logical_physics_steps']} | {continuation['logical_physics_steps']} |

Fitness values are shown for changing track mixtures. Decreases across a stage transition are not a controlled estimate of forgetting or regression. Every generation is available in the separate cleaned CSVs and figures below. Recorded wall-time sums include scheduling, warmup and potential sleep/resource contention; they are not measured active GPU time or matched-budget algorithm benchmarks. The original generation-27 wall-time outlier was not explained by physics workload alone; sleep/power-event access did not establish a cause. Overlapping resumes mean continuation wall-time totals are descriptive retained-row sums rather than an exact total of all consumed compute.

![Original phase learning curves](<{figure_prefix}/28-10-2026-learning.png>)

![Continuation learning curves, kept separate](<{figure_prefix}/28-10-2026-hard-learning.png>)

### Selected-policy historical GPU validation results

{chr(10).join(table)}

These are macro-means over tracks: each episode's observed-duration RMSE is calculated first and then averaged equally. Failed runs remain included. Successful completion times apply only to completed episodes and cannot be interpreted independently of completion fraction. Each selected model has 14 off-route failures and no other failure type in its 30 recorded selection evaluations. Both complete every beginner route but only 3/10 intermediate and 3/10 advanced. Repeated validation selection and the single lineage prevent an unbiased population-level generalization claim.

The historical summaries do not contain mean absolute lateral error (MAE), and RMSE does not determine MAE. To address the missing measurement without inventing it, 60 **new frozen-policy CPU validation replays** were performed after stopping NEAT, with the appropriate saved speed limit and timeout for each phase. No optimization or test-set access occurred. Their MAE values are separately labelled:

| Frozen selected policy | CPU replay completed / n | CPU mean observed RMSE (mm) | CPU mean observed MAE (mm) |
|---|---:|---:|---:|
{chr(10).join(replay_table)}

The completion counts match historical GPU validation, but numerical trajectory differences change errors slightly. These MAEs are not asserted to be values from the historical GPU episodes. See artifacts/neat/posthoc-cpu-validation.csv for all per-track rows.

### Sharp-corner and zigzag outcomes

The earlier diagnosis's **75,710 failed population evaluations on 379 training tracks containing zigzags** is verified for continuation generations 100-167 after logical deduplication. The finalized continuation through 173 contains **{continuation['population_zigzag']['episodes']} such evaluations on {continuation['zigzag_unique_tracks']} distinct tracks, zero completed routes**. The original phase separately contains **{parent['population_zigzag']['episodes']} evaluations on {parent['zigzag_unique_tracks']} zigzag-containing tracks, zero completions**. These are repeated evaluations of shared routes by related genomes; they are not independent trials or proof that each failure occurred at a zigzag.

The selected generation-109 policy completes an isolated 90-degree corner but fails at the next alternating corner of a minimized zigzag. Generation 167 fails at the first sharp corner. Replays of actual saved training tracks show the same distinction: on the 4.322 m intermediate route, corner arclengths are 0.61891 and 0.84930 m; generation 167 reaches 0.64015 m, selected g109 reaches 0.89094 m. A parallel pattern occurs on the 6.772 m advanced route. The diagnostic traces localize the failures, unlike broad feature tags.

The independent mechanical probe uses fixed timed commands and completes the minimized zigzag in 18.383 s (fitness 1.94915). The selected policy fails in 5.667 s at p=0.57730 (fitness 0.57048), and generation 167 in 3.117 s at p=0.37297 (fitness 0.36231). The slower successful run is strongly preferred by fitness. These probes demonstrate feasibility and failure mechanisms; a calibrated fixed command schedule is not a generalized deployable baseline.

![Actual diagnostic trajectories: selected model, later model and successful fixed commands](<{trajectory}>)

## 4. Failure analysis

**Directly demonstrated:** The generation-167 policy ignores the outer left sensor and produces approximately [0.9912,0.6627] for that sensor alone and for all-zero observations, steering right during the left-corner replay. Its sigmoid motor outputs cannot command wheel reversal or counter-rotation. The simulator itself reverses and pivots correctly. The selected model's tiny feedforward network also omits several sensor channels. These are properties of the learned policies, not disconnected hardware or reversed motor wiring.

**Confirmed observation limitation:** Fifteen instantaneous binary inputs omit speed, heading, previous actions and the last-seen line direction. Identical zero vectors occur after different kinds of loss and from opposite sides. Direction-aware pivots reacquired line signals from mirrored displaced poses; each fixed policy showed a directional bias. Such mechanical reacquisition does not guarantee recovery within the episode corridor or subsequent completion. The array has limited preview and no rear sensing; at a static left 90-degree approach it can lose every signal before the axle reaches the corner if no turn is executed.

**Confirmed evaluator constraint:** Chassis progress is ordered and gate-authenticated; seeing paint at the forward board is not sufficient. In g109's minimized zigzag the axle truly leaves the 45 mm corridor, misses a gate, then approaches a later visible segment without returning to the authenticated interval. Off-route termination is therefore not simply an all-zero sensor trigger. Gates intentionally prevent shortcuts and branch hopping; their recovery effects must be included in any interpretation.

**Confirmed schedule behaviour, unisolated causal effect:** The plateau rule advanced before target competence. Hardening added longer routes and multiple features while early corners remained unsolved. A genome failing the first corner cannot experience later motifs during that episode. The available counts therefore support a persistent sharp-turn learning problem rather than a shortage of nominal track evaluations.

**Hypotheses, not experimentally proven remedies:** Forty-percent weight replacement, balanced node addition/deletion, small selected networks and five-generation stagnation on changing batches may impede incremental skill preservation. The snapshot at next generation 168 has 200 distinct genomes across 28 species, so simple genetic collapse is not supported. A continuous geometric tracking term or temporal observations may help, but no controlled NEAT ablation established that they do.

**Fitness interpretation:** Completion and progress dominate. The historical secondary term is bounded to one percent of progress and commonly saturates, limiting guidance for partial corner improvements. Speed is mainly rewarded through finishing sooner; measured chassis speed and geometric RMSE are not direct primary terms. Mean/min aggregation can focus ranking on one unlearned worst track. This is different from claiming a fast partial route beats a slow complete route.

**CPU/GPU consistency:** Ten fixed-policy closed-loop CPU/CUDA comparisons had matching outcomes. A 270-control-step identical-action check had maximum position/orientation differences about 5.96 micrometres / 5.51e-6 rad, zero sensor-disagreement frames and exact same-pose batched sensing. A 17-to-8-world compaction probe preserved outcomes. These checks found no GPU discrepancy explaining the reproduced failures. Float32 versus double-precision trajectories can still diverge later at binary pixel thresholds; universal equivalence was not proven.

**Data integrity:** Concurrent resumes of generation 167 produced duplicate rows with slightly conflicting numerical values. The first logical record is retained for analysis, with raw records preserved and all exclusions audited. No evidence establishes that this caused the already-existing corner-learning failure. The precise branch associated with competing checkpoint writes is uncertain; this weakens reproducibility of the overlap interval and must not be concealed. No new run-ownership fix was mixed into the preserved NEAT experiment.

## 5. Discussion

Under this robot, instantaneous observation space, evolved activation-dependent motor semantics, fitness, procedural generator and budget, the selected policies follow easy tracks and some harder routes but do not reliably negotiate alternating sharp turns. Their success on beginner held-out routes demonstrates some transfer; failure on more complex held-out routes and on newly authored short zigzags limits a broad generalization claim. Because the training population also fails these feature-containing routes, the evidence is not simply strong training performance followed by unseen-track memorization. It indicates incomplete acquisition of the hard manoeuvre as well as limited transfer.

The continuation selected a model with the same 16/30 validation completion count as the earlier selected model. Lower sample RMSE and different elapsed times cannot be attributed uniquely to further evolution: speed ceiling, track distribution and timeout changed, and validation was reused repeatedly. The later generation number does not imply a better saved controller. Single-track versus multi-track and curriculum-versus-fixed-mixture experiments were planned but were not completed as matched independent-seed studies; the small single-track smoke run is not that baseline.

Difficulty stratification does not perfectly balance feature composition. Validation contains nine zigzag-tagged routes out of 30, while the later 90-track test contains 40. A validation-to-test completion decrease therefore cannot, by itself, identify memorization or validation overfitting; the challenge mix also differs. The common test compares controllers on the same individual tracks.

The physical feasibility probes narrow the explanation: the task is not automatically impossible for the simulated mechanism. They do not establish that memoryless binary sensing is sufficient for every route, nor that a PID or PPO implementation is superior. Those claims require controllers and evaluation conditions to be defined and tested. NEAT is not universally inferior based on this one configuration and lineage. The present outcome is an implementation-specific experimental baseline, including unsuccessful episodes, not an algorithm ranking.

## 6. Limitations and future work

- One independent evolution seed; parent and continuation share ancestry. Telemetry samples, repeated validation and related genomes are not independent runs. No significance test is justified by treating them as independent observations.
- One nominal robot model, uncalibrated mass, motor and friction assumptions, ideal binary point sensors, and one procedural generator family. No hardware validation or sim-to-real claim.
- Restricted observations and output activation semantics; no trained recurrent controller or encoder-informed NEAT condition. Giving future methods memory or signed outputs changes the controller comparison and must be disclosed.
- Curriculum and generator changed across phases; no controlled factorial ablation of speed, curriculum, mutation or fitness. Fresh batches make training scores nonstationary.
- Repeated validation-based selection over generations risks validation overfitting. Final tests were untouched at NEAT closure. Sensor faults, disturbances and motor degradation were not tested and must not be called robustness results.
- MAE missing from historical summary schema; retrospective measurements are separately labelled. Incomplete-run errors cover observed duration, and successful-run times have survivor selection.
- Concurrent-resume logging/checkpoint races, raw developmental-run inconsistencies and unrecorded hardware power-state events limit exact computational accounting. The archive preserves these limitations rather than repairing history silently.

NEAT development was intentionally stopped at this point to establish classical PID and, potentially later, gradient-based PPO baselines. No further NEAT optimization was launched. Useful future strengthening would include multiple independent seeds, a true single-/multi-track comparison, controlled action/observation ablations, short-corner competence tests, generator-held-out feature combinations and a single frozen final test after all tuning. Recommendations are hypotheses until tested.

## 7. Reproducibility appendix

### Raw data, checkpoints and source identity

- Main raw data: runs/28-10-2026 and runs/28-10-2026-hard; each has run.json, source.zip, neat-config.txt, champion-episodes.csv, training-episodes.csv, generations.csv, tracks/, champions/, best.pkl and checkpoints/latest.pkl.gz.
- The preserved final continuation checkpoint has next generation 174, so **173 is the last completed generation**. Generation 174 has track/progress artifacts but no completed result and is excluded from outcome tables.
- The raw continuation status.json remains its initial next-generation-100/training-not-started record because it was not refreshed before the active process was stopped. It was preserved unchanged; the validated checkpoint, completed generation rows and stop receipt determine the actual stopped state. This stale status is not evidence that training never ran.
- Archive: research/archive/neat-20261029/raw-experiments-and-source.zip; 2,628 files, 196,137,417 uncompressed bytes. SHA256 **4c64201ce24689e4a19665efd99efcf19140006d8b936fe90c565282270b27e2**. manifest.json contains per-file hashes and the pre-PID environment; stop-receipt.json records verified stopped process identities and time. continuation-final-checkpoint.pkl.gz is an additional preserved copy.
- Generator/cohorts: benchmarks/suite/suite.json for original v1; benchmarks/progressive-v2/suite.json versions training separately while preserving the original 30/90 held-outs. Complete JSON geometry, hashes, split membership and seed records are archived.
- Diagnostics: runs/diagnostics/zigzag-20261029, including the full technical report, frozen models/tracks, CPU/GPU checks, physical/recovery probes, trajectories and source hash verification. Engineering validation is also under runs/validation/adaptive-20261029 and earlier validation folders.
- Derived data: research/artifacts/neat/summary.json, duplicate-audit.csv, per-run generations-clean.csv and champion-episodes-clean.csv, figures, and posthoc-cpu-validation.csv. Extraction scripts are research/analyze_neat.py, research/replay_neat_metrics.py and research/write_neat_draft.py.

### Inventory and incomplete records

The original 100-generation run is complete; the continuation is intentionally stopped after 74 additional completed generations. runs/my-curriculum-run is a developmental session with status claiming five generations but only two surviving generation CSV rows; it is archived but excluded from the principal learning-curve analysis. runs/smoke-single contains one completed short smoke generation, with a remaining target, not a fully trained single-track baseline. runs/smoke-suite, runs/tool-checks, runs/previews, geometry-pilot-v1/v2 and runs/validation contain engineering/tool/geometry checks, not independent NEAT study replications. The machine-readable summary inventory records each available run's source hash, configuration, status and observed row count; unavailable identifiers are null rather than invented.

### Cleaning and aggregation rules

Original CSVs remain byte-identical to the archive. Logical keys are generation for generation summaries; generation/role/track_hash for champion episodes; generation/genome_id/track_hash for population episodes. The first occurrence is retained. Primary outcome analysis excludes **2,051 duplicate rows**: one generation row, 40 champion rows and 2,010 population rows. Supporting episode-fitness, diagnostics, GPU-profile and curriculum logs are also audited using generation/episode-slot/track, generation/role/category/value, generation/phase/batch, and generation respectively. The full audit contains **{len_data_audit(data)}** duplicate logical records across all seven logs, with per-file counts below. Every raw row is preserved; no failed physical episode is dropped. Cleaned data never overwrites raw data.

```json
{json.dumps(data['duplicate_count_by_file'], indent=2)}
```

Completion is all ordered gates plus finish. Progress is greatest authenticated arclength divided by route length. Per-episode RMSE is sqrt(mean(e_t^2)); MAE is mean(abs(e_t)), with e_t the signed error to the authenticated local route window. Both use equally spaced 60 Hz post-step samples over the observed episode duration. Tables average episode metrics equally across tracks; they are not pooled telemetry estimates. Failure progress averages failed episodes only; successful time averages completed episodes only. Feature-tagged performance means routes containing that feature and does not localize the failure to that feature. Technical errors are separate from normal failures. Timing includes host overhead; no energy or pure active-GPU estimate was recorded.

### Exact frozen fitness specification

The specification below gives the continuation values. In the original phase, the legacy threshold m=max_speed/2 is 0.15 rather than 0.175, and the explicit time-limit override is 60 s. Beta remains zero. The unit adapter and all other default arithmetic are retained as recorded.

{fitness}

### Source references

The source links below are the pre-PID implementation inspected for the NEAT diagnosis. Source ZIPs and hashes remain authoritative if later tools add files. The equations retain documented historical arithmetic rather than retrospectively correcting the experiment.

{references}
"""
    summary = f"""# NEAT results summary

29 October 2026; preserved experimental baseline, not an algorithm-wide conclusion.

**Scope:** one evolution lineage (seed 0), 15 instantaneous binary sensors, feedforward NEAT, two evolved motor outputs, MuJoCo contact dynamics. Original phase: 100 generations (0-99), four tracks/genome, 0.30 m/s, 60 s timeout. Continuation: 74 completed generations (100-173), ten tracks/genome, adaptive progressive tracks, 0.35 m/s, length-based 45-120 s limits. These phases have separate learning curves and are not independent repetitions.

**Selected models:** original generation 38 and continuation generation 109 each completed 16/30 repeatedly used validation tracks: 10/10 beginner, 3/10 intermediate, 3/10 advanced. Historical GPU mean observed lateral RMSE: 24.987 mm and 17.742 mm. Different settings and repeated selection prevent attributing that difference to a controlled improvement. Historical MAE was unavailable; separately labelled CPU replays measured 14.780 mm and 12.084 mm, with 16/30 completion in both cases.

**Sharp-turn result:** the g109 policy passes an isolated 90-degree turn but fails consecutive alternating corners; g167 fails at the first sharp corner. The original phase has 0/{parent['population_zigzag']['episodes']} completions on {parent['zigzag_unique_tracks']} zigzag-containing training tracks; the finalized continuation has 0/{continuation['population_zigzag']['episodes']} on {continuation['zigzag_unique_tracks']} tracks. The earlier diagnostic 0/75,710 on 379 tracks is verified for generations 100-167. These are dependent evaluations, and mixed-route failure may precede the zigzag.

**Mechanism evidence:** g167 ignores the outer left sensor and steers the wrong way on that cue and all-zero input. Selected policies' sigmoid outputs cannot command reverse/pivot, although motors accept signed actions. Stateless sensing lacks last-seen direction. Curriculum advanced at 1.19% and 7.61% intermediate elite completion via plateau rules. Every non-elite offspring weight is perturbed or resampled; its causal effect remains untested. Fitness strongly prefers a slow completion over a fast failure but has weak continuous corner shaping.

**Physical/backend checks:** fixed timed commands complete the minimized zigzag in 18.383 s; a ground-truth oracle completes actual mixed routes. These are engineering probes, not deployable sensor-policy successes. Ten CPU/GPU comparisons matched outcomes; short identical-action differences were below 6 micrometres. No GPU bug explaining the failure was demonstrated.

**Resources/integrity:** retained generation wall-time sums are {parent['recorded_wall_s']/3600:.3f} h and {continuation['recorded_wall_s']/3600:.3f} h, including overhead/possible contention, not active-GPU benchmarks. A concurrent resume duplicated generation-167 records; raw files are preserved and logical exclusions audited. Source archives/hashes identify versions, with no Git commit. Checkpoint after g173 is preserved; unfinished g174 is excluded.

**Limits and next step:** one uncalibrated robot/generator and one independent seed; no faults, disturbances, hardware transfer or matched NEAT/PID/PPO study yet. NEAT development is stopped to establish PID and potentially PPO baselines. The 90 test tracks were untouched at NEAT phase closure. No universal superiority/inferiority or statistical significance is claimed.

Details: [NEAT_Experimental_Results.md](NEAT_Experimental_Results.md); data and figures under artifacts/neat; archive and hashes under archive/neat-20261029.
"""
    common_path = RESEARCH / "artifacts/pid/summary.json"
    if common_path.exists():
        common = json.loads(common_path.read_text(encoding="utf-8"))
        result = next(r for r in common["test"] if r["condition"] == "neat_frozen_g109" and r["difficulty"] == "all")
        body += f"""\n## 8. Subsequent frozen common assessment\n
After the historical NEAT phase was closed and archived, the unchanged generation-109 model was assessed alongside finalized PID policies on the same 90 held-out tracks. Its artifact hash was frozen before PID tuning; no NEAT optimization occurred. All methods used CPU MuJoCo, the continuation's unchanged 0.35 m/s robot and the same length-based episode rules. This is a **new fixed-policy test assessment**, separate from the historical GPU validation and the post-hoc CPU validation replay.\n
The frozen model completed **{result['completed']}/{result['episodes']}** tracks, with mean observed lateral RMSE **{1000*result['observed_lateral_rmse_m']:.3f} mm**, directly measured MAE **{1000*result['observed_lateral_mae_m']:.3f} mm**, mean progress **{result['mean_progress']:.4f}**, and **55 off-route failures**. It completed 30/30 beginner, 2/30 intermediate and 3/30 advanced. None of the 40 routes containing zigzags was completed. No model was reselected against these outcomes. Source data: runs/pid-research-20261029/test-episodes.csv; protocol/freeze and paired conditions are documented in Comparative_Analysis.md.\n
The 90 tracks were untouched at NEAT phase closure and were consumed only after all current controllers were frozen. Future model development must not use these now-known outcomes as an optimization signal; a new confirmatory cohort may be needed. This test result does not constitute a new independent evolution run or establish algorithm-wide superiority/inferiority.\n"""
        summary += f"\n**Later frozen common test:** unchanged g109 completed {result['completed']}/90 (30/30 B, 2/30 I, 3/30 A), RMSE {1000*result['observed_lateral_rmse_m']:.3f} mm and MAE {1000*result['observed_lateral_mae_m']:.3f} mm on CPU under the continuation's settings. This new test is separate from historical GPU validation; no NEAT training or reselection followed. See Comparative_Analysis.md.\n"
    (RESEARCH / "NEAT_Experimental_Results.md").write_text(body, encoding="utf-8")
    (RESEARCH / "NEAT_Results_Summary.md").write_text(summary, encoding="utf-8")
    print("Saved NEAT research draft and one-page summary")


def len_data_audit(data: dict) -> int:
    return sum(data["duplicate_count_by_file"].values())


if __name__ == "__main__":
    main()
