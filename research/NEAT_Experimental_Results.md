# NEAT experimental setup, results and research interpretation

Scientific research draft. Prepared 29 October 2026 (Asia/Amman) from preserved records. Editable Markdown with LaTeX equations; no journal manuscript rewrite or claims of statistical significance are made here.

## 1. Experimental setup

### Research objective and scope

The study asks whether an evolved neural line-following controller offers useful performance relative to classical PID, and eventually possibly PPO, under a reproducible physical simulation. MuJoCo is a methodological choice, not the scientific contribution. The two main NEAT phases form **one evolution lineage, seed 0**, with a changed protocol at generation 100; they are not independent replications. Historical PyGame outcomes are not included as MuJoCo results.

The simulated competition-style robot uses a free chassis, two contact-driven actuated wheels and a passive rolling ball caster on a flat floor. Nominal mass is 0.55 kg, wheel-centre spacing 0.175 m, wheel radius 0.0165 m and width 0.012 m. The main PCB body is 0.090 by 0.130 by 0.0025 m; an explicit lumped mass/inertia models electronics and power hardware. These are simulation assumptions rather than measured hardware identification. The visible integrated PCB and its convex collision pieces share the sensor geometry. The caster is 0.045 m ahead of the axle; body +X is forward.

Wheel targets are filtered with a 0.050 s time constant and converted to torque-limited speed-servo actuation (gain 0.005, limit 0.03 N m per wheel). Wheel/caster sliding friction is 0.75/0.05. Physics runs at 600 Hz and controllers/sensors at 60 Hz, independently of rendering. Contact dynamics produce motion; no custom kinematic position update bypasses contacts. Ideal no-slip relationships are v=(uL+uR)/2 and yaw_rate=(uR-uL)/0.175. Signed motor commands permit pivots, although the selected sigmoid policies do not command them.

The sensor arc radius is 0.110 m and span 1.70 rad. The middle sensor is 0.090 m ahead of the wheel axle, outer sensors 0.052598 m ahead, lateral span 0.165282 m, nominal optical height 0.006 m. The separate 0.140 m-offset preset was **not** the geometry of these runs. Rays intersect the floor from actual 3-D robot poses. Reflectance and visible floor use one 1 mm raster over a 4 by 4 m arena. Observations are ideal binary point samples, with no tested noise, faults or hardware calibration.

### Observations, motor semantics and neural architecture


The network receives exactly **15 instantaneous binary reflectance readings**, in sensor order 0–14. There is no centering, standardization or additional normalization: each input is a floating-point **0 or 1**. Black line pixels below intensity 128 produce 1; white pixels, positions outside the finite floor and invalid sensor-ray intersections produce 0. See [tracks.py:476][reflectance] and [simulation.py:22][rays].

| Input index | NEAT input key | Sensor position x from axle (m) | Sensor position y, left positive (m) | Range / normalization |
|---:|---:|---:|---:|---|
| 0 | -1 | 0.052598 | +0.082641 | {0,1}; unchanged |
| 1 | -2 | 0.062074 | +0.073238 | {0,1}; unchanged |
| 2 | -3 | 0.070341 | +0.062758 | {0,1}; unchanged |
| 3 | -4 | 0.077278 | +0.051352 | {0,1}; unchanged |
| 4 | -5 | 0.082782 | +0.039191 | {0,1}; unchanged |
| 5 | -6 | 0.086772 | +0.026452 | {0,1}; unchanged |
| 6 | -7 | 0.089190 | +0.013324 | {0,1}; unchanged |
| 7 | -8 | 0.090000 | 0.000000 | {0,1}; unchanged |
| 8 | -9 | 0.089190 | -0.013324 | {0,1}; unchanged |
| 9 | -10 | 0.086772 | -0.026452 | {0,1}; unchanged |
| 10 | -11 | 0.082782 | -0.039191 | {0,1}; unchanged |
| 11 | -12 | 0.077278 | -0.051352 | {0,1}; unchanged |
| 12 | -13 | 0.070341 | -0.062758 | {0,1}; unchanged |
| 13 | -14 | 0.062074 | -0.073238 | {0,1}; unchanged |
| 14 | -15 | 0.052598 | -0.082641 | {0,1}; unchanged |

Sensor 0 is leftmost, sensor 7 central, sensor 14 rightmost. The physical forward direction is body **+X**; positive left and right wheel targets produce forward motion. The sensor array is ahead of the axle. The observation does **not** include velocity, wheel encoder readings, heading, yaw rate, prior actions, earlier sensor readings, elapsed time, track identity, route geometry or progress. `act(observation, dt)` receives a timestep, but `NEATController.act()` does not feed it to the network. `reset()` has no state to clear because the network is feedforward. See [controllers.py:82][controller] and [installed NEAT feedforward implementation][feedforward].

For output node j, NEAT computes weighted sums and node activations. In the installed NEAT-Python 0.92 implementation:

$$\sigma(z)=\frac{1}{1+\exp[-\operatorname{clip}(5z,-60,60)]},\qquad
\operatorname{tanh}_{NEAT}(z)=\tanh[\operatorname{clip}(2.5z,-60,60)].$$

Output node 0 is the **left** motor; output node 1 the **right** motor. The native project models use no legacy motor swap. Outputs are clipped to [-1,1] and interpreted directly as normalized wheel-rim speed targets:

$$a_j=\operatorname{clip}(o_j,-1,1),\quad
v_j^*=0.35a_j\ \mathrm{m/s},\quad
\omega_j^*=v_j^*/0.0165\ \mathrm{rad/s}.$$

**There is no mapping `2*sigmoid-1`.** A sigmoid output therefore supplies only nonnegative targets, despite the motor interface accepting negative targets. A tanh output can supply signed targets. Both inspected saved policies have two **sigmoid motor outputs**; neither can command a genuine reverse or counter-rotating pivot. Mutation permits tanh, so it would be incorrect to say that every possible evolved genome is forward-only. At the snapshot, 113/200 genomes have at least one tanh output, and 21/200 have two. See [controllers.py:96][action], [simulation.py:181][step], [activations.py:11][activations], and [model/population inspection][records].

There is no positive minimum motor command and no imposed constant forward speed. Saturated sigmoid values can approximate stopping. Targets pass through a **50 ms actuator filter**, a wheel-speed servo and a **0.03 N·m torque limit**. Acceleration is limited by actuator response, inertia and contact physics; there is no separate hard acceleration or jerk clamp. See [simulation.py:83][actuators] and [robot.xml:36][mjcf].



### Evolutionary configuration


The saved configuration, rather than a hypothetical current library default, is authoritative. See [saved neat-config.txt][neatconfig].

| Setting | Actual value / meaning |
|---|---|
| Initial topology | `full_direct`: 15 inputs, 2 outputs, 0 hidden; initially 30 input-output links |
| Network type | Feedforward; cycles prohibited; recurrent config explicitly rejected by NEATController |
| Activation | Initially sigmoid; options tanh/sigmoid; activation mutation attempt probability 0.2 per node |
| Aggregation | Sum; no aggregation mutation |
| Weight initialization/bounds | Gaussian mean 0, SD 1; [-10,+10] |
| Weight mutation | Probability 0.6: add Gaussian noise, SD 0.3 |
| Weight replacement | Probability 0.4: redraw from initialization distribution |
| Bias initialization/bounds | Gaussian mean 0, SD 1; [-5,+5] |
| Bias mutation/replacement | 0.4 add noise SD 0.3; 0.1 redraw; otherwise unchanged |
| Node response | Initially 1; mutation power 0, so the configured mutation does not change it |
| Node addition/deletion | 0.02 / 0.02 attempted per offspring mutation |
| Connection addition/deletion | 0.2 / 0.2 attempted per offspring mutation |
| Enabled-link mutation | 0.07 attempt; implementation randomly assigns a boolean, so an attempted mutation can retain the value |
| Structural mutation | `single_structural_mutation=False`; addition and deletion attempts can occur in the same offspring |
| Compatibility coefficients | Disjoint 0.8, weight/attribute difference 0.8 |
| Species threshold | 2.0 |
| Reproduction elitism | 2 retained genomes **per species**, not two across the population |
| Parent selection | Top 15% of each species, with at least two parents available |
| Stagnation | Species maximum fitness, 5 generations; top 2 species protected |
| Extinction | Reset population on complete extinction |
| Stopping | No fitness-threshold termination; configured generation budget governs |

**Important implementation detail:** 0.6 weight mutation plus 0.4 replacement covers the entire random interval in NEAT-Python 0.92. Every surviving connection in a non-elite offspring is either perturbed or resampled; there is no ordinary unchanged-weight branch. For a 22-connection offspring this means, on average, about **8.8 weights are redrawn** in addition to the other perturbations. Elites are retained intact, so this does not mean all existing policies are destroyed every generation. Activation mutation selects from the options and may select the original activation. See [attributes.py:65][attributes] and [reproduction.py:140][reproductionimpl].

The snapshot has **200 genetically distinct genomes across 28 species**, not a collapsed single genotype. Hidden-node counts are: **33 with zero, 155 with one, 11 with two, one with three**. Median stored connections: **22** (range 15–25). Mean pairwise NEAT distance: **2.670**, range **0.300–4.867**. Among the 68 saved continuation champions, **31 have no hidden nodes and 37 have one**. These measurements demonstrate persistent small networks and diversity at this snapshot; they do not establish that more nodes would solve the task. Full historical population/species diversity was not logged, so it cannot be honestly reconstructed from champion models alone. See [records.json][records].

Two plausible learning impediments follow from code, but have not been isolated by retraining ablations:

1. Aggressive weight replacement, frequent connection deletion and only occasional retained topology growth can disrupt partial corner skills. The selected model is just two sigmoid units over ten expressed sensor channels.
2. Five-generation stagnation compares species maxima across **fresh track batches and changing difficulty**. Those fitness numbers are not measurements of the same task distribution at a curriculum transition. A species can appear stagnant when the tracks became harder. The curriculum does not reset species histories at each stage change. The current population's diversity rules out a simple monoculture explanation, but not excessive turnover or poor preservation of useful partial behaviours. See [installed stagnation implementation][stagnation] and [experiments.py:565][advance].

Recurrence is not currently supported by this controller implementation: merely setting `feed_forward=False` will raise an error. A future recurrent experiment would need a recurrent phenotype and independent per-episode state, including correct reset and GPU compaction handling. No recurrent state is silently being lost in the present feedforward experiment.



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
| Package source SHA256 | eacb3caf4cb87bbe2a4dbbd1097fd5a3bda21814ba56b6374ed06943c5ce8d21 | e2cf5a043d888d347c959b1c4bc361c0e12efc716f8929f7663fb06f84f714b6 |

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
| Initial generation mean / best training fitness | 0.43523276721644627 / 1.9961909722222222 | 0.3223282807491233 / 0.6146592712842315 |
| Final generation mean / best training fitness | 0.2936662482846492 / 0.9568493909524455 | 0.13876611554931675 / 0.2805643239884543 |
| Population-mean fitness range | [0.14256436963366748, 1.0820558296923106] | [0.10219758707358777, 0.4673606497106822] |
| Population-best fitness range | [0.2785607297140751, 1.998847222222222] | [0.2683800760264481, 0.9249779215125371] |
| Selected generation / genome | 38 / 5963 | 109 / 17934 |
| Selected validation completion | 16/30 (53.33%) | 16/30 (53.33%) |
| Last generation validation completion | 15/30 | 12/30 |
| Sum of retained generation wall times | 34247.137 s (9.513 h) | 30257.082 s (8.405 h) |
| Logical physics steps in retained generation records | 440503700 | 621544040 |

Fitness values are shown for changing track mixtures. Decreases across a stage transition are not a controlled estimate of forgetting or regression. Every generation is available in the separate cleaned CSVs and figures below. Recorded wall-time sums include scheduling, warmup and potential sleep/resource contention; they are not measured active GPU time or matched-budget algorithm benchmarks. The original generation-27 wall-time outlier was not explained by physics workload alone; sleep/power-event access did not establish a cause. Overlapping resumes mean continuation wall-time totals are descriptive retained-row sums rather than an exact total of all consumed compute.

![Original phase learning curves](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/neat/28-10-2026-learning.png>)

![Continuation learning curves, kept separate](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/research/artifacts/neat/28-10-2026-hard-learning.png>)

### Selected-policy historical GPU validation results

| Phase / difficulty | Completed / n | Mean observed RMSE (mm) | Mean progress | Failed progress | Successful completion time (s) |
|---|---:|---:|---:|---:|---:|
| Original, g38: all | 16/30 | 24.987 | 0.7345 | 0.4311 | 11.484 |
| Original, g38: beginner | 10/10 | 5.537 | 1.0000 | n/a | 7.482 |
| Original, g38: intermediate | 3/10 | 40.963 | 0.5653 | 0.3790 | 15.933 |
| Original, g38: advanced | 3/10 | 28.461 | 0.6382 | 0.4832 | 20.378 |
| Continuation, g109: all | 16/30 | 17.742 | 0.7587 | 0.4830 | 14.774 |
| Continuation, g109: beginner | 10/10 | 8.008 | 1.0000 | n/a | 9.808 |
| Continuation, g109: intermediate | 3/10 | 24.713 | 0.5672 | 0.3818 | 21.056 |
| Continuation, g109: advanced | 3/10 | 20.506 | 0.7089 | 0.5842 | 25.044 |

These are macro-means over tracks: each episode's observed-duration RMSE is calculated first and then averaged equally. Failed runs remain included. Successful completion times apply only to completed episodes and cannot be interpreted independently of completion fraction. Each selected model has 14 off-route failures and no other failure type in its 30 recorded selection evaluations. Both complete every beginner route but only 3/10 intermediate and 3/10 advanced. Repeated validation selection and the single lineage prevent an unbiased population-level generalization claim.

The historical summaries do not contain mean absolute lateral error (MAE), and RMSE does not determine MAE. To address the missing measurement without inventing it, 60 **new frozen-policy CPU validation replays** were performed after stopping NEAT, with the appropriate saved speed limit and timeout for each phase. No optimization or test-set access occurred. Their MAE values are separately labelled:

| Frozen selected policy | CPU replay completed / n | CPU mean observed RMSE (mm) | CPU mean observed MAE (mm) |
|---|---:|---:|---:|
| 28-10-2026 | 16/30 | 25.425 | 14.780 |
| 28-10-2026-hard | 16/30 | 17.421 | 12.084 |

The completion counts match historical GPU validation, but numerical trajectory differences change errors slightly. These MAEs are not asserted to be values from the historical GPU episodes. See artifacts/neat/posthoc-cpu-validation.csv for all per-track rows.

### Sharp-corner and zigzag outcomes

The earlier diagnosis's **75,710 failed population evaluations on 379 training tracks containing zigzags** is verified for continuation generations 100-167 after logical deduplication. The finalized continuation through 173 contains **83226 such evaluations on 417 distinct tracks, zero completed routes**. The original phase separately contains **24054 evaluations on 120 zigzag-containing tracks, zero completions**. These are repeated evaluations of shared routes by related genomes; they are not independent trials or proof that each failure occurred at a zigzag.

The selected generation-109 policy completes an isolated 90-degree corner but fails at the next alternating corner of a minimized zigzag. Generation 167 fails at the first sharp corner. Replays of actual saved training tracks show the same distinction: on the 4.322 m intermediate route, corner arclengths are 0.61891 and 0.84930 m; generation 167 reaches 0.64015 m, selected g109 reaches 0.89094 m. A parallel pattern occurs on the 6.772 m advanced route. The diagnostic traces localize the failures, unlike broad feature tags.

The independent mechanical probe uses fixed timed commands and completes the minimized zigzag in 18.383 s (fitness 1.94915). The selected policy fails in 5.667 s at p=0.57730 (fitness 0.57048), and generation 167 in 3.117 s at p=0.37297 (fitness 0.36231). The slower successful run is strongly preferred by fitness. These probes demonstrate feasibility and failure mechanisms; a calibrated fixed command schedule is not a generalized deployable baseline.

![Actual diagnostic trajectories: selected model, later model and successful fixed commands](<C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/trajectories.png>)

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

Original CSVs remain byte-identical to the archive. Logical keys are generation for generation summaries; generation/role/track_hash for champion episodes; generation/genome_id/track_hash for population episodes. The first occurrence is retained. Primary outcome analysis excludes **2,051 duplicate rows**: one generation row, 40 champion rows and 2,010 population rows. Supporting episode-fitness, diagnostics, GPU-profile and curriculum logs are also audited using generation/episode-slot/track, generation/role/category/value, generation/phase/batch, and generation respectively. The full audit contains **2129** duplicate logical records across all seven logs, with per-file counts below. Every raw row is preserved; no failed physical episode is dropped. Cleaned data never overwrites raw data.

```json
{
  "runs/28-10-2026-hard/generations.csv": 1,
  "runs/28-10-2026-hard/champion-episodes.csv": 40,
  "runs/28-10-2026-hard/training-episodes.csv": 2010,
  "runs/28-10-2026-hard/episode-fitness.csv": 10,
  "runs/28-10-2026-hard/diagnostics.csv": 55,
  "runs/28-10-2026-hard/gpu-profile.csv": 12,
  "runs/28-10-2026-hard/curriculum.csv": 1
}
```

Completion is all ordered gates plus finish. Progress is greatest authenticated arclength divided by route length. Per-episode RMSE is sqrt(mean(e_t^2)); MAE is mean(abs(e_t)), with e_t the signed error to the authenticated local route window. Both use equally spaced 60 Hz post-step samples over the observed episode duration. Tables average episode metrics equally across tracks; they are not pooled telemetry estimates. Failure progress averages failed episodes only; successful time averages completed episodes only. Feature-tagged performance means routes containing that feature and does not localize the failure to that feature. Technical errors are separate from normal failures. Timing includes host overhead; no energy or pure active-GPU estimate was recorded.

### Exact frozen fitness specification

The specification below gives the continuation values. In the original phase, the legacy threshold m=max_speed/2 is 0.15 rather than 0.175, and the explicit time-limit override is 60 s. Beta remains zero. The unit adapter and all other default arithmetic are retained as recorded.


Let p be the greatest authenticated ordered-route progress divided by route length, C=1 only for completed episodes, t elapsed simulation time, T the episode time limit, N the number of control steps, and R the cumulative historical scorer return. The implemented episode fitness is

$$F=p+C-0.1\frac{t}{T}
+0.01p\tanh\left(\frac{R}{\max(1,N)}\right)
+\beta C\operatorname{clip}\left(1-\frac{t}{T},0,1\right).$$

**This continuation uses beta = 0.** Thus no additional completion-speed bonus is active. For its configured length-dependent timeout,

$$T=\min(120,\max(45,L/0.10))\quad\text{seconds},$$

where L is route length in metres. The original parent run instead had an explicit 60 s override; do not confuse the two conditions. See [fitness.py:71][fitness], [config.py:127][timeout], [continuation run settings][runrecord] and [parent settings][parentrecord].

### What is actually rewarded

| Quantity | Actual contribution |
|---|---|
| Authenticated forward progress | +p; cannot be collected repeatedly by oscillating |
| Completion | +1, in addition to reaching p=1 |
| Elapsed time | -0.1t/T; longer survival by itself is penalized |
| Measured chassis speed | Logged; no direct primary fitness term |
| Geometric lateral error | Logged; influences corridor validity and failure, not a continuous primary error penalty |
| Historical centering/speed/smoothness | Only through bounded secondary term, magnitude at most 0.01p |
| Action smoothness / action difference | Logged; no separate primary penalty |
| Normal failed episode | Keeps authenticated progress; no general extra fixed failure penalty |
| Controller/invalid-physics technical error recorded by Episode | Score overridden to -1; evaluator aborts on technical errors |

The legacy term is often saturated, so variations in its already-large raw return can supply almost no additional fitness gradient. A pivot or reverse is not prohibited by the primary score. It costs elapsed time, can hurt the legacy term and must respect the stall/corridor rules. The legacy term can only change the primary score by at most 0.01p either way. A careful, successful controller still scores far above a fast failure on the same route. See [episodes.py:171][result].

### Complete historical secondary scorer

For physical wheel-rim speeds uL,uR in m/s, the historical adapter forms hL=1000uL, hR=1000uR and

$$\nu=0.0165\frac{h_L+h_R}{2},\quad
\alpha=\frac{180}{\pi}\frac{(h_R-h_L)0.0165}{1000b},\quad
A=|\nu|,\quad c=o_6+o_7+o_8,\quad m=0.35/2=0.175.$$

These derived quantities deliberately reproduce old arithmetic. **They are not calibrated SI speed or physical yaw rate.** In particular, nu is 16.5 times mean rim speed; alpha is the historical angular quantity. The code's comparison of nu to m retains its documented unit mismatch. This is a legacy-reference limitation, not an identified MuJoCo dynamics error. See [fitness.py:15][legacy].

For an ordinary step, define:

$$V=A\begin{cases}10&c\ge2\\0.5&c<2\end{cases},\quad
V\leftarrow V/2\ \text{if }|\alpha|>20,$$

$$B_m=50\,\mathbf1[o_7=1],\quad B_c=2c^{1.5}\,\mathbf1[\nu>3],\quad
B_s=0.5\max(0,2-|h_L-h_R|),$$

$$O=\begin{cases}50+50k&\sum_i o_i=0\\0&\text{otherwise},\end{cases}\quad
P_v=\begin{cases}
1000(A+0.05)&\nu\le0\\
-100(m-A)&0<\nu<m\\
-100(A-m)&\nu\ge m,
\end{cases}\quad
P_\alpha=\min(0.8|\alpha|,30).$$

Here k is the number of consecutive all-zero steps already recorded before this step; it increases on loss and resets on an ordinary step with an active sensor. The ordinary legacy increment is

$$r=V+B_m+B_c+B_s-O-P_v-P_\alpha.$$

On the final completed step, the increment is instead **10000+100A**. On an `off_route` terminal step it is **-500**. Other terminal conditions have no corresponding special red/green override. The historical yellow-zone expression exists in `LegacyScorer`, but the current episode loop never supplies yellow; it contributes zero in this pipeline. R is the sum of the post-step increments. The unusual negative `speed_penalty` branches are subtracted exactly as written. See [fitness.py:32][legacyupdate] and [episodes.py:125][accounting].

### Progress and recovery rules

Progress projects the **axle-centered chassis position**, not the front sensor board, onto an ordered local interval. Gates occur every **0.04 m**, plus every explicit corner and the finish. Gate acceptance radius is **0.035 m**, with a 3 mm projected-distance tolerance. A short 0.06 m recovery tail is included in projection. Progress only advances while inside a **0.045 m half-width corridor** and is monotonically retained. Nearby branches, parallel sections or crossings cannot simply award future progress. See [fitness.py:103][route].

Termination occurs on completion; tipping (upright cosine <0.5 or chassis below floor); persistent corridor departure for **one second**; no new **10 mm of authenticated progress for five seconds**; timeout; or a technical invalid-state/controller error. Completion requires all gates, including corner gates. A sensor-all-zero event is logged but does **not** directly terminate the episode, which allows intentional gaps. These rules restrict a wide recovery manoeuvre even if the sensor board is still seeing paint. See [episodes.py:154][termination].
A replay of the selected policy's minimal zigzag explains why visible paint is not equivalent to route validity. At **4.683 s**, the axle genuinely leaves the corridor: global nearest-route distance is **45.53 mm**, just beyond the 45 mm bound. Progress freezes at **0.95255 m**, with the next gate at 0.960 m. By **5.583 s**, the axle is only **22.26 mm** from a later route segment, but its error relative to the authenticated local route window is **98.70 mm**. It has not returned to the missed gate interval, so the departure timer continues. The trace has **zero all-zero sensor events** throughout this failed episode. This confirms both an actual initial departure and a restrictive ordered-route recovery requirement; it does not demonstrate a spurious original departure or justify skipping gates. See [gate-probe.json][gateprobe].

### Numerical comparisons

For the following illustrative calculations, T=45 s, beta=0 and R=0 so the secondary term is exactly zero:

| Behaviour | Progress p | Completion C | Time (s) | Fitness |
|---|---:|---:|---:|---:|
| Slow successful traversal | 1.00 | 1 | 30 | 1.93333 |
| Fast failure halfway | 0.50 | 0 | 3 | 0.49333 |
| Fast failure near the finish | 0.95 | 0 | 5 | 0.93889 |
| Stationary until five-second stall | 0 | 0 | 5 | -0.01111 |

With actual legacy returns in the measured 1.65 m sharp-zigzag fixture, fixed slow commands complete in **18.383 s**, score **1.94915**; the selected NEAT policy fails at progress **0.57730** in **5.667 s**, score **0.57048**; the generation-167 policy fails at progress **0.37297** in **3.117 s**, score **0.36231**. No faster-prefix reward outweighs completing that same episode. See [physics.json][physics], [reproduction.json][reproduction] and [calculated examples][additional].

Nevertheless, a genome can rank well relative to a weak population while consistently failing corners. At fixed progress, failing sooner slightly improves the time term. Improving lateral alignment without gaining progress may change only the tiny saturated secondary term. Slowing before a corner has an incentive **through subsequent progress/completion**, not through an explicit corner-speed target or reward.

### Ten-episode aggregation

Each genome's evolutionary fitness is

$$J=0.5\left(\frac1{10}\sum_{i=1}^{10}F_i\right)+0.5\min_iF_i.$$

With a unique worst episode, that episode has weight **0.55** and each other episode **0.05** in the local linear expression. A newly sampled very difficult route can dominate ranking, even if another manoeuvre improves elsewhere. Five scores of 1.93333 and five of 0.49333 yield **J=0.85333**, while succeeding on all ten at 1.93333 yields J=1.93333. Thus the objective still prefers comprehensive success, but can give weak local guidance when every genome fails early on the hardest route.

The optional mean/CVaR aggregator is implemented but **not active**. With alpha=0.3 and ten episodes it would replace the minimum by the mean of the three worst scores. Its benefit here remains untested. See [fitness.py:86][aggregate] and [experiments.py:531][selectioncode].



### Source references

The source links below are the pre-PID implementation inspected for the NEAT diagnosis. Source ZIPs and hashes remain authoritative if later tools add files. The equations retain documented historical arithmetic rather than retrospectively correcting the experiment.

[provenance]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/provenance.json>
[records]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/records.json>
[selection]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/selection.json>
[physics]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/physics.json>
[reproduction]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/reproduction.json>
[additional]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/additional-checks.json>
[gpu]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/gpu.json>
[recovery]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/recovery-compaction.json>
[fixtures]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/fixtures>
[trace167]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/traces/champion-0167.pkl-single_right_angle.csv>
[trajectoryfigure]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/trajectories.png>
[sensorfigure]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/corner-sensors-actions.png>
[launches]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/28-10-2026-hard/launches.csv>
[generations]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/28-10-2026-hard/generations.csv>
[curriculumlog]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/28-10-2026-hard/curriculum.csv>
[robotrecord]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/28-10-2026-hard/robot.json:1>
[runrecord]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/28-10-2026-hard/run.json:1>
[parentrecord]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/28-10-2026/run.json:1>
[controller]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/controllers.py:82>
[action]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/controllers.py:96>
[rays]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/simulation.py:22>
[locations]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/simulation.py:43>
[xmlbuilder]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/simulation.py:61>
[actuators]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/simulation.py:83>
[step]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/simulation.py:181>
[readsensors]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/simulation.py:196>
[reset]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/simulation.py:160>
[reflectance]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/tracks.py:476>
[mask]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/tracks.py:420>
[motifs]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/tracks.py:300>
[zigzag]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/tracks.py:316>
[packages]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/robot_design.py:196>
[arccenter]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/config.py:77>
[timeout]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/config.py:127>
[mjcf]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/assets/robot.xml:36>
[fitness]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/fitness.py:71>
[legacy]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/fitness.py:15>
[legacyupdate]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/fitness.py:32>
[aggregate]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/fitness.py:86>
[route]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/fitness.py:103>
[accounting]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/episodes.py:125>
[termination]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/episodes.py:154>
[result]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/episodes.py:171>
[observe]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/episodes.py:110>
[batchdifficulty]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/suites.py:116>
[trainingbatch]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/suites.py:140>
[focus]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/suites.py:165>
[curriculum]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/curriculum.py:53>
[generationloop]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/experiments.py:469>
[selectioncode]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/experiments.py:531>
[advance]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/experiments.py:565>
[restore]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/experiments.py:294>
[train]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/experiments.py:377>
[gpuevaluator]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/gpu_backend.py:43>
[gpubuffer]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/gpu_backend.py:83>
[gpuloop]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/line_follower/gpu_backend.py:132>
[neatconfig]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/28-10-2026-hard/neat-config.txt:10>
[oracle]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/benchmark_training.py:34>
[activations]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/.venv/Lib/site-packages/neat/activations.py:11>
[attributes]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/.venv/Lib/site-packages/neat/attributes.py:65>
[reproductionimpl]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/.venv/Lib/site-packages/neat/reproduction.py:140>
[stagnation]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/.venv/Lib/site-packages/neat/stagnation.py:32>
[feedforward]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/.venv/Lib/site-packages/neat/nn/feed_forward.py:11>

[gateprobe]: <C:/Users/yazan/OneDrive/Desktop/Yazan Stuff/Coding Projects/NEAT Line Follower/runs/diagnostics/zigzag-20261029/gate-probe.json>


## 8. Subsequent frozen common assessment

After the historical NEAT phase was closed and archived, the unchanged generation-109 model was assessed alongside finalized PID policies on the same 90 held-out tracks. Its artifact hash was frozen before PID tuning; no NEAT optimization occurred. All methods used CPU MuJoCo, the continuation's unchanged 0.35 m/s robot and the same length-based episode rules. This is a **new fixed-policy test assessment**, separate from the historical GPU validation and the post-hoc CPU validation replay.

The frozen model completed **35/90** tracks, with mean observed lateral RMSE **26.525 mm**, directly measured MAE **18.170 mm**, mean progress **0.5714**, and **55 off-route failures**. It completed 30/30 beginner, 2/30 intermediate and 3/30 advanced. None of the 40 routes containing zigzags was completed. No model was reselected against these outcomes. Source data: runs/pid-research-20261029/test-episodes.csv; protocol/freeze and paired conditions are documented in Comparative_Analysis.md.

The 90 tracks were untouched at NEAT phase closure and were consumed only after all current controllers were frozen. Future model development must not use these now-known outcomes as an optimization signal; a new confirmatory cohort may be needed. This test result does not constitute a new independent evolution run or establish algorithm-wide superiority/inferiority.

<!-- FROZEN-NEAT-NEW90 -->
## Subsequent frozen NEAT assessment on the new cohort — 2 November 2026

| Frozen policy | Completed / 90 | Rate | RMSE (mm) | MAE (mm) | Mean progress |
|---|---:|---:|---:|---:|---:|
| Filtered PID | 51/90 | 56.67% | 30.108 | 17.712 | 0.7602 |
| PID + corner supervision, original gains | 78/90 | 86.67% | 8.606 | 5.593 | 0.9218 |
| PID + corner supervision, tuned gains | 84/90 | 93.33% | 6.654 | 4.479 | 0.9670 |
| Frozen NEAT g109 | 38/90 | 42.22% | 23.194 | 15.308 | 0.6814 |

| Frozen policy | Beginner | Intermediate | Advanced |
|---|---:|---:|---:|
| Filtered PID | 30/30 | 11/30 | 10/30 |
| PID + corner supervision, original gains | 30/30 | 29/30 | 19/30 |
| PID + corner supervision, tuned gains | 30/30 | 29/30 | 25/30 |
| Frozen NEAT g109 | 30/30 | 5/30 | 3/30 |

All rows use identical new-cohort track hashes, CPU physics and physical/episode limits. NEAT g109 is unchanged; this adds inference results, not training or reselection. The original 90-track comparison remains separate. See [NEAT_New_Cohort_Evaluation.md](NEAT_New_Cohort_Evaluation.md) for model/suite hashes, failures, paired comparisons and limitations.
