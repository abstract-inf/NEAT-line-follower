# Frozen NEAT generation 134 versus retuned filtered PID at the 1.00 m/s cap

> **Methodological correction, 19 November 2026:** PID's saved `base = 0.75` restricted its straight-line request to 0.75 m/s. The recorded 9.37% finish-time difference is a comparison against that restricted-cruise configuration; it does not establish an advantage over a full-speed-capable PID. The [separate full-cruise protocol](PID_Fullspeed_100_Protocol.md) sets PID base to 1.00, verifies near-1 m/s travel, and requires a fresh comparison. The historical measurements and frozen files below remain unchanged.

The [corrected full-cruise comparison is now recorded separately](NEAT134_vs_Fullspeed_PID_100_Results.md), with new PID tuning, independent test geometry, and canonical frozen files.

Recorded 19 November 2026. This is a fresh comparison of two fixed resulting controllers. The NEAT checkpoint was identified through earlier validation and user replay; the PID was selected in its separate smoothness refinement. Neither was tuned on this comparison's outcomes.

## Results

NEAT had shorter shared-course finish times and lower overall shared-finish tracking error. PID completed two additional courses and had substantially smoother commands and lower measured steady-section yaw oscillation. The completion difference is not statistically resolved by the exact paired test on this suite (p=0.500); the shared-finish speed interval excludes zero. These findings describe this fixed pair of controllers.

| Courses | NEAT generation 134 | Retuned filtered PID |
| --- | ---: | ---: |
| all | 148/150 | 150/150 |
| mixed | 90/90 | 90/90 |
| speed | 29/30 | 30/30 |
| corner | 29/30 | 30/30 |

On the **148 courses both controllers finished**, mean completion time was **5.546 s for NEAT** and **6.119 s for PID**. NEAT's relative finish-time change was **-9.37%**, with a stratified paired-bootstrap 95% interval of **[-10.63%, -7.92%]**. Negative values indicate shorter NEAT finish times. NEAT was faster on 108 shared courses, PID on 40, with 0 ties.

NEAT alone finished 0 courses; PID alone finished 2; both failed 0. The completion-rate difference, NEAT minus PID, was **-1.33 percentage points**, with a paired-bootstrap 95% interval of **[-3.33, +0.00] percentage points**. The exact two-sided McNemar test gives **p=0.5000**. This describes completion evidence for these fixed controllers on the generated track mixture.

| Measure | NEAT | PID |
| --- | ---: | ---: |
| Mean route progress, all episodes | 99.41% | 100.00% |
| Geometric RMSE, all episodes (mm) | 17.631 | 23.529 |
| Geometric RMSE, shared finishes (mm) | 17.118 | 23.517 |
| Command jitter, all episodes | 0.007462 | 0.000927 |
| Command jitter, shared finishes | 0.007485 | 0.000928 |
| Mean episode forward speed, all episodes (m/s) | 0.707 | 0.634 |
| Peak forward chassis speed (m/s) | 1.002 | 0.748 |
| Mean sustained dedicated-straight speed (m/s) | 0.972 | 0.743 |
| Minimum sustained dedicated-straight speed (m/s) | 0.972 | 0.743 |
| Dedicated straight segments measured / expected | 9/9 | 9/9 |
| Steady-section yaw oscillation RMS (rad/s) | 0.052995 | 0.005724 |

Failure reasons: NEAT **{"stalled": 2}**; PID **{}**. Every failed attempt remains in the completion, progress, and all-episode quality summaries. Finish-time comparisons use shared successes only.

The trade-off depends on the course group. Each row below uses only that group's shared successful courses:

| Group | Shared finishes | NEAT mean finish (s) | PID mean finish (s) | NEAT geometric RMSE (mm) | PID geometric RMSE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed | 90 | 5.946 | 6.446 | 13.134 | 26.347 |
| speed | 29 | 7.074 | 8.887 | 3.194 | 2.999 |
| corner | 29 | 2.774 | 2.334 | 43.410 | 35.252 |

NEAT was quicker and more precise on the mixed-course group, and quicker on the speed-course group. PID was quicker and more precise on the shared corner drills. PID also had slightly lower geometric RMSE on the shared speed courses. The overall averages should be read alongside these group-specific results.

## What was held constant

Both policies were copied into a frozen folder and fingerprinted before this suite was generated. They used the same MuJoCo 4.0.0 robot, 15 physical binary sensors, 60 Hz control, 600 Hz physics, actuator limits, starting pose per track, route evaluator, and timeout formula. Off-route termination requires two seconds outside the 100 mm corridor without sensor contact; the existing five-second stall check remains active. NEAT-Python 0.92 and each policy's existing recovery behavior are unchanged.

The suite contains **150 fresh tracks: 90 mixed (30 per difficulty), 30 speed courses, and 30 corner drills**. Its geometry is disjoint from saved prior suites and training batches, including the PID refinement's development/validation/test tracks. The original five-seed NEAT study's final test set was not evaluated.

The speed courses comprise nine dedicated straights, nine gentle-curve courses, six broad S-curves, and six straight-to-turn courses. Their straight lengths are 4–6 m, gentle radii 1–3 m, line width 20 mm, and floor sampling 2 mm. Mixed and corner courses retain their existing generator settings. All geometries are validated and hashed.

This is an **equal wheel-speed-cap comparison**, at 1.00 m/s. PID retains base command **0.75** and its saved slowdown behavior; NEAT retains its learned throttle. Actual speed is measured and reported separately. Differences in finish time therefore include the policies' throttle choices. A comparison against a PID with a separately optimized higher cruise command would be a different experiment.

The cap limits requested wheel-rim speed targets, rather than imposing an exact ceiling on measured chassis velocity. A small transient chassis-speed overshoot can occur in the actuator dynamics.

## Metric and uncertainty definitions

Geometric RMSE is calculated per episode from chassis-center distance to the intended route polyline, then averaged across episodes. Command jitter measures normalized squared changes in the two wheel commands; it is distinct from visible chassis vibration. Mean forward speed is an unweighted mean of episode mean forward speeds, with failures included.

Sustained straight speed uses the central 2 m of each dedicated straight. The minimum and number of measured versus expected segments are reported to expose incomplete measurements. Physical yaw oscillation uses the RMS residual after subtracting a centered 15-sample moving average at 60 Hz, in predefined constant-radius or long-straight windows. Only sufficiently long windows are used, and completed speed courses must contain every expected window. The comparison uses **29 shared successful speed courses**.

Confidence intervals use 20,000 deterministic paired bootstrap resamples, stratified by course group and difficulty. Completion uses all 150 paired outcomes; time uses the shared successful tracks. Wilson intervals for individual completion rates are available in the structured summary. These intervals concern generated-track variation for this fixed pair. They do not estimate variation across independently evolved NEAT seeds.

## Interpretation limits

This comparison measures completion, speed, tracking accuracy, and smoothness separately. A speed advantage does not establish superiority on every quality measure. The selected NEAT belongs to one lineage, so algorithm-wide conclusions require replication across independently trained seeds. Different sensor preprocessing, recovery policies, training histories, and throttle choices are part of the resulting controllers being compared.

The policies remain frozen after this assessment. Any further tuning should use a separately declared experiment and fresh evaluation data.

## Frozen files for future use

The canonical folder is **`runs/neat134-vs-pid100-20261119/frozen/`**.

| Policy | Frozen filename | SHA-256 |
| --- | --- | --- |
| neat | `neat-generation-0134.pkl` | `6114a6859544fec99c26dad124d7ac44d9b0e077a43102532f5fca02114b4314` |
| pid | `pid-filtered-100.json` | `607416a046855ce013fbf25bbbdbb1829acbfe061e34f133a85f5decb282202c` |

- [Frozen NEAT genome, network configuration, and metadata](../runs/neat134-vs-pid100-20261119/frozen/neat-generation-0134.pkl).
- [Frozen PID parameters](../runs/neat134-vs-pid100-20261119/frozen/pid-filtered-100.json).
- [Freeze manifest with original source paths, hashes, and software fingerprints](../runs/neat134-vs-pid100-20261119/frozen/manifest.json).
- [Shared robot configuration](../runs/neat134-vs-pid100-20261119/frozen/robot.json) and [episode rules](../runs/neat134-vs-pid100-20261119/frozen/episode.json).
- [NEAT training provenance](../runs/neat134-vs-pid100-20261119/frozen/neat-training-run.json), [comparison protocol](../runs/neat134-vs-pid100-20261119/protocol.json), and [source snapshot](../runs/neat134-vs-pid100-20261119/source.zip).

The source controllers were `C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\runs\neat-fast-study\seed-01\training\champions\generation-0134.pkl` and `C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\runs\pid-100-smoothness-20261119\tuned-1.00.json`. Original files and earlier study inputs remain unchanged. Evaluation and replay load the canonical frozen copies and verify their hashes.

## Replay both controllers on ten tracks

The ten tracks were chosen by fixed indices before outcomes were observed: four mixed courses, one straight, one gentle curve, one broad S-curve, one straight-to-turn course, and two opposite-direction corner drills.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py replay_neat_pid_100.py
```

The viewer starts with frozen NEAT generation 134. **C** switches between that genome and the **exact frozen tuned recovery PID**, resetting the same track. **N/P** selects the next/previous track, **R** restarts, **Space** pauses, and **T** toggles the route overlay. Both policies use the recorded 1.00 m/s cap and episode rules. The terminal names the active controller and reports each episode's outcome.

Use the dedicated replay entry point above. Its common display canvas supports the different arena sizes while retaining each original sensor-sampling mask. Replay tests check controller switching, all ten tracks, preservation of physical masks, and agreement of rendered versus headless episode dynamics. The benchmark driver and its original source fingerprint remain unchanged.

```powershell
# List the ten replay tracks without opening a window
& $py compare_neat_pid_100.py list

# Verify and reuse the saved comparison
& $py compare_neat_pid_100.py run

# Reproduce the same comparison in another output directory
& $py compare_neat_pid_100.py run --output runs/neat134-vs-pid100-reproduction

# Regenerate the report and figures from saved results
& $py runs/neat134-vs-pid100-20261119/analyze_results.py
```

## Exploratory long course

Both frozen policies were also replayed headlessly on the existing exploratory long course after selection.

| Policy | Completed | Progress | Finish time (s) | Mean forward speed (m/s) | Peak forward speed (m/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| neat | True | 100.0% | 26.233 | 0.796 | 0.983 |
| pid | True | 100.0% | 29.317 | 0.707 | 0.748 |


## Data and figures

- [Full structured summary and confidence intervals](../runs/neat134-vs-pid100-20261119/results/summary.json), [raw episodes](../runs/neat134-vs-pid100-20261119/results/episodes.csv), [paired per-track table](../runs/neat134-vs-pid100-20261119/results/per-track-comparison.csv), and [ten-track replay index](../runs/neat134-vs-pid100-20261119/results/replay-tracks.csv).
- [Primary comparison, PNG](../runs/neat134-vs-pid100-20261119/plots/01-completion-time-tracking-smoothness.png), [SVG](../runs/neat134-vs-pid100-20261119/plots/01-completion-time-tracking-smoothness.svg).
- [Paired finish times, PNG](../runs/neat134-vs-pid100-20261119/plots/02-paired-finish-times.png), [SVG](../runs/neat134-vs-pid100-20261119/plots/02-paired-finish-times.svg).
- [Paired speed and quality, PNG](../runs/neat134-vs-pid100-20261119/plots/03-paired-speed-and-quality.png), [SVG](../runs/neat134-vs-pid100-20261119/plots/03-paired-speed-and-quality.svg).
- [Ten replay trajectories, PNG](../runs/neat134-vs-pid100-20261119/plots/04-ten-replay-tracks.png), [SVG](../runs/neat134-vs-pid100-20261119/plots/04-ten-replay-tracks.svg).
- [Dedicated replay script](../replay_neat_pid_100.py) and [separate replay/analysis source archive](../runs/neat134-vs-pid100-20261119/replay-source.zip).
