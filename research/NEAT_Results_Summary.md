# NEAT results summary

29 October 2026; preserved experimental baseline, not an algorithm-wide conclusion.

**Scope:** one evolution lineage (seed 0), 15 instantaneous binary sensors, feedforward NEAT, two evolved motor outputs, MuJoCo contact dynamics. Original phase: 100 generations (0-99), four tracks/genome, 0.30 m/s, 60 s timeout. Continuation: 74 completed generations (100-173), ten tracks/genome, adaptive progressive tracks, 0.35 m/s, length-based 45-120 s limits. These phases have separate learning curves and are not independent repetitions.

**Selected models:** original generation 38 and continuation generation 109 each completed 16/30 repeatedly used validation tracks: 10/10 beginner, 3/10 intermediate, 3/10 advanced. Historical GPU mean observed lateral RMSE: 24.987 mm and 17.742 mm. Different settings and repeated selection prevent attributing that difference to a controlled improvement. Historical MAE was unavailable; separately labelled CPU replays measured 14.780 mm and 12.084 mm, with 16/30 completion in both cases.

**Sharp-turn result:** the g109 policy passes an isolated 90-degree turn but fails consecutive alternating corners; g167 fails at the first sharp corner. The original phase has 0/24054 completions on 120 zigzag-containing training tracks; the finalized continuation has 0/83226 on 417 tracks. The earlier diagnostic 0/75,710 on 379 tracks is verified for generations 100-167. These are dependent evaluations, and mixed-route failure may precede the zigzag.

**Mechanism evidence:** g167 ignores the outer left sensor and steers the wrong way on that cue and all-zero input. Selected policies' sigmoid outputs cannot command reverse/pivot, although motors accept signed actions. Stateless sensing lacks last-seen direction. Curriculum advanced at 1.19% and 7.61% intermediate elite completion via plateau rules. Every non-elite offspring weight is perturbed or resampled; its causal effect remains untested. Fitness strongly prefers a slow completion over a fast failure but has weak continuous corner shaping.

**Physical/backend checks:** fixed timed commands complete the minimized zigzag in 18.383 s; a ground-truth oracle completes actual mixed routes. These are engineering probes, not deployable sensor-policy successes. Ten CPU/GPU comparisons matched outcomes; short identical-action differences were below 6 micrometres. No GPU bug explaining the failure was demonstrated.

**Resources/integrity:** retained generation wall-time sums are 9.513 h and 8.405 h, including overhead/possible contention, not active-GPU benchmarks. A concurrent resume duplicated generation-167 records; raw files are preserved and logical exclusions audited. Source archives/hashes identify versions, with no Git commit. Checkpoint after g173 is preserved; unfinished g174 is excluded.

**Limits and next step:** one uncalibrated robot/generator and one independent seed; no faults, disturbances, hardware transfer or matched NEAT/PID/PPO study yet. NEAT development is stopped to establish PID and potentially PPO baselines. The 90 test tracks were untouched at NEAT phase closure. No universal superiority/inferiority or statistical significance is claimed.

Details: [NEAT_Experimental_Results.md](NEAT_Experimental_Results.md); data and figures under artifacts/neat; archive and hashes under archive/neat-20261029.

**Later frozen common test:** unchanged g109 completed 35/90 (30/30 B, 2/30 I, 3/30 A), RMSE 26.525 mm and MAE 18.170 mm on CPU under the continuation's settings. This new test is separate from historical GPU validation; no NEAT training or reselection followed. See Comparative_Analysis.md.

<!-- FROZEN-NEAT-NEW90 -->
**New matched cohort:** frozen g109 completed 38/90; tuned supervised PID 84/90. Inference only; see [evaluation](NEAT_New_Cohort_Evaluation.md) (2 November 2026).
