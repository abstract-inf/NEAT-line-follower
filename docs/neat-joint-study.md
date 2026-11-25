# One joint speed/quality development attempt

This prospective follow-up has one arm, 50 generations, population target 100, six common training courses and top-three distinct-genotype validation per generation at the 1 m/s cap. No extension or additional arm is automatic. Preparation does not start evolution.

The completed continuous-reward study produced no qualifier. Its fast arm had speed-passing candidates with quality excesses; its smooth arm could improve quality by slowing. This experiment jointly changes the selected development parent, objective, and two mutation powers. It cannot isolate their causal effects or establish general superiority to filtered PID.

The parent rule examines all 150 saved fast-arm candidate decisions. Require all 60 primary and 20 retention completions, paired time at most 1.05, and every required primary straight observed and at least .95 m/s. Minimize the maximum positive relative excess in mean geometric RMSE/native-reference mean, mean jitter/native-reference mean, maximum yaw/.15 and maximum steering/.02; break ties by jitter, generation, then candidate rank. This reproduces fast generation 33 rank 2, genotype `064ad6554b353318b29b9d3690873597befd5040836a5104dfab4d69574795bf`. It remains quality-ineligible. The complete selection audit, source hashes, original model, selected cached episodes and native reference bytes are frozen.

The initialization template contains the same genome. Only standard config values `bias_mutate_power` .12→.03 and `weight_mutate_power` .14→.035 change. All rates, structural settings, operator implementations, sensor-history inputs, action mapping, robot and recovery remain unchanged. Preparation proves normalized genotype identity, equality of 100 deterministic sensor-history action steps, and that restoring those two config values recovers the original serialized configuration. One exact parent plus 99 stock mutants and the resulting Python RNG state are sealed before training.

The original V1 scorer still computes all measurements and five normalized quality violations. Define `quality_max=max(violations)` and `speed_shortfall=max(0,(.95-minimum_required_speed)/.05)`, then `joint_violation=max(quality_max,speed_shortfall)`. The fixed .05 scale is the 1 m/s reference cap minus its existing .95 straight target. Incomplete-batch fitness is exactly V1's. Completed batches receive `10*m+9/(1+joint_violation)` when violation is positive and `10*m+10+9*S` otherwise, with unchanged S. Thus quality-only slow driving does not become joint-feasible. Positive-violation scores are graded and continuous there; the joint-feasibility band retains a declared jump. Improving a non-dominant max component can leave fitness flat.

Genome logs preserve original sum violation/fitness and add `quality_only_feasible`, `quality_max_violation`, `speed_shortfall`, `joint_violation`, `joint_feasible`, and component labels. The inherited `quality_feasible` and generation `quality_feasible_genomes` remain quality-only measurements. Full raw training rows, all candidate decisions, models, deterministic validation cache and atomic population/RNG checkpoints are retained by the shared trainer, using explicit callbacks rather than changing module globals.

All 50 V1 course batches/reference rows and the previously consulted 60-primary/20-retention validation suite are copied byte-for-byte. Reference simulations are reused (300 slots, zero new episodes). Final completion, time, geometric RMSE, jitter, every-window speed/yaw/steering and retention gates and selection are unchanged. The selected parent and reward use development evidence, not an untouched test. No PID imitation, PID targets or threshold relaxation is introduced. No fresh test or confirmation is claimed.

Parent verification executes its frozen entry point because the live shared helper gains explicit callback parameters after the old runs completed. New preparation snapshots that helper, callback modules and all runtime files. Worker count is runtime-only; the planned launch uses six workers. Fresh initialization and resumable generation-zero checkpoints remain in the shared trainer.

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_neat_joint_study.py tests/test_quality_followup_training.py -q --basetemp .cache/jt
& .\.venv\Scripts\python.exe run_neat_joint_study.py prepare
& .\.venv\Scripts\python.exe run_neat_joint_study.py verify
# Starts the one arm only when explicitly launched:
& .\.venv\Scripts\python.exe run_neat_joint_study.py train --workers 6
# Interrupted runs retain their exact prepared inputs and checkpoint:
& .\.venv\Scripts\python.exe run_neat_joint_study.py train --workers 6 --resume
```
