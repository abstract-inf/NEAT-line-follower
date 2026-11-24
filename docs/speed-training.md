# Warm start from g94 with a rising speed cap

From the project folder in PowerShell, run:

```powershell
.\.venv\Scripts\python.exe run_speed_training.py
```

For a live view, open another terminal in the same folder after the first generation starts:

```powershell
.\.venv\Scripts\python.exe -m line_follower preview-training --run runs/neat-speed-100-20261104/training
```

Use `--dry-run` on the training command to inspect the protocol. If interrupted after a checkpoint, use `--resume` on the same command. A different output folder can be chosen with `--output`. Point the viewer to that folder's `training` subdirectory.

The run starts with one exact copy of the frozen g94 controller and 99 independently mutated copies. It uses g94's saved NEAT settings with a population target of 100. NEAT reproduction can vary the actual population slightly, so the logs record the real counts.

| Generations | Wheel speed target cap |
| --- | ---: |
| 0–9 | 0.50 m/s |
| 10–24 | 0.65 m/s |
| 25–39 | 0.80 m/s |
| 40–54 | 1.00 m/s |
| 55–69 | 1.25 m/s |
| 70–99 | 1.50 m/s |

The generation schedule reserves 30 generations at 1.50 m/s. This is a cap on the commanded wheel speeds. The controller can slow down or reverse. The run uses the same robot, sensors, torque limits, and episode rules as the parent experiment. The population carries across each phase. Validation ranking and species stagnation history restart when the cap changes because scores from different caps measure different conditions.

Every generation uses four new tracks shared by all genomes. The mix is one beginner, one intermediate, and two advanced. One intermediate and one advanced slot guarantee zigzags. Model selection uses the same 30 fresh validation tracks at each cap. The final model is chosen from generations 70–99 by validation completion, with validation fitness breaking ties.

The run saves every episode, track, speed per generation, checkpoint, and champion. After training it freezes the final model and evaluates both it and the original g94 at **0.50 and 1.50 m/s** on the same 90 new test tracks. The 360 paired results include finish time, progress, tracking error, and measured forward speed. Selected routes include full telemetry.
