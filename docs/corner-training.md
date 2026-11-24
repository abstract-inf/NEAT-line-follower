# Fresh NEAT corner training

Use PowerShell in the project folder. The launcher creates a fresh, randomly initialized population and runs 100 generations, numbered 0 through 99.

```powershell
Set-Location -LiteralPath 'C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower'
.\.venv\Scripts\python.exe run_corner_training.py
```

Once the training terminal prints `Starting generation 0`, open a second PowerShell window in the same folder and attach the observer.

```powershell
.\.venv\Scripts\python.exe -m line_follower preview-training --run runs/neat-corner-100-20261104/training
```

The observer reads throttled snapshots. Close its window to detach. Press `N` to change the displayed track and `Space` to pause the display. Physics and selection run in the training process.

```powershell
.\.venv\Scripts\python.exe run_corner_training.py --dry-run
.\.venv\Scripts\python.exe run_corner_training.py --resume
```

The first command prints the protocol. The second resumes the last committed generation after an interruption or finishes an interrupted final assessment. Keep the same source, configuration, and dependencies for a resume. If training was interrupted before its first checkpoint, choose a new folder with `--output`.

## Training protocol

Each generation evaluates all genomes on the same four fresh tracks. Three slots give short practice on both left and right turns. The fourth slot exposes the population to progressively harder procedural courses, beginning with beginner routes in generations 0–19, intermediate in 20–54, and advanced in 55–99. The drill stage advances separately, based on the trailing four generations of training completion among the top 10% of the population.

| Drill stage | Three practice slots |
| --- | --- |
| 0 | Left and right single 35–50° corners, plus two alternating 45–60° corners |
| 1 | Left and right single 65–85° corners, plus two alternating 75–90° corners |
| 2 | Three alternating 90° corners in each starting direction, plus one isolated 90° corner |
| 3 | Three alternating 90° corners in each starting direction, plus a four-corner sequence |
| 4 | Three and four alternating 90° corners, with both starting directions represented |

The previous controller lost its steering cue when all sensors went dark and its sigmoid wheel outputs could only drive forward. The new feedforward network receives the same 15 physical readings plus six derived signals. These carry the current and previous line centroids, recent edge evidence, line-loss duration, and the previous forward and steering commands. Mirroring puts left and right cases in a common frame. The tanh outputs evolve forward and steering commands, which can produce a pivot or reverse. Controller state resets each episode. Route geometry and simulator state are available only to the evaluator. [Prior corner diagnostics](../runs/diagnostics/zigzag-20261029/report.md).

The NEAT settings start with no hidden nodes and allow topology to grow through speciation and structural mutation. The run preserves two elites per species and reduces destructive mutations. Weight replacement drops from 40% to 2.5%, with small perturbations on 22% of weights. The 96-genome target and eight CPU workers balance population diversity with runtime on this machine. The training score averages mean fitness and the mean of the two lowest scores, so multiple hard routes influence selection. These settings address weaknesses from the previous run and are a single proposed configuration. [Configuration](../line_follower/configs/neat-corner.txt) · [Original NEAT paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

## What the run saves

- `protocol.json` and `suite/` preserve the settings, provenance, and exact tracks. Training uses the validation and test hashes to keep new training geometry disjoint. Each of the 30 fixed validation tracks has the same role throughout the run.
- `training/` holds every training episode, all generation champion and validation results, species sizes, champion sizes, curriculum transitions, actual tracks, timings, generation champions, the selected `best.pkl`, and the last committed checkpoint. Validation completion rate selects the best model, with validation fitness as the tiebreaker.
- `assessment/` holds the frozen selected model, assessment manifest, and `test-episodes.csv`. The launcher evaluates five frozen controllers on each of the same 90 new test tracks, for 450 paired rows. It saves completion, termination, progress, time, lateral RMSE and MAE for all runs, plus sensor, action, and chassis traces on six tracks per controller.

The assessment uses the same CPU MuJoCo robot and episode settings for the new NEAT model, g109, filtered PID, and both supervised PID versions. The prior benchmark data remains available for historical comparisons. Exact historical replays use the source archived with each study.
