# Experiment closeout — 25 November 2026

The current work is finalized. No further training or tuning is authorized by this closeout. All completed attempts, including failures, remain preserved. The full protected-quality target was **not** achieved; no official NEAT winner is claimed. The selected fast prototype was frozen for one exploratory fresh-track comparison rather than silently relaxing its failed gates.

## Fresh comparison

Controllers were frozen before generating 150 new shared geometries: 90 mixed (30 per difficulty), 30 speed/recovery and 30 corner courses. Both controllers used identical physics, sensors, stopping rules and caps. Two cap-specific full-base PID tunes were used. There were exactly 600 evaluation episodes and no test-based retraining or model selection.

| Cap | NEAT completion | PID completion | Shared finishes | NEAT / PID paired mean time |
|---:|---:|---:|---:|---:|
| 1.0 m/s | 148/150 | 141/150 | 140 | 5.729 / 6.710 s |
| 0.5 m/s | 145/150 | 148/150 | 144 | 9.473 / 9.305 s |

At 1 m/s the prototype was **14.6% faster** and had **21.2% lower mean geometric RMSE** on their 140 shared successful tracks. Its command jitter was **7.6 times higher**. It sustained at least 0.95018 m/s over every required fresh straight window; its worst measured straight yaw RMS was 0.13318 rad/s. Mean episode-average chassis speed was 0.7093 m/s, with peak 1.00245 m/s. A wheel target cap is not a compulsory chassis speed or an absolute instantaneous physical-speed bound.

PID had lower jitter at both caps, and better completion/time at 0.5 m/s. This is a comparison of the resulting controllers from one NEAT ancestry, not proof of general NEAT superiority or five-seed training repeatability. Paired errors/jitter and all failures are in the numerical report. All-episode observed means include partial failure exposure and must not substitute for shared-success comparisons.

## Protected validation gates

The frozen NEAT model is joint-stage generation 49, candidate rank 1, genotype `f520e5e3592fcd8563156ad1ab8a7def8554aa83c72f42997d8760410ec7c3fa`. It completed all 60 primary and 20 retention validation tracks, passed straight speed/yaw/steering and paired-time checks, but geometric RMSE was 16.4621 mm versus the protected native reference's 14.4928 mm, and command jitter 0.00752684 versus 0.00652988. The fresh results do not retroactively turn it into a qualifying winner.

## Training attempts retained

| Study | Generations | Candidate training episodes | New validation episodes | Outcome |
|---|---:|---:|---:|---|
| Quality-speed strict V1 | 50 | 30,000 | 7,840 | No qualifier; strongest speed-only near miss 0.8842 m/s |
| Continuous reward, smooth parent | 50 | 30,000 | 8,800 | No qualifier |
| Continuous reward, fast parent | 50 | 30,036 | 8,320 | No qualifier; 17 speed-passing records represented 11 genotypes |
| Joint constraints, finer mutations | 50 | 30,030 | 10,800 | No qualifier; 23 training-joint-feasible rows; validation error failed all 150 candidates |

These runs are related development attempts, not four independent seeds. The joint stage changed the objective, selected development parent and two standard mutation powers. NEAT-Python 0.92 operators and disconnected-node evaluation, policy interface, action mapping and physics remained unchanged. Source/input hashes, actual populations, course batches, all rejected candidates, raw episodes and resumable checkpoints remain in each run folder. Reused references/cached validation are separately counted in the ledger.

## PID tuning

The search evaluated 178 distinct candidates and 5,360 episode records per cap: **16,080 records total**, with zero evaluation errors. Base was fixed at 1.0 so full cruise was feasible. The 1.0 and 0.5 m/s selections completed 60/60 selection tracks plus the known 45.77 m Rhythm Changes development course. Their long-course times were 68.333 and 100.667 seconds, respectively. Dedicated speeds reached at least 0.990512 and 0.495256 m/s.

No eligible 1.5 m/s tune emerged: the best observed candidate completed the long course in 82.267 seconds and reached isolated speed 1.485768 m/s, but finished only 29/60 selection tracks. It was retained as a failed tuning outcome and was not relabeled as a validated controller.

The 1.0 m/s selected tune improved paired selection time by 5.30%, while yaw/jitter/RMSE worsened 17.47%/0.69%/1.53% against its saved control. At 0.5 m/s, yaw/jitter improved 4.94%/5.62%, while RMSE worsened 10.19% and time increased 0.11%. These are the frozen ranking's trade-offs, not a claim that every metric improved.

## Reproducible handoff

- [Fresh comparison, complete numerical data and figures](../results/neat-pid-finalization-20261125/report.md).
- [Experiment ledger](experiment-ledger.md), including every attempted protocol and terminal result.
- [PID search and exact commands](pid-quality-tuning.md).
- [Joint-stage method and resume commands](neat-joint-study.md).
- [Profile diagnostic](../results/neat-joint-study-v1/profile-diagnostic/report.md).
- Frozen handoff models: `results/neat-pid-finalization-20261125/frozen/neat-prototype.pkl`, `pid-100.json`, `pid-050.json`, and their manifest. Original identical copies remain under `runs/neat-pid-finalization-20261125/frozen/`.
- Raw fresh tracks/episodes and controller/test fingerprints: `runs/neat-pid-finalization-20261125/`.

Regenerate saved assessment results/figures without evolution:

```powershell
& .\.venv\Scripts\python.exe research/finalize_neat_pid.py --workers 6
& .\.venv\Scripts\python.exe research/write_final_closeout.py
```

The completed first-development archive and hash remain under `results/neat-quality-speed-v1/`; all 1,383 archived file hashes and its clean root launcher were verified. Later studies retain their own full source/input snapshots and checkpoints.

## Unfinished research and future work

Original strict error/jitter targets, independent-lineage confirmation and algorithm-wide claims remain unproven. The saved profile diagnostic identifies a plausible training mismatch: training weights speed/mixed/corner equally, while validation weights them 1/3, 1/2 and 1/6. For the selected prototype, reweighting its exact six training episodes changes error ratio from 1.0458 to 1.1477 and jitter ratio from 1.0962 to 1.1687, close to observed validation ratios 1.1359 and 1.1527. Corner gains can mask mixed-course deficits. This motivates a future preregistered group-protected or mixture-aligned training objective; **it has not been implemented or run**. Any later optimization must use development data and a new untouched test set, since this finalization suite has now been inspected.


## Practical goal accepted

After viewing the two verified Rhythm Changes videos, the user explicitly accepted this controller and tuned-PID comparison as the practical result they wanted. This closes the controller-development goal. It does not change the original failed validation gates, convert the prototype into a formal qualifying winner, or establish independent-lineage repeatability. Further research would be a separate task.
