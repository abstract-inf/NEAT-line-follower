# Offline quality-speed objective diagnostic

**Interim logged-horizon snapshot: generations 0–38 (39/50 logged).** Later progress is outside this captured evidence.

Conditional project proposal only; no new experiment is registered or activated. The frozen V1 run, thresholds, policy, training data and final acceptance rules are unchanged.

`Completed batch: F = 10*m + 9*S/(1+V)`. Incomplete rows retain their exact saved V1 fitness. Existing S and V are reused without coefficients or threshold changes.

Rescored **3900** genome rows; preserved 1020 incomplete scores. Excluded 0 captured rows outside the generation-log horizon. Top-one genome changed in **7/39** generations; the top-three genome set changed in **10/39**.

The original versus proposed top-one completed-quality-feasible counts are **39 versus 32**. Proposed top-one genome IDs lack archived validation for that generation in 5 cases. Feasibility of the stored measurements themselves never changes.

| Generation | Population | Completed quality-feasible | Top one, original → proposal | Top-three overlap | Feasible top one, original → proposal |
|---:|---:|---:|---|---:|---|
| 0 | 100 | 27 | 73 → 73 | 3/3 | True → True |
| 1 | 100 | 21 | 148 → 148 | 3/3 | True → True |
| 2 | 100 | 29 | 272 → 272 | 3/3 | True → True |
| 3 | 100 | 16 | 332 → 332 | 3/3 | True → True |
| 4 | 100 | 16 | 396 → 421 | 2/3 | True → False |
| 5 | 100 | 21 | 504 → 504 | 2/3 | True → True |
| 6 | 100 | 13 | 504 → 504 | 3/3 | True → True |
| 7 | 100 | 14 | 712 → 717 | 2/3 | True → False |
| 8 | 100 | 11 | 874 → 874 | 3/3 | True → True |
| 9 | 100 | 4 | 902 → 902 | 3/3 | True → True |
| 10 | 100 | 1 | 1056 → 1056 | 3/3 | True → True |
| 11 | 100 | 2 | 1102 → 1037 | 2/3 | True → False |
| 12 | 100 | 6 | 1193 → 1193 | 3/3 | True → True |
| 13 | 100 | 7 | 1193 → 1193 | 3/3 | True → True |
| 14 | 100 | 5 | 1193 → 1193 | 3/3 | True → True |
| 15 | 100 | 10 | 1525 → 1501 | 2/3 | True → False |
| 16 | 100 | 5 | 1193 → 1193 | 3/3 | True → True |
| 17 | 100 | 12 | 1755 → 1755 | 3/3 | True → True |
| 18 | 100 | 16 | 1814 → 1814 | 3/3 | True → True |
| 19 | 100 | 14 | 1814 → 1814 | 3/3 | True → True |
| 20 | 100 | 10 | 1814 → 1814 | 3/3 | True → True |
| 21 | 100 | 10 | 2078 → 2078 | 3/3 | True → True |
| 22 | 100 | 6 | 2078 → 2078 | 2/3 | True → True |
| 23 | 100 | 8 | 2078 → 2078 | 2/3 | True → True |
| 24 | 100 | 4 | 2078 → 2078 | 3/3 | True → True |
| 25 | 100 | 11 | 2078 → 2078 | 3/3 | True → True |
| 26 | 100 | 11 | 2440 → 2440 | 3/3 | True → True |
| 27 | 100 | 13 | 2440 → 2440 | 3/3 | True → True |
| 28 | 100 | 13 | 2440 → 2440 | 3/3 | True → True |
| 29 | 100 | 10 | 2440 → 2440 | 3/3 | True → True |
| 30 | 100 | 5 | 2914 → 2914 | 3/3 | True → True |
| 31 | 100 | 16 | 3096 → 3096 | 3/3 | True → True |
| 32 | 100 | 7 | 3158 → 3209 | 2/3 | True → False |
| 33 | 100 | 1 | 3298 → 3158 | 3/3 | True → False |
| 34 | 100 | 6 | 3158 → 3158 | 3/3 | True → True |
| 35 | 100 | 9 | 3410 → 3410 | 2/3 | True → True |
| 36 | 100 | 3 | 3621 → 3621 | 3/3 | True → True |
| 37 | 100 | 6 | 3621 → 3632 | 1/3 | True → False |
| 38 | 100 | 12 | 3778 → 3778 | 3/3 | True → True |

## Synthetic ordering

All seven executable checks pass. Completion stays dominant: every incomplete score is at most `10*C+9`, whereas a completed batch scores at least `10*m`, even with zero speed utility. At V=0, F depends only on S: improving already acceptable quality has no reward. At fixed positive V, increasing S still increases fitness. At fixed S, violation still reduces fitness. S remains the existing capped utility, so its existing saturation is retained.

| Synthetic completed batch | S | V | V1 score | Proposed score |
|---|---:|---:|---:|---:|
| slower feasible | 0.75 | 0 | 76.750000 | 66.750000 |
| faster small excess | 0.95 | 0.05 | 68.571429 | 68.142857 |
| faster larger excess | 0.95 | 0.5 | 66.000000 | 65.700000 |

The faster small-excess example can outrank the slower feasible example. That is the intended trade-off; strict feasible-first ordering is lost. Passing final quality and speed gates is still required.

## Archived 60-course calibration

Also rescored 200 stored calibration records without recomputing physical metrics.

| Proposal rank | Archived branch / generation | Original rank | Original score | Proposal score | Quality-feasible completed |
|---:|---|---:|---:|---:|---|
| 1 | smooth / 83 | 1 | 616.501658 | 606.501658 | True |
| 2 | smooth / 30 | 13 | 608.053364 | 606.353884 | False |
| 3 | smooth / 31 | 14 | 608.053364 | 606.353884 | False |
| 4 | smooth / 20 | 6 | 608.407778 | 606.295320 | False |
| 5 | smooth / 21 | 7 | 608.407778 | 606.295320 | False |
| 6 | smooth / 22 | 8 | 608.407778 | 606.295320 | False |
| 7 | smooth / 80 | 9 | 608.275025 | 606.231147 | False |
| 8 | smooth / 81 | 10 | 608.275025 | 606.231147 | False |
| 9 | smooth / 45 | 11 | 608.167639 | 606.169962 | False |
| 10 | smooth / 46 | 12 | 608.167639 | 606.169962 | False |

## Limits and reproduction

- These are counterfactual rankings of populations generated by V1. Different fitness would change future parents, species allocations and populations; no learning or final-performance improvement is demonstrated.
- Only the archived distinct-genotype top-three candidates were validated. Unsaved alternatives cannot acquire validation outcomes through rescoring. The CSV lacks genotype identities for all genomes, so reported top-three genome ranks are not reconstructed distinct-genotype validation selections. Same-generation archival coverage is conservative; clones or prior-generation cache matches are not inferred.
- Training quality-feasible means completed with V=0 on that six-course batch, not passing the 60-primary/20-retention final validation gates. Changing batch difficulty also prevents interpreting raw scores across generations as a fixed-suite learning curve.
- Calibration aggregates 60 courses, while breeding aggregates six. Absolute bases/bands differ (`600` versus `60`); never compare their numeric fitness directly or treat the calibration ranking as re-evolution evidence.
- Small excess can be rewarded for speed; substantial violation or missing-window penalties may still produce attractive counterfactual scores. Existing final gates are essential. No coefficient or threshold has been tuned by this diagnostic.

Exact captured input bytes, hashes and this executable source are preserved in `C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\results\neat-quality-speed-v1\objective-diagnostic\snapshots\g0038-5398ea01cea05ddb`. The root summary includes their SHA-256 values. Refreshing after the run finishes creates a separate immutable snapshot and updates this report.

Refresh from current saved logs:

```powershell
& .\.venv\Scripts\python.exe research/diagnose_quality_objective.py
```

Replay the exact snapshot into a separate output directory:

```powershell
& .\.venv\Scripts\python.exe "C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\results\neat-quality-speed-v1\objective-diagnostic\snapshots\g0038-5398ea01cea05ddb\diagnostic.py" --replay-snapshot "C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\results\neat-quality-speed-v1\objective-diagnostic\snapshots\g0038-5398ea01cea05ddb" --out ".cache/objective-diagnostic-replay"
```
