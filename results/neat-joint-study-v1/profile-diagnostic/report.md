# Training/validation profile diagnostic

Saved data only. No controller, score, threshold, source or prepared input changed; no physics or evolution ran.

Focus: generation 49, rank 1, genome 4739, genotype `f520e5e3592fcd8563156ad1ab8a7def8554aa83c72f42997d8760410ec7c3fa`. It completes 60/60 primary and 20/20 retention; only geometric error and jitter fail validation. **It was not joint-feasible on its own training batch** (joint violation 0.09621509).

Training assigns equal episode weight: two speed, two mixed, two corner courses (1/3 each). Validation uses 20 speed, 30 mixed, 10 corner courses (1/3, 1/2, 1/6). Reweighting below changes neither episodes nor measurements.

| Focus profile | Geometric RMSE candidate / ref (mm) | Ratio | Jitter candidate / ref | Ratio |
|---|---:|---:|---:|---:|
| Actual equal training means | 21.5634 / 20.6185 | 1.04583 | 0.00812007 / 0.00740737 | 1.09622 |
| Same six courses, validation weights | 19.5986 / 17.0772 | 1.14765 | 0.00932066 / 0.00797499 | 1.16874 |
| Actual 60-course validation | 16.4621 / 14.4928 | 1.13588 | 0.00752684 / 0.00652988 | 1.15268 |

| Focus group | Training RMSE candidate / ref (mm; ratio) | Validation RMSE candidate / ref (mm; ratio) | Training jitter candidate / ref (ratio) | Validation jitter candidate / ref (ratio) |
|---|---|---|---|---|
| speed | 1.62364161 / 1.43007015 (1.13536) | 4.57437022 / 4.04781470 (1.13008) | 0.00088067 / 0.00185802 (0.47398) | 0.00119953 / 0.00195205 (0.61450) |
| mixed | 25.63886102 / 19.58877617 (1.30885) | 16.34458518 / 12.18328274 (1.34156) | 0.01534155 / 0.01188492 (1.29084) | 0.01084028 / 0.00869923 (1.24612) |
| corner | 37.42762102 / 40.83658118 (0.91652) | 40.58996909 / 42.31143482 (0.95931) | 0.00813798 / 0.00847917 (0.95976) | 0.01024112 / 0.00917748 (1.11590) |

The weighting change in candidate-minus-reference difference equals `(mixed difference - corner difference)/6`; speed weighting is identical. Positive shifts mean that extra corner weight made training look more favorable.

For the focus candidate the RMSE shift is **+1.5765 mm** and jitter shift **+0.00063297**.

| Other saved training-joint-feasible candidate | Primary / retention finishes | Equal-training RMSE / jitter ratios | Reweighted-training ratios | Actual-validation ratios |
|---|---|---|---|---|
| g16 r0 key1461 | 60/60 + 20/20 | 0.96415 / 0.98429 | 0.98711 / 1.05338 | 1.20883 / 1.18410 |
| g37 r0 key3589 | 60/60 + 20/20 | 0.98419 / 0.98422 | 1.02141 / 1.04184 | 1.21358 / 1.18962 |
| g41 r0 key4005 | 60/60 + 20/20 | 0.97292 / 0.99845 | 1.04815 / 1.05204 | 1.18798 / 1.27168 |

The archive contains 23 training-joint-feasible rows and 17 distinct such genotypes with same-generation saved validation. This note shows up to three, preferring complete primary validations. Exact six-slot values, per-group counts, all selected input hashes and helper hash are in diagnostic.json.

Interpretation: equal group weighting can hide group-specific deficits when corner gains offset mixed-course losses. The numerical tables test that mechanism for these saved cases. Reweighting two examples per group cannot predict a 60-course result: geometry, difficulty, reference errors and failure exposure differ as well. Ratios use paired candidate/reference means; they are not averages of per-track ratios. Training floors (5 mm and .0005 jitter) are not applied to these descriptive raw-reference ratios. Any incomplete comparison is observed-exposure only, not whole-lap equivalence. No unsaved alternative receives invented validation evidence.

Reproduce from the project root:

```powershell
& .\.venv\Scripts\python.exe results/neat-joint-study-v1/profile-diagnostic/analyze_profile.py --run runs/neat-joint-study-v1
```
