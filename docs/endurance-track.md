# Endurance Track

The user named the existing long course **Endurance Track** on 25 November 2026. Its historical name is Rhythm Changes (candidate 4). This is a display-name change: the original JSON, geometry and old results are preserved.

- Route length: 45.7738738649 m.
- Original track: `runs/rhythm-changes-case-study-20261121/track.json`.
- Geometry SHA-256: `77fb2a4f7caad1dd33e3594335e1639faf2d03f339e25694b4dc3744c9205d6a`.
- Layout includes long straights, broad curves, S-curves, sharp corners, U-turns and zigzags.
- It is a known development/showcase course, not an untouched final test.

## Accepted simple-fitness controller

The requested replay uses the validation-selected lambda-0 generation-44/rank-0 controller (genome 4323), fixed before the course run. This is the selected simple-fitness model, not a controller chosen retrospectively for this course.

Frozen study model: `results/neat-simple-pilot-50-v1/accepted-development/neat-lambda0-g44-r0.pkl`.

File SHA-256: `9a04af4033ce75ab745422ee1183bb089756b92c5aab1f47bba7cf1cd067208e`.

Showcase run: `runs/endurance-simple-lambda0-20261125/`. It contains its own exact model/track copies, protocol, source copies, raw episode result, fitness verification and video sidecar. The original course JSON still contains its historical name; the video displays Endurance Track.

| Measurement | Result |
|---|---:|
| Wheel-speed cap | 1.00 m/s |
| Completion | 100%, completed |
| Completion time | 59.50 s |
| Mean chassis forward speed | 0.79791 m/s |
| Peak chassis forward speed | 0.99184 m/s |
| Geometric RMSE to nearest route segment | 25.738 mm |
| Normalized command jitter | 0.00851668 |
| Line-loss events / duration | 12 / 3.7333 s |
| Control / physics steps | 3,570 / 35,700 |

Mean chassis speed is step-weighted measured forward speed; it differs from route length divided by completion time. One headless evaluation and one verified rendering replay were performed, with the same outcome, time, progress, steps, speed and jitter. These are two executions of one deterministic case, not independent trials. No training, tuning, PID reevaluation or additional controller selection occurred. The 180 s timeout and all existing corridor, line-contact, stall and recovery behavior were retained.

Video: `runs/endurance-simple-lambda0-20261125/videos/endurance-neat-lambda0-g44.mp4`. It plays at 1x, 15 fps, with an approximately two-second final hold. The sidecar records 922 frames and the video SHA-256. Replay execution was checked against the saved headless episode before encoding.

```powershell
# From the repository root: reproduce missing outputs or verify saved outputs.
& .\.venv\Scripts\python.exe research/record_endurance_simple.py
```

## Fitness double-check

The authoritative frozen simple scorer uses `F = C + U/(1+U)`. With lambda = 0, U is mean progress for incomplete batches, otherwise mean `route_length / completion_time / cap`. The compatibility wrapper overwrites the historical diagnostic fitness, and the trainer assigns only this resulting fitness to `genome.fitness`.

Independent inspection checked the protocol, all 65 frozen source files, sealed parent sources/history and inherited data. It reproduced all 5,002 lambda-0 genome fitness records without a discrepancy and rescored all 150 validation candidate slots, confirming g44/rank0 as the own-objective selection. The full three-arm audit additionally reproduced all 15,002 scores without discrepancies.

For chosen genome 4323 at generation 44, the logged, independently calculated and frozen-scorer training fitness all equal **6.409444877656034**, from six completed courses with U = 0.6933220323801613. Setting valid jitter inputs to 0 versus 1 and RMSE fields to 0 versus 1,000 m, while also changing historical score/legacy fields, leaves this fitness exactly unchanged. The direct evidence is `runs/endurance-simple-lambda0-20261125/fitness-verification.json`.

The old per-episode `score`, component logs and historical quality-gate diagnostics can still include jitter/error calculations. They are not the selected lambda-0 breeding fitness. Their continued presence in raw logs does not mean those penalties were used for this evolution.

The warm start matters: the native g134 ancestor previously used an explicit `-0.20 * command_jitter` term and bounded legacy shaping. Its original logs show a mean jitter contribution of -0.001541140630412865 at generation 134. RMSE was used in its validation gate, not as a direct breeding reward. Thus the supported statement is that the **lambda-0 continuation** succeeded without direct jitter/RMSE penalties. The complete ancestry was not trained without smoothness shaping. This showcase does not establish from-scratch learning or independent-seed repeatability.
