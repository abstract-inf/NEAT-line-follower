# NEAT five-seed held-out assessment

The assessment used the same 90 held-out tracks for all 12 frozen policies at each speed cap.
The low and ramp models are paired by evolutionary seed. Means and sample SDs describe variation across five lineages on this shared suite.

## Held-out completion

| Wheel-speed cap | Stage | Seeds 1–5, each out of 90 | Mean | Sample SD | Frozen PID |
| --- | --- | --- | ---: | ---: | ---: |
| 0.50 m/s | Low-speed NEAT | 89, 87, 88, 89, 86 | 97.6% | 1.4 pp | 86/90 |
| 0.50 m/s | Ramp NEAT | 74, 82, 82, 74, 83 | 87.8% | 5.1 pp | 86/90 |
| 1.50 m/s | Low-speed NEAT | 16, 13, 27, 22, 27 | 23.3% | 7.1 pp | 47/90 |
| 1.50 m/s | Ramp NEAT | 88, 85, 86, 87, 87 | 96.2% | 1.3 pp | 47/90 |

### Ramp performance by seed

| Cap | Seed | Finished | Shared PID finishes | NEAT / PID mean time on shared finishes | Geometric RMSE | Command jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 m/s | 1 | 74 | 72 | 18.32 / 10.17 s | 19.7 mm | 0.0025 |
| 0.50 m/s | 2 | 82 | 81 | 19.49 / 10.78 s | 10.4 mm | 0.0040 |
| 0.50 m/s | 3 | 82 | 79 | 19.48 / 10.74 s | 12.7 mm | 0.0024 |
| 0.50 m/s | 4 | 74 | 72 | 18.22 / 10.03 s | 12.2 mm | 0.0035 |
| 0.50 m/s | 5 | 83 | 81 | 19.31 / 10.74 s | 8.8 mm | 0.0026 |
| 1.50 m/s | 1 | 88 | 47 | 5.03 / 4.41 s | 12.8 mm | 0.0036 |
| 1.50 m/s | 2 | 85 | 46 | 5.05 / 4.38 s | 7.7 mm | 0.0041 |
| 1.50 m/s | 3 | 86 | 47 | 5.03 / 4.41 s | 8.7 mm | 0.0035 |
| 1.50 m/s | 4 | 87 | 47 | 5.05 / 4.41 s | 8.3 mm | 0.0043 |
| 1.50 m/s | 5 | 87 | 46 | 5.05 / 4.38 s | 7.5 mm | 0.0045 |

## Interpretation

At 1.50 m/s, the ramp winners ranged from 85/90 to 88/90 completions. Leaving out one seed at a time gave four-seed mean rates from 95.8% to 96.7%. This describes seed sensitivity on the shared test cohort.

At that cap, pairs of ramp winners agreed on 84–88 of 90 track outcomes. Their median absolute time gaps on shared finishes ranged from 0.2% to 0.7%.

The low-speed winners had 23–35 reachable links, and the ramp winners had 9–14. 0 of the five ramp networks had a structural path from an input to the drive output.

The PID lines use the gain files frozen in this assessment protocol. At 1.50 m/s this is the earlier high-speed tune from 15 November. The later 1.00 and 1.50 m/s PID follow-up used its own held-out cohort and has separate results.

## Figure captions

1. **[Held-out completion](01-held-out-completion.svg).** A line links the two winners from the same seed. The dashed line gives the cap-specific frozen PID reference.
2. **[Ramp policy similarity](02-ramp-policy-similarity.svg).** The top row counts tracks with matching finish or failure outcomes. The bottom row gives the median absolute time difference divided by the two times' mean on shared finishes. Each cell lists its shared denominator. Color scales are shared across caps within each row.
3. **[Speed and tracking](03-speed-and-tracking.svg).** Time differences use each policy's matched finishes with PID. Negative values indicate shorter times. Geometric RMSE and normalized command jitter average all observed episodes.
4. **[Validation learning curves](04-validation-learning-curves.svg).** Every line shows one seed's generation champions. Speed labels apply to the shaded or white intervals beneath them.
5. **[Network input paths](05-network-input-paths.svg).** Color marks structural paths through nonzero links in the expressed feedforward network. The two right columns count reachable hidden nodes and links. The architecture describes the network's possible dependencies.
6. **[Ramp fitness across seeds](06-ramp-fitness-across-seeds.svg).** The upper panel averages population mean training fitness and the lower panel averages champion validation fitness. Shading shows one sample SD across the five seeds. Each cap uses its own line segment.

Figures are available as SVG and 300 dpi PNG.

Raw data directory: `C:\Users\yazan\OneDrive\Desktop\Yazan Stuff\Coding Projects\NEAT Line Follower\runs\neat-speed-study\assessment\results`.
