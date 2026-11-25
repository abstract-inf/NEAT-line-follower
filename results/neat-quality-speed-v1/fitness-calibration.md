# Quality-speed fitness calibration

The task objective was calibrated before its new evolution run using existing development episodes. This is not an independent performance result.

All synthetic ordering checks passed: additional completion dominates; acceptable quality plateaus; among feasible policies speed increases fitness; quality violations retain graded scores; missing windows and invalid geometry are rejected.

| Rank | Previous branch / generation | New scalar fitness | Quality-feasible completed batch | Minimum straight speed |
|---:|---|---:|---|---:|
| 1 | smooth / 83 | 616.501658 | True | 0.713 m/s |
| 2 | smooth / 51 | 608.531576 | False | 0.711 m/s |
| 3 | smooth / 53 | 608.531576 | False | 0.711 m/s |
| 4 | smooth / 74 | 608.434052 | False | 0.717 m/s |
| 5 | smooth / 50 | 608.432547 | False | 0.720 m/s |
| 6 | smooth / 20 | 608.407778 | False | 0.741 m/s |
| 7 | smooth / 21 | 608.407778 | False | 0.741 m/s |
| 8 | smooth / 22 | 608.407778 | False | 0.741 m/s |
| 9 | smooth / 80 | 608.275025 | False | 0.748 m/s |
| 10 | smooth / 81 | 608.275025 | False | 0.748 m/s |

The scores here aggregate 60 validation episodes, while breeding aggregates six training episodes. Their numeric score bands therefore differ; the ordering principle is the same. Smooth g83 is the only fully completed candidate satisfying these quality constraints among the 200 stored validation champions. Its selection is explicitly a development warm-start decision.

Limits, decomposition, archived inputs and hashes are recorded in `fitness-calibration.json`. The proposed task fitness preserves stock NEAT operators; it changes genome-level aggregation and must be described as a new experiment.
