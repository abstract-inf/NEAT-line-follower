# Curated research results

Start here for the current accepted controller. Older studies remain available as research evidence below; their original paths are preserved because reports and frozen protocols depend on them.

## Current accepted result: compact lambda-0 fitness

| Find | Open |
|---|---|
| Main results and PID comparison | [Completed 50-generation readout](../docs/neat-simple-pilot-50-results.md) |
| Verified equation and paper finding | [SIMPLE-FITNESS-001](../research/Simple_NEAT_Method_Audit.md#simple-fitness-001--successful-continuation-without-direct-tracking-or-smoothness-penalties) |
| Selected generation-44 controller | [Frozen model](neat-simple-pilot-50-v1/accepted-development/neat-lambda0-g44-r0.pkl), [identity and metrics](neat-simple-pilot-50-v1/accepted-development/manifest.json) |
| Completion, speed and quality learning curves | [PNG](neat-simple-pilot-50-v1/learning-and-quality.png), [SVG](neat-simple-pilot-50-v1/learning-and-quality.svg) |
| Command changes versus measured body motion | [PNG](neat-simple-pilot-50-v1/command-motion-tradeoffs.png), [SVG](neat-simple-pilot-50-v1/command-motion-tradeoffs.svg) |
| All candidate metrics and paired denominators | [Numerical report](neat-simple-pilot-50-v1/report.json) |
| Endurance Track replay and terminal command | [Course and video guide](../docs/endurance-track.md) |
| Current automated checks | [Test index](../tests/README.md) |

The accepted controller completed 60/60 primary and 20/20 retention validation courses. This is warm-start development evidence; five independent runs have not started. The two simple-pilot snapshot folders now contain only their terminal captures. Superseded interim captures and temporary previews were removed after a verified push and remain recoverable from the backup tag.

## Earlier evidence

<details>
<summary>Open earlier experiments, failed attempts and comparison reports</summary>

- [Twenty-generation compact-fitness pilot](neat-simple-pilot-v1/report.md): the precursor retained for the matched continuation.
- [Earlier accepted NEAT prototype versus tuned PID](../docs/final-results-20261125.md): frozen models and the earlier fresh-track assessment.
- [PID full-speed tuning](../docs/pid-quality-tuning.md): cap-specific tuning outcomes and failures.
- [Joint development](../docs/neat-joint-study.md) and [quality follow-up](../docs/neat-quality-followup.md): earlier methodology and archived outcomes.

- [Smooth-driving archive audit, 24 November 2026](neat-smooth-study-v1/README.md): all 14 archived candidates, the five parents, frozen input hashes, raw audit episodes and PNG/SVG figures.
- [Completed 100-generation old-fitness pilot](neat-smooth-study-v1/pilot-old-100-generations/report.md): all champion acceptance decisions, speed/smoothness trade-offs, CSV/JSON evidence and PNG/SVG figures. No champion qualified; the smooth-fitness pilot remains a separate experiment.
- [Completed old-versus-smooth pilot comparison](neat-smooth-study-v1/pilot-comparison/report.md): both branches finished. Smooth fitness produced 84/100 champions passing the straight yaw/steering gates versus 0/100 with old fitness, but neither produced a fully qualifying model. Includes frozen illustrative models, paired metrics, reward diagnostics and three PNG/SVG figure pairs.
- [Quality-constrained speed development](neat-quality-speed-v1/README.md): prospective task-fitness revision, archived-data calibration, frozen-source bundle and matched PID baseline. Its new 50-generation development evolution is recorded separately; no final-test success is claimed.
- [Accepted simple-fitness finding](../research/Simple_NEAT_Method_Audit.md#simple-fitness-001--successful-continuation-without-direct-tracking-or-smoothness-penalties): verified completion/progress/speed-only continuation, selected frozen lambda-0 model, paired PID results, Endurance Track video and warm-start limitations. [Full 50-generation readout](../docs/neat-simple-pilot-50-results.md).

</details>

## Storage and cleanup

Raw `runs/`, videos, caches, the manuscript and other intentionally ignored content were not copied or uploaded during the Git backup. Those local files are needed for some replay and resume commands. The committed results include deliberately curated controller files and analysis captures that were already nonignored.

The pre-cleanup nonignored repository is preserved by tag **`backup/pre-cleanup-20261125`**, commit `43c474b3733adc2def83651ad5c0dec20b55438e`. See the [cleanup record and restoration command](../docs/repository-cleanup-20261125.md). Failed studies and required automated tests were retained; superseded captures and generated PDF previews were the disposable content.

The current methodology and commands are in [the simple-fitness guide](../docs/neat-simple-study.md). Earlier commands remain in [the smooth-study guide](../docs/neat-smooth-study.md). Git instructions are in [the GitHub workflow](../docs/github-workflow.md).
