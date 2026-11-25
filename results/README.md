# Curated research results

This directory contains small, versioned result snapshots and selected frozen inputs, including a compact immutable prepared-input bundle for the current pilot. Raw run folders, changing checkpoints, videos and bulk historical archives remain excluded by the root `.gitignore`. A Git push does not back up those excluded files.

- [Smooth-driving archive audit, 24 November 2026](neat-smooth-study-v1/README.md): all 14 archived candidates, the five parents, frozen input hashes, raw audit episodes and PNG/SVG figures.
- [Completed 100-generation old-fitness pilot](neat-smooth-study-v1/pilot-old-100-generations/report.md): all champion acceptance decisions, speed/smoothness trade-offs, CSV/JSON evidence and PNG/SVG figures. No champion qualified; the smooth-fitness pilot remains a separate experiment.
- [Completed old-versus-smooth pilot comparison](neat-smooth-study-v1/pilot-comparison/report.md): both branches finished. Smooth fitness produced 84/100 champions passing the straight yaw/steering gates versus 0/100 with old fitness, but neither produced a fully qualifying model. Includes frozen illustrative models, paired metrics, reward diagnostics and three PNG/SVG figure pairs.
- [Quality-constrained speed development](neat-quality-speed-v1/README.md): prospective task-fitness revision, archived-data calibration, frozen-source bundle and matched PID baseline. Its new 50-generation development evolution is recorded separately; no final-test success is claimed.
- [Accepted simple-fitness finding](../research/Simple_NEAT_Method_Audit.md#simple-fitness-001--successful-continuation-without-direct-tracking-or-smoothness-penalties): verified completion/progress/speed-only continuation, selected frozen lambda-0 model, paired PID results, Endurance Track video and warm-start limitations. [Full 50-generation readout](../docs/neat-simple-pilot-50-results.md).

The full methodology and terminal commands are in [the smooth-study guide](../docs/neat-smooth-study.md). Git/backup instructions are in [the GitHub workflow](../docs/github-workflow.md).
