# Implementation and research gates

The objective is a reproducible MuJoCo line-following environment and tools
that the user runs. Do not start additional NEAT evolution on the user's behalf.

- [x] Recover historical PID, NEAT interface, fitness variants and source units.
- [x] Record assumptions, manuscript/source discrepancies and attribution.
- [x] Build a contact-driven robot with two independent motors and a passive caster.
- [x] Redesign as a cohesive integrated PCB assembly inspired by the blue reference.
- [x] Keep all 15 registered sensor apertures; retain the historical geometry preset.
- [x] Expose sensor offset independently from the main chassis and compare 9/12/14/16 cm with a common tuned PID.
- [x] Generate ordered diverse tracks, explicit gaps/branches/crossings, raster sensing and geometry validation.
- [x] Provide one shared episode/metrics loop and MuJoCo viewer with reset/track switching.
- [x] Implement single-track, multi-track and curriculum scheduling and a frozen suite tool.
- [x] Implement user-run training, CPU multiprocessing, optional experimental GPU batching, checkpoints and exports.
- [x] Test physical motion, sensor alignment, progress exploits, scheduling, serial/parallel equivalence and fixed-network GPU/window operation.
- [ ] User selects final robot geometry and explicitly creates its configuration/MJCF freeze.
- [ ] User runs a small evolution and a real checkpoint continuation check before committing a large compute budget.
- [ ] Finalize main-study PID tuning, sample sizes, budgets, metrics and statistical protocol before test evaluation.
- [ ] User runs independent-seed NEAT experiments and common held-out/seen-track comparisons.
- [ ] Analyze actual outcomes, investigate failures and preserve all configurations/results.
- [ ] Only after explicit authorization: rewrite the manuscript as one coherent Overleaf-compatible LaTeX study.

The engineering pilot is not the final research comparison. The current
90 mm offset is a recommendation, not a silent final selection. All main
evolution tools require an explicit geometry freeze. No claim of NEAT superiority,
hardware calibration, significance or untested fault robustness is justified.
