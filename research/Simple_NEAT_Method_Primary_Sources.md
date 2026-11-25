# One compact NEAT pilot

Research recommendation, 25 November 2026. The user subsequently activated the methodology-development goal. The candidate scorer is implemented; learning performance and independent confirmation remain unvalidated. Authoritative pilot commitment and execution status are separate from this literature note.

## Recommendation

Use stock NEAT-Python 0.92 with completion priority and one speed/smoothness tradeoff. Preserve policy inputs, direct outputs, physics and operators. Remove the five simultaneous quality gates from **training fitness** for this new experiment. Keep their measurements and the historical failed results; this changes the research objective rather than fulfilling the old protected-quality specification.

For the same fixed batch of `m` episodes, let `C` be completed count, `p_i` ordered route progress in `[0,1]`, `L_i` route length (m), `t_i` successful completion time (s), `v_cap` wheel-speed cap (m/s), and `j_i` the existing normalized squared wheel-command-change metric. Define:

```text
U = mean(p_i)                                      if C < m
U = mean((L_i/(t_i*v_cap))/(1 + lambda*sqrt(j_i)))  if C = m
fitness = C + U/(1+U)
```

For finite nonnegative `U`, the residual lies in `[0,1)`: one additional completion always outranks any residual improvement. Failed early termination never becomes a successful speed measurement. This is an explicit project-designed scalar objective, not a published NEAT formula.

`sqrt(j)` measures RMS normalized command change per simulation step. Fix timestep and metric definition; it does **not** measure chassis vibration directly. At `sqrt(j)=0.1`, `lambda=1` discounts speed utility by 9.1%; `lambda=2` discounts it by 16.7%. These are comprehensible tradeoffs, not evidence that these coefficients are sufficient. The recorded prototype's jitter around 0.0075 implies RMS change around 0.087, so these penalties are mild.

Use only a small development comparison with identical initial populations, course batches and budgets across variants. The initial research suggestion was `lambda={0,1,2}`; before any new outcomes were consulted, the pilot design widened this to `lambda={0,1,10}` to include a stronger penalty (50% discount at RMS change 0.1). This is a declared development range, not a literature-derived recommendation. Freeze one coefficient and protocol before five independent confirmation runs from freshly initialized populations. This limited sensitivity check does not establish reward optimality. Keep curriculum schedules and group proportions explicit; the closeout documents a training/validation mixture mismatch that another scalar formula alone cannot repair.

## Why this is defensible, and its limits

[Deb, 2000, published journal article](https://www.sciencedirect.com/science/article/abs/pii/S0045782599003898) demonstrates penalty-parameter sensitivity and develops feasibility-first tournament comparisons. It supports making tradeoffs explicit; it neither validates this reward nor makes this stock-NEAT experiment Deb's algorithm.

[Official NEAT-Python 0.92 reproduction source](https://raw.githubusercontent.com/CodeReclaimers/neat-python/v0.92/neat/reproduction.py) shows species offspring allocation depends on normalized **mean scalar fitness**. Avoid calling rank transformations equivalent: even preserved individual ordering can change species allocation. This proposal keeps stock reproduction but deliberately changes its supplied numerical fitness. It retains ordinary NEAT hyperparameters and introduces one declared task coefficient.

The tradeoff is substantial: partial-batch speed/smoothness has no selection pressure, a slow policy can benefit from reduced command variation, and neither geometric accuracy nor yaw is directly optimized. The formula promises no feasible solution, minimum speed or PID-level smoothness. Keeping the original full quality specification would require constraints or additional tradeoffs; deleting coefficients cannot establish it.

## Confirmation and claims

Treat the existing **0.95 m/s in every required straight window** as a declared application requirement, with missing windows failing. Published sources do not supply this numerical limit. Report failures/counts, paired successful-lap time (s), geometric RMSE (mm), yaw RMS (rad/s), command-change RMS (normalized command/step), and straight speed (m/s), per seed. Compare smoothness directly with PID instead of inventing an unsupported acceptable jitter threshold.

Tune the complete PID on development data, with matching caps, observations, physics and stopping rules; freeze both controllers before new shared tests. All five runs count, including failures. Five seeds are a practical budget, not a journal sufficiency guarantee. [Agarwal et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) supports reporting uncertainty with few training runs; [Cawley and Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html) supports separating model selection from final evaluation. Consulted test suites are now development evidence. Track-level uncertainty cannot substitute for independent training variability.
