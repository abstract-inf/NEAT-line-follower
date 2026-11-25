# Constraint relaxation as a possible NEAT follow-up

Research-only note, **24 November 2026**. QUALITY-SPEED-V1 is still an ongoing, frozen **50-generation development experiment** when this note is written. No outcome is assumed. This note changes no source, protocol, threshold, controller selection, or running process; no training or simulation was performed for it.

## Can strict bands impede progress?

Yes, as a search mechanism; this is not a diagnosis of the ongoing run. The [frozen objective](../docs/neat-quality-speed-study.md) gives a completed six-course batch `60 + 9/(1+V)` when quality violation `V > 0`, but `70 + 9*S` when `V = 0`. Speed has no influence in the first case. A slightly violating, much faster policy therefore ranks below every fully quality-feasible policy, including a slow ancestor. The score jump at the boundary approaches `1 + 9*S` as violation approaches zero.

If reaching a faster, acceptable policy requires intermediate networks with temporary quality excess, that path can receive weak selection support. Small weight changes can also switch which required window supplies the maximum yaw or steering measurement. Maxima are continuous but nonsmooth; episode failures, missing windows, and feasibility bands add discontinuities. Gradient-free evolution can handle these features, but does not make them harmless.

These are mathematical properties of the objective, not evidence that the feasible policy region is actually disconnected or that the current population has stopped exploring. Stock NEAT still breeds within species, uses species mean fitness for offspring allocation, and can retain some infeasible candidates. Scalar priority does not constitute a population-wide ban on them. [Official NEAT-Python v0.92 reproduction source](https://raw.githubusercontent.com/CodeReclaimers/neat-python/v0.92/neat/reproduction.py)

Published motivation exists for allowing temporary constraint excess. Runarsson and Yao explain that excessive feasibility pressure can hinder movement between disjoint feasible regions and that infeasible regions can provide connecting paths. Their stochastic-ranking method changes selection and was evaluated with an evolution strategy; it is not proposed here because reproduction must remain stock NEAT. This supports investigating the trade-off, not claiming that our robot exhibits their benchmark geometry. [Runarsson and Yao, 2000, IEEE Transactions on Evolutionary Computation 4(3), 284–294](https://doi.org/10.1109/4235.873238); [paper PDF, Introduction and Section IV](https://faculty.csu.edu.cn/_resources/group1/M00/00/65/wKiylWIIxVaADiBnAAb4eve5Bz8876.pdf)

## Hypothesis A: predeclared epsilon constraint schedule

Takahama and Sakai publish epsilon-level comparisons: candidates whose violations are both within epsilon compete by objective, while violation otherwise takes priority, with an objective tie-break for equal violations. Their CEC2006 Equation 11 decreases epsilon to zero at a specified generation. The full published algorithm also includes differential evolution, gradient-based mutation, and special feasible elites; none should be imported into this NEAT experiment. [Takahama and Sakai, 2006, CEC, DOI 10.1109/CEC.2006.1688283](https://doi.org/10.1109/CEC.2006.1688283); [author-uploaded paper, Sections III–IV](https://www.researchgate.net/profile/Tetsuyuki-Takahama/publication/224645691_Constrained_Optimization_by_the_e_Constrained_Differential_Evolution_with_Gradient-Based_Mutation_and_Feasible_Elites/links/0a85e533cc05cadb8f000000/Constrained-Optimization-by-the-e-Constrained-Differential-Evolution-with-Gradient-Based-Mutation-and-Feasible-Elites.pdf)

The published schedule is:

```text
epsilon(t) = epsilon0 * (1 - t/Tc)^p,  0 <= t < Tc
             0,                       t >= Tc
```

**Project proposal:** in a new experiment only, preserve the current bands and replace the completed-batch quality predicate `V == 0` with `V <= epsilon(t)`. Keep full completion and every required window observed as separate requirements; missing measurements must not become epsilon-feasible. Use the existing quality limits and normalized excess measurements. Epsilon permits temporary training excess; it never changes final validation acceptance. This is epsilon-inspired scalar task fitness, not the published epsilon-DE algorithm.

Freeze `epsilon0`, `p`, `Tc`, generation indexing, and a substantial final zero-epsilon phase before launch. Alternatively freeze an initial-training-population quantile rule for `epsilon0`, including its interpolation and zero-value behavior. CEC2006 uses the violation of the initial population's 20%-ranked individual. If the chosen rule gives zero, the schedule degenerates to the strict control; do not increase it after observing the run.

Risks remain: epsilon can favor fast but persistently violating policies; closing the allowance can cause late collapse; the bands still have a moving discontinuity. The authors' later chapter explicitly notes that useful relaxation control depends on population and search behavior, so no schedule is guaranteed transferable. [Takahama and Sakai, 2008, publisher summary](https://link.springer.com/chapter/10.1007/978-3-540-68830-3_5)

**Stock-NEAT caveat:** historical best-genome and species-stagnation decisions compare fitness across generations. Shrinking epsilon can lower scores even while true constraint violation improves. Existing changes in course difficulty already affect comparability; a schedule adds another source. Log raw quality, speed, epsilon, and species history. Do not reset histories, replace stagnation, or treat an early relaxed training maximum as the final controller. [v0.92 population source](https://raw.githubusercontent.com/CodeReclaimers/neat-python/v0.92/neat/population.py); [v0.92 stagnation source](https://github.com/CodeReclaimers/neat-python/blob/v0.92/neat/stagnation.py)

## Hypothesis B: fixed continuous excess penalty

**Simpler project proposal:** keep the incomplete-batch formula, the existing quality limits/measurements, and the existing bounded speed utility `S`. For fully completed batches, replace both quality bands with:

```text
v_j = existing nonnegative normalized quality excess j
P   = mean(v_j^2 / (1 + v_j^2))
F   = 60 + 9 * (S + lambda*(1-P)) / (1+lambda),  lambda > 0
```

Include the missing-window violation in `P`; missing windows also retain zero speed. This preserves completion priority because complete-batch scores lie in `[60,69]`, above the incomplete formula's maximum of 59. Quality penalty is zero throughout the acceptable region; all such candidates compete only on speed. Outside it, speed remains visible and may compensate for a modest excess. The fixed expression removes the explicit quality-band jump and avoids schedule-induced nonstationarity. It does not remove all simulator/measurement discontinuities or guarantee feasibility.

The equation and coefficient are ours, not a published NEAT result. Freeze `lambda` using declared development calibration and synthetic comparisons before evolution, and keep every final gate unchanged. A weak penalty can reward unacceptable speed; a strong one can recreate the original barrier. Bounded penalties can compress distinctions between severe violations. Retain raw components and reject unqualified final candidates regardless of fitness. Runarsson–Yao's penalty discussion motivates these cautions; it does not validate this equation or a particular coefficient.

## Prospective experiment and control

If the completed 50-generation pilot leaves the question unresolved, **prefer B as the next minimal comparison**, with A as a separate, explicitly budgeted alternative. Do not extend or alter V1 while it runs.

- Freeze a new experiment directory, hypothesis, equations, all coefficients/schedule rules, fixed generation budget, and source/input hashes before any new evolution. Preserve V1's complete outcome, including an unsuccessful outcome if that is what occurs.
- Compare against the original strict objective using the same disclosed g83 warm start, identical initial mutated populations, paired evolution seeds, common six-course generation batches, quality references, population target, and candidate-validation budget. A new matched control is preferable; the completed V1 run remains development context. Keep stock NEAT-Python 0.92 reproduction, mutation, crossover, speciation, and stagnation unchanged.
- Vary fitness alone: keep physics, inputs, action mapping, training quality limits/floors, observed-window rules, PID baseline, and final acceptance unchanged. A different warm start or speed utility would need its own declared intervention.
- Before launch, test synthetic ranking cases: one more completion; equal quality/different speed; tiny excess versus large speed gain; severe excess; a missing window; and, for A, both sides of `Tc`. Freeze expected ordering and finite-score bounds. Stored-episode rescoring is calibration, not new independent performance evidence.
- Log completion, raw excess per constraint, true and epsilon feasibility separately, minimum straight speed, route speed, all fitness components, every rejection, actual episode counts, genotype diversity, and species history. Final selection uses the original gates and fixed validation rule, never relaxed feasibility alone.
- Compare all attempts, not the best-looking checkpoint across branches. Repeated development-suite consultation remains selection. Freeze independent confirmation and a fresh common NEAT/PID test after development; do not use test results to choose penalty strength, schedule, branch, or budget. [Cawley and Talbot, 2010, JMLR](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)

## Verification scope

Inspected the frozen V1 specification and scoring source without modifying either. Read the published CEC2006 paper's comparisons/schedule and Runarsson–Yao's discussion/results; checked the 2008 publisher summary. The stock stagnation behavior was additionally inspected in the installed 0.92 source because direct retrieval of its tagged upstream file failed. No proposal in this note is activated or experimentally validated.
