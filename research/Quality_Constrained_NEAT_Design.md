# Quality-constrained NEAT fitness design

Research note, **24 November 2026**. Design recommendations only: no training, simulations, controller changes, or test evaluation were performed for this note.

## What the evidence supports

The [completed pilot](../results/neat-smooth-study-v1/pilot-comparison/report.md) produced a useful development result, but no qualifying winner. Smooth generation 83 completed 60/60, reduced geometric RMSE from 14.49 to 7.35 mm and jitter from 0.006530 to 0.006050, and passed every straight yaw/steering window; its roughly 0.713 m/s straight speed and 12.07% longer laps failed the combined target. The rescore also demonstrates that the new reward can prefer a slow controller on these stored validation episodes. It does not establish an unavoidable speed/smoothness limit.

**Recommended next hypothesis:** reward acceptable quality, then speed, rather than continuously paying for increasingly small motion below the accepted limits. Keep the limits and final acceptance rules fixed. This is a new, separately versioned development experiment, not a reinterpretation of the failed pilot.

## Published basis and algorithm boundary

Stanley and Miikkulainen's original NEAT paper separates the evolutionary machinery from task fitness. In its non-Markovian pole-balancing experiment, the task fitness explicitly penalizes oscillation (Section 4.3.3, Equations 3–4). Thus, physically measured smoothness terms are compatible with NEAT; the paper does not prescribe a universal line-following reward or guarantee that a chosen weighting works. Its core mechanisms include historical markings, speciation, and incremental structural growth. [Stanley and Miikkulainen, 2002, final journal paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

Deb's constraint handling compares candidates by feasibility first, objective among feasible candidates, and total violation among infeasible candidates (Section 3, Equation 4). This motivates a quality-constrained task objective without requiring a precisely tuned exchange rate between quality and speed. The original procedure uses tournament selection and includes additional search mechanisms; applying its ordering as a scalar objective inside NEAT is **Deb-inspired fitness assignment**, not a reproduction of Deb's algorithm. [Deb, 2000, author manuscript](https://repository.ias.ac.in/9407/1/310.pdf); [published article, DOI 10.1016/S0045-7825(99)00389-8](https://doi.org/10.1016/S0045-7825(99)00389-8)

NEAT-Python **v0.92** explicitly accepts a user callback assigning each genome a float fitness. Its stock reproduction sorts members within species, retains configured elites, samples parents from the survival fraction, and allocates species offspring using normalized species mean fitness. Negative finite fitness is supported. Consequently, a scalar reward can change without replacing NEAT, but individual scalar ordering does not make species allocation equivalent to Deb tournaments. [Official v0.92 population source](https://raw.githubusercontent.com/CodeReclaimers/neat-python/v0.92/neat/population.py); [official v0.92 reproduction source](https://raw.githubusercontent.com/CodeReclaimers/neat-python/v0.92/neat/reproduction.py)

For this project, changing measured rewards, their aggregation, training distributions, or validation selection is task/experimental design and must be reported. Replacing reproduction, speciation, mutation, crossover, or stagnation is an algorithm change. PID imitation targets, hand-coded route decisions, or output filtering would change the claimed controller and are outside this proposal. Evaluator-only route geometry for measuring progress/error does not grant the policy route information; preserve the existing 21 sensor-history inputs and direct drive/steering outputs.

## Fitness proposal

The following equations are **project design proposals**, not equations claimed to be published NEAT or Deb results.

For a quality measurement `x` with upper limit `L > 0`, use:

```text
excess(x, L) = max(0, x/L - 1)
penalty(x, L) = excess(x, L)^2 / (1 + excess(x, L)^2)
```

This cost is exactly zero throughout the acceptable region, continuously differentiable at the limit, increasing outside it, and bounded. Apply it to raw straight yaw, straight steering, training tracking error, and whole-course command jitter. Use each required straight window, with an explicit missing-window cost. Freeze normalization, weights, and mean/tail/worst-window aggregation before the run. Whole-course yaw is not a suitable zero target on curves. Training route-relative error and independently measured geometric RMSE remain different quantities.

**Replace all competing quality incentives**, including the inherited continuous jitter penalty; simply adding plateau terms leaves the previous incentive to slow down for ever-smaller jitter. Audit the inherited legacy bonus as well. Keep a monotonic completion-time/speed objective active once quality passes. A bounded utility with no flat region for slow successful laps is:

```text
s_i = completed_i * route_length_i /
      (route_length_i + speed_cap * completion_time_i)
S = mean(s_i)
```

Define `s_i = 0` for failures without using their early termination time as a successful lap. Avoid capping the speed objective at 0.95 m/s if faster acceptable driving is desired. The 0.95 m/s straight gate and 1.05 paired-time gate still decide final qualification.

Two defensible branches are possible; choose one prospectively, or budget both explicitly:

1. **Simple plateau reward.** Retain completion/progress scoring and the existing tail aggregation, replace continuous quality costs with the bounded excess penalties, and emphasize elapsed time or route speed. Within the acceptable region, quality costs provide no incentive for further slowdown. Finite weighted penalties still allow quality/speed trade-offs outside that region; hard validation gates remain necessary.
2. **Stricter scalar priority.** For each common `m`-episode training batch, let `C` be completion count, `V >= 0` a fixed normalized violation sum, and `S` the bounded utility above. Include quality excesses, missing required windows, and `mean((1-completed_i) + (1-progress_i))` in `V`, with ordered progress clipped to `[0,1]` and set to one on completion. Incomplete candidates then receive graded direction and cannot masquerade as quality-feasible. Assign:

   ```text
   U = 0.5/(1+V)       if V > 0
       0.5 + 0.5*S     if V = 0
   fitness = C + 0.9*U
   ```

   One additional completion dominates every residual difference. At equal completion count, feasibility dominates; among feasible candidates speed determines score. Among infeasible candidates lower violation wins. Normalize violation dimensions with training-only, frozen scales. This branch changes batch aggregation and must be documented separately from the simple plateau branch. It preserves stock reproduction but does not forbid infeasible parents from breeding in their species.

## Failure modes to avoid

- Do not delete every genome failing the final gates: the pilot shows that the qualifying set can be empty. Evaluate all genomes and give finite, graded fitness. Assigning every failure the same score removes selection direction; it does not magically create a feasible parent.
- The strict scalar branch has a discontinuity where feasibility activates the speed term. NEAT does not need differentiable fitness, but narrow feasibility regions, noisy threshold crossings, and large cliffs can make selection brittle. The plateau branch is smoother. Any feasibility margin, tolerance, or staged schedule is a declared development choice, not a post hoc relaxation.
- Bounded excess penalties saturate for severe violations. Preserve progress/completion direction and log violation magnitudes separately; the bounded cost alone may poorly distinguish disastrous policies.
- Aggregate means can hide one bad recovery straight. Preserve the every-window validation gates and explicit unobserved-window failures. A training proxy passing on six courses is not proof that all 60 validation courses pass.
- Do not use final-test measurements to set limits, choose a checkpoint, choose branches, extend budgets, or tune PID. Repeatedly consulted validation is selection data. Model-selection overfitting can bias performance comparisons even when training itself never sees the test data. [Cawley and Talbot, 2010, JMLR](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)

## Claims, comparison, and experiment record

A single paired development seed supports statements about those two runs and motivates another hypothesis. One hundred correlated champions are not 100 independent evolution seeds. A fixed controller's many test episodes quantify its course behavior; they do not establish training repeatability. Few-run evaluation needs uncertainty and per-run results rather than a selected maximum. This methodological lesson is supported in a different, deep-RL setting by [Agarwal et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).

Use independent confirmation seeds after freezing the protocol. Pair courses, physics, speed caps, sensor access, and episode termination across NEAT and filtered PID. Describe the PID derivative filter, speed law, and recovery as part of its complete baseline; the comparison then concerns complete controllers. Report failures first, paired time with shared-success denominators, all-window speed/yaw/steering, geometric RMSE, jitter, and per-seed spread. Failure-penalized time complements successful-lap timing.

For every attempted branch, preserve its hypothesis, full equation/version, thresholds, source/input hashes, parent hash, seeds, training batches, actual populations/episode budgets, component logs, every rejection, checkpoint selection rule, and outcome. Retain failed experiments and label retrospective rescoring as diagnostics. Equalize new branch budgets and initial populations when isolating a fitness intervention; comparisons against old frozen parents also include extra evolution. No blanket claim of NEAT superiority, reward optimality, hardware robustness, or statistically established algorithm-wide improvement follows from this pilot.

## Source verification

The final 2002 NEAT PDF and v0.92 source were directly inspected. Deb's publisher abstract and indexed primary-manuscript Section 3/Equation 4 excerpts were inspected; direct repository PDF retrieval returned HTTP 403. The Cawley–Talbot and Agarwal papers were checked through their primary publication records/PDF text. These sources justify the design principles and evaluation cautions; none validates the proposed coefficients or predicts success on this robot.
