# Full-cruise PID at the 1.00 m/s cap: protocol and correction

Recorded before the new tuning and shared test on 19 November 2026.

## Correction to the previous comparison

The previous filtered PID used `base = 0.75`. At a 1.00 m/s wheel-speed cap, its nominal straight-line wheel-speed request was therefore 0.75 m/s. It sustained approximately 0.743 m/s on dedicated straights. The earlier comparison documents those particular saved policies, but its 9.37% NEAT finish-time advantage cannot establish superiority over a PID allowed to request full 1.00 m/s travel. It must be labeled as a **restricted-cruise baseline comparison**, rather than the primary full-speed comparison.

The restriction came from retaining a configuration parameter during the smoothness refinement. It is not a mathematical limitation of PID. The same controller implementation with `base = 1.00`, at the unchanged 1.00 m/s cap, was verified to sustain **0.990512 m/s** over the central 2 m of a 6 m straight. Both wheel commands were 1.00. Physics can briefly overshoot a requested speed target.

## Full-speed eligibility

All new candidate PID configurations use **base exactly 1.00**. This permits full forward requests on a centered straight. The existing error-dependent slowdown and steering mixing may reduce travel speed on corners. Recovery logic, robot, sensors, actuation, and evaluation rules remain unchanged.

The candidate variables are P, I, D, the existing derivative-filter time constant, and slowdown gain. Other recovery parameters remain fixed. A candidate must complete every dedicated validation straight and sustain at least **0.95 m/s** over each central 2 m interval. The same approximate 1 m/s target was used for NEAT. The final report must include the measured speeds and actual wheel commands.

## Development and selection

- Fifty fresh development courses: 20 mixed, 20 speed, and 10 corner drills.
- Sixty separate validation courses: 30 mixed, 20 speed, and 10 corner drills.
- A structured/local search plus seeded random candidates, followed by two refinement rounds. The previous base-0.75 tune is evaluated as context and is ineligible for full-speed selection.
- Eight CPU workers use the existing shared MuJoCo episode loop.
- Select full-speed-feasible candidates with the highest validation completion count. Among those, retain candidates whose failure-penalized mean episode time is within 2% of the fastest. Select the smoothest steady-section yaw response, then command jitter, geometric RMSE, and time as tie-breaks.

For selection, an incomplete episode receives its full configured timeout as its time cost. This avoids rewarding early failures with short elapsed times. The final scientific time comparison still uses only tracks both fixed policies complete and reports that denominator.

These criteria are independent of NEAT's new test outcomes. They do not retroactively modify the original five-seed NEAT selection rules.

## Fresh comparison

Freeze the selected full-cruise PID and the unchanged seed-1 NEAT generation 134 before generating 150 new shared tracks: 90 mixed, 30 speed, and 30 corner drills. Exclude previous geometry, including the earlier restricted-cruise comparison, both PID refinement datasets, and NEAT training/validation/test geometry. The original five-seed study's final test set remains untouched.

Both controllers use a 1.00 m/s wheel-speed cap, identical robot physics, sensor sampling, starting poses, actuator limits, timeout rules, 100 mm corridor, two-second off-route allowance without sensor contact, and the five-second stall check. Retain the stock NEAT-Python 0.92 evaluator and the filtered PID's remembered-side recovery.

Report completion, progress, shared-finish time, all-episode and shared-finish geometric RMSE and command jitter, measured mean/peak speed, sustained straight speed, physical yaw oscillation on steady windows, and failure reasons. Use the same paired-track bootstrap and exact paired completion test as the previous comparison. Treat the existing long course as exploratory. No parameters may be selected or retuned from these fresh test outcomes.

This is a comparison of fixed resulting controllers. Algorithm-wide conclusions still require independent NEAT lineages and an appropriately scoped control study.

## Files and commands

PID tuning and feasibility evidence are saved under `runs/pid-fullspeed-100-20261119/`. The new frozen comparison is saved separately under `runs/neat134-vs-pid100-fullspeed-20261119/`. The previous policies, measurements, and source snapshots remain intact.

```powershell
$py = ".\.venv\Scripts\python.exe"

# Verify full-cruise feasibility without tuning or evolution
& $py run_pid_fullspeed_100.py probe

# Run PID tuning, freeze the chosen controllers, and assess the new shared suite
& $py run_pid_fullspeed_100.py all

# Replay the new frozen policies on ten tracks after the assessment
& $py replay_neat_pid_100.py --run runs/neat134-vs-pid100-fullspeed-20261119
```

These commands perform no NEAT evolution. Repeating the completed workflow verifies and reuses frozen selections/results.
