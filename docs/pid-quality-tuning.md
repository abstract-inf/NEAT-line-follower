# PID quality tuning, version 1

`run_pid_quality_tuning.py` runs a new, deterministic development search at caps
1.0, 1.5 and 0.5 m/s, in that order. An already eligible saved tune is a control,
never a reason to skip the search. This is a bounded search for the best observed
PID; it does not establish a global optimum.

## Frozen inputs and scope

Preparation copies the physical robot and episode settings from
`runs/neat-quality-speed-v1/prepared/protocol.json`, the existing PID-only
50-development and 60-selection course suites from
`runs/neat-smooth-study-v1/prepared/pid/suite`, the saved full-base 1.0 and 0.5
PID tunes with their selection records, and the legacy 1.5 tune with base set to
1.0 as a starting control. Each policy preserves the stock `ResearchPIDController`,
binary sensor interface, physical model and recovery settings. Only `kp`, `ki`,
`kd`, `derivative_filter_s` and `slowdown_gain` vary; base is always 1.0.

The previously inspected 45.77 m Rhythm Changes course
(`runs/rhythm-changes-case-study-20261121/track.json`, the former long-course
candidate option 4) is explicitly **development/tuning data**. It appears in
every search and selection phase and cannot be presented as a fresh final test.
No fresh final test is created or consulted by this launcher.

Source, dependency versions, settings, course files and reference inputs are
hashed and copied under `runs/pid-quality-tuning-v1/prepared`. Evaluations run
from that frozen source tree. Preparation rejects changes to the recorded
controller, physics or evaluator. Resume validates the frozen protocol, sources,
dependencies, inputs, candidate parameters and ordered course list.

## Search and selection

Each cap evaluates 130 coarse candidates: exact saved controls, explicit 3%
single-axis perturbations, 20% axis steps, a broad PD/filter/slowdown grid and
seeded random points. Generated points use the existing bounds:

| Field | Bounds |
| --- | --- |
| kp | 0.05–3.0 |
| ki | 0–0.25 |
| kd | 0.005–0.20 |
| derivative_filter_s | 0.02–0.20 |
| slowdown_gain | 0.3–10.0 |

Exact saved reference gains are allowed outside those bounds, including the
legacy 1.5 control; they are disclosed as controls rather than generated points.
The coarse bank includes 4 isolated straights, 4 post-turn courses, 2 mixed,
2 corner courses and the long case. Twelve policies, including the current tune,
advance to all 50 development courses plus the case. Two rounds each add 24
local policies, with a smaller second-round step. At most 10 finalists, including
the current tune, receive all 60 selection courses plus the case. The default
plan has a maximum budget of 5,360 episode records per cap, 16,080 across three
caps; fewer feasible promoted policies or evaluation errors can reduce it. The
protocol derives its development/selection counts from the copied suite. Repeated geometry across phases is
deliberately reevaluated and counted as additional episodes, not fresh courses.

The long-case timeout is exactly `180 / min(cap, 1)` seconds for every candidate:
360 seconds at 0.5 m/s, 180 seconds at 1.0 or 1.5 m/s. A local adapter temporarily
sets equal minimum/maximum limits in the unchanged evaluator and restores the
normal episode plan afterward. All other courses retain the original stopping
rules. Policies still terminate early for completion, stalls or leaving the route.

A selectable policy must complete all isolated courses and observe every isolated
straight window at a speed of at least 95% of the cap, complete at least 57/60
selection courses, and complete the long case. Among eligible policies, selection
maximizes completed episodes, keeps candidates within 2% of the fastest
failure-penalized mean time, then minimizes observed straight yaw RMS, command
jitter, geometric RMSE and time. Failed episodes contribute their time limit to
the mean time. Refinement can continue when the coarse bank has no feasible
policy; the final gates stay unchanged. A cap with no eligible winner records its
best observed candidate and failures, and writes no selected configuration.

## Running and reviewing

From the repository root in the recorded virtual environment:

```powershell
.venv/Scripts/python.exe run_pid_quality_tuning.py prepare
.venv/Scripts/python.exe run_pid_quality_tuning.py verify
.venv/Scripts/python.exe -m pytest tests/test_pid_quality_tuning.py -q
.venv/Scripts/python.exe run_pid_quality_tuning.py tune --workers 1
.venv/Scripts/python.exe run_pid_quality_tuning.py status
```

`--caps 1.0`, `--caps 1.5`, or `--caps 0.5` selects a subset. `--run` selects a
separate run directory. Repeating `tune` resumes committed candidate records and
verifies a completed cap rather than repeating its search. Use a new run version
for a changed search, source or input. Default one worker conserves memory;
`--workers 2` permits two simulation workers, each with a 64 MiB mask-cache
budget plus simulator/process overhead. Large uncached masks can temporarily
exceed that cache budget. No NEAT training occurs.

Each `cap-XXX/episodes/PHASE/CANDIDATE.json` is an atomic record of parameters,
signature and raw episodes (or a captured evaluation error). Candidate parameters
are committed before each new phase. Per-phase summary files retain rejection
reasons and failure termination counts for every evaluated policy. `report.json`
contains every finalist, selection shortlist, best observed candidate and paired
comparisons with the current tune. Paired time/error/jitter means disclose their
shared-finish denominator; completion differences disclose all paired courses.
All-course failure-penalized time and observed-window yaw have separate explicit
denominators in the summaries.

An eligible winner produces `pid-fullspeed.json` and `selection-freeze.json`;
an unsuccessful search produces `failed.json`. `complete.json` counts committed
episode records. Runtime records identify starts, workers and source. Unknown
partial work interrupted before an atomic candidate commit is not included in
those totals; already committed records remain reusable.

Preparation initially stopped on an incorrect assumption of 60 development
courses, before any simulation or protocol commitment. Inspection established
the existing suite has 50 development courses (20 mixed, 10 isolated, 10
post-turn, 10 corner) and 60 selection courses. The copied partial preparation
was preserved under the run's `preparation-attempts` directory, the course-count
check was corrected, and the complete protocol was prepared from the actual
existing suite. No new courses were invented to fill the difference.

## Validation

The focused tests cover deterministic candidate generation and tiny perturbations,
fixed recovery/base settings, strict eligibility boundaries, completion/time/quality
selection ordering, per-cap case timeouts, restoration after errors, paired metric
denominators, failed-search continuation and cache resume/input rejection. Run
preparation and source verification before launching the expensive search.

## Completed development search — 25 November 2026

The authorized command was
`.venv/Scripts/python.exe run_pid_quality_tuning.py tune --workers 1`.
It ran from 2026-11-24 21:30:44 UTC to 2026-11-25 02:18:53 UTC and exited 0.
All three caps evaluated 178 distinct candidate policies and committed 5,360
episode records each: **16,080 records total**. Every cap completed 130 coarse,
12 expanded, 24 first-refinement, 24 second-refinement and 10 selection candidate
evaluations. There were no evaluation errors, process restarts or polling errors;
the frozen source/input/dependency verification passed after completion.

| Cap (m/s) | Outcome | Selection suite | Long case | Minimum isolated speed (m/s) |
| --- | --- | --- | --- | --- |
| 1.0 | Eligible: `refine2-169` | 60/60 | Completed, 68.333 s / 180 s | 0.990512 |
| 1.5 | No eligible policy; best observed `refine2-154` | 29/60 | Completed, 82.267 s / 180 s | 1.485768 |
| 0.5 | Eligible: `refine2-160` | 60/60 | Completed, 100.667 s / 360 s | 0.495256 |

The 1.5 m/s cap failed its unchanged completion gate: every finalist finished
fewer than 57/60 selection courses. Its best observed policy had 28 off-route
and 3 stalled selection episodes. No `pid-fullspeed.json` or selection freeze
was written for that cap. Completing this single long case does not establish
general course reliability.

| Cap (m/s) | kp | ki | kd | Derivative filter (s) | Slowdown gain |
| --- | --- | --- | --- | --- | --- |
| 1.0, selected | 0.3678177594534439 | 0.09419721849091163 | 0.029728500168262978 | 0.09251637059465112 | 3.351432508684658 |
| 1.5, ineligible best observed | 0.06511591193481737 | 0.1997607900791799 | 0.043611549241825066 | 0.10877091382149143 | 7.427963765291383 |
| 0.5, selected | 1.67726074940466 | 0.14312551487705227 | 0.05408622253905537 | 0.17640031752697902 | 0.4512635854307131 |

All parameters use base=1 and preserve the saved recovery settings. Selected
configurations are `runs/pid-quality-tuning-v1/cap-100/pid-fullspeed.json` and
`runs/pid-quality-tuning-v1/cap-050/pid-fullspeed.json`. The ineligible cap-1.5
parameters remain in `cap-150/candidates.json`, its raw candidate records and
`completion-summary.json` for reproducibility.

There are material quality/speed tradeoffs. At 1.0 m/s the selected mean time is
8.383880 s versus 8.853552 s for the current control, a 5.30% improvement, while
observed-window yaw RMS is 17.47% higher, command jitter 0.69% higher and geometric
RMSE 1.53% higher. The slower control lies outside the fixed 2% time shortlist.
At 0.5 m/s the selected mean time is 12.332240 s versus 12.319126 s (0.11% slower),
yaw RMS is 4.94% lower and jitter 5.62% lower, while geometric RMSE is 10.19%
higher. These are the winners under the frozen ordering, not claims of improving
every metric. Both selected caps have 61 shared finishes for paired time, jitter
and geometric RMSE; yaw uses 20 observed windows per candidate. The failed
1.5 m/s best observed policy shares only 23 finishes with its current control.

`runs/pid-quality-tuning-v1/completion-summary.json` records exact gains, phase
counts, every phase's rejection totals, selected metrics, paired denominators
and control changes. Per-cap `report.json`, phase summaries and all atomic raw
records retain individual failures/rejections. `launch-state.json` records the
exact command, PID 26736, exec session 63611, timestamps, exit code and protocol
hash; `search-stdout.log` and `search-stderr.log` preserve the process logs. The
45.77 m case remains consulted development data, and this run supplies no fresh
final-test or global-optimum evidence.
