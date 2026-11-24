# Prespecified PID gains-only follow-up

Prepared before the new search and evaluation on 29 October 2026.

## Question and fixed quantities

Can changing only `kp`, `ki`, and `kd` improve either previously selected PID
variant? The two reference configurations are the filtered PID and PID with
corner supervision saved in `runs/pid-research-20261029`. Their other parameters,
controller code, physical robot, observations, actuator limits and episode rules
remain fixed. NEAT is neither trained nor reselected in this follow-up.

## Finite search

- Four persistent Windows-spawn CPU workers; no rendering or real-time pacing.
- Each family starts with its baseline and 15 one-gain perturbations: P and D
  multiplied by 0.5, 0.75, 1.25, 1.5, 2; I multiplied by 0, 0.5, 0.8, 1.2, 1.5.
- Three reproducible random streams (11, 29, 47), each contributing 24 local and
  24 wider stratified gain triples per family. These are search streams pooled
  into one selection, not independent controller-training replications.
- Bounds: P 0.1–5, I 0–0.4, D 0–0.25 in the existing normalized-error units.
- All 320 initial configurations face the same 15 training cases: nine existing
  handcrafted cases plus six new procedural routes (two per difficulty).
- Promote the top 12 per family, always including the baseline, to the same
  expanded 45-case training suite: the nine handcrafted cases plus 36 new
  procedural routes (12 per difficulty).
- Generate 24 additional local refinements per family around its top three
  expanded-training configurations. Evaluate on all 45 training cases.
- Validate the top three per family plus each original baseline on 30 new frozen
  validation routes, ten per difficulty. Include the baseline in final selection.
- Maximum 368 distinct gain configurations; duplicate triples are rejected.
  Ranking throughout is completion fraction, mean authenticated progress,
  negative mean observed lateral RMSE, negative successful completion time.

## Splits and final assessment

New suite seed: 2026102902. Procedural training has a separate seed namespace.
Known historical track hashes are protected. Existing handcrafted training
cases and their mirrors may be reused within training. No new validation/test
geometry can duplicate training, each other, or known historical geometry.
Use the unchanged version-1 generator distribution, matching the original
comparison; this is new-route testing, not generator-family extrapolation.

Freeze one selected configuration per family, all non-gain values, source and
runner hashes before loading final test geometry. Evaluate the two baselines and
two selected policies on the same **90 new test routes**, 30 per difficulty,
once per policy. If selection retains a baseline, its repeated evaluation is a
consistency check, not independent evidence. Report paired completion changes,
RMSE, MAE, progress, failure reasons, and completion times on the common-success
subset. A gain is not called a speed improvement from unequal success subsets.

Only after freezing, also replay the original 90 known test routes as a labelled
legacy/development benchmark. They cannot supply a fresh confirmatory estimate
and never affect selection. Representative traces use handcrafted training cases
and predetermined new-test indices 0, 1, 2, 31 and 62.

No controller, gain, ranking or cohort changes after observing final outcomes.
Do not replace a frozen choice with a better test-performing candidate. Keep
unfavorable results, all failures and every evaluated candidate/episode. No
statistical-significance or algorithm-wide superiority claim is planned from this
single pooled search. Preserve previous runs and archives byte-for-byte. Save
new configurations, sources, seed schedules, CSVs, traces, figures and reports
under a new run/artifact directory.
