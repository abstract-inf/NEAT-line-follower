# Quality-speed follow-up development protocol

This separate experiment follows completed `neat-quality-speed-v1`: 50 generations and no qualifying model. It preserves the original 49 execution files and all V1 evidence. No evolution starts during preparation.

## Controlled contrasts

Two arms each receive 50 generations, target population 100, six common courses per genome and the top three distinct training genotypes validated per generation. `smooth` starts from the original smooth g83; `fast` starts from native-1 m/s fast g134. Both use evolution seed `7036297164298145801`, the V1 seed.

The completed-batch fitness becomes `10*m + 9*S/(1+V)`, where the V1 speed utility `S`, quality violation `V`, measurements and training floors are unchanged. Incomplete-batch fitness is identical to V1. Completion retains priority; quality violations continuously discount speed instead of imposing V1's feasibility band. Final completion, shared-finish accuracy/jitter/time and required-straight acceptance gates remain unchanged.

Strict V1 g83 versus revised g83 examines the reward change. Revised g83 versus revised native-fast initialization examines the initialization contrast under the new reward. This is not a factorial interaction test or independent confirmation. These are previously consulted development data, with one seed, not evidence of training independence or generalization.

## Immutable preparation

`prepare` requires the completed V1 protocol, matching dependency versions and all 49 unchanged original source files. It copies the original suite, both parents, cached g134 validation reference, and every original generation's exact six track JSON files, `batch.json` and `reference.json`. It checks geometry, batch order, reference identities and 1 m/s training caps. Original absolute paths inside copied batch manifests remain provenance; execution resolves copied files inside `prepared/batches/`.

Both arms reuse the same 50 frozen batches and 300 saved reference episodes. Each arm runs **zero new reference episodes**; courses and reference physics are never regenerated. Primary validation uses the same 60 courses at 1 m/s; retention uses the same 20 courses at 0.5 m/s. Validation caches and checkpoints remain separate per arm.

After seeding Python's mutation RNG, preparation creates each stock warm-start population and saves its post-mutation RNG state in `prepared/initial/{smooth,fast}.pkl.gz`. A normalized genotype inventory is preserved. For the smooth reconstruction, the three saved V1 generation-zero candidate genotypes must match their reconstructed genome IDs. V1's complete initial population was not preserved, so matching all 100 historical genomes cannot be proven and is not claimed.

Full runtime source, robot, episode rules, settings, dependencies and all prepared inputs are hashed. Later launcher commands execute the copied source. Changed input/source/settings are rejected; preserve partial preparation and use a new directory rather than silently rebuilding it.

The historical validated full-speed PID configurations and selections at 1 and 0.5 m/s are copied only as baseline provenance. They are not newly retuned baselines. Proper PID retuning, including the known long development course, is a separate task. The eventual comparison must freeze the chosen PID/NEAT controllers and settled protocol before any fresh shared test; this follow-up opens no final test.

## Explicit commands

From the project root, after implementation checks:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pytest tests/test_quality_followup.py -q --basetemp .cache/pytest-quality-followup
& $py run_neat_quality_followup.py prepare
& $py run_neat_quality_followup.py verify

# Start only the explicitly selected arm:
& $py run_neat_quality_followup.py train --arm smooth --workers 3
& $py run_neat_quality_followup.py train --arm fast --workers 3

# Resume an interrupted arm with its recorded checkpoint/RNG:
& $py run_neat_quality_followup.py train --arm smooth --workers 3 --resume
& $py run_neat_quality_followup.py status
```

The default run is `runs/neat-quality-followup-v1`. There is no automatic two-arm launch. Actual population/episode totals, repeated validation, cached work and interruptions remain development accounting; worker wall time alone is not algorithmic sample efficiency.
