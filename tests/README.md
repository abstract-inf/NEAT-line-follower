# Automated test index

These files are software checks, not folders of trained-controller results. The 37 test modules cover retained functionality and remain useful even when an experiment failed to produce a winning controller. Experiment outcomes are indexed in [current research results](../results/README.md).

## Current compact-fitness checks

- `test_neat_simple_fitness.py`: formula, completion priority, speed direction, zero-weight jitter independence and invalid inputs.
- `test_neat_simple_study.py`: preparation and command dispatch with evolution stubbed.
- `test_simple_pilot_analysis.py`: paired denominators, committed-only reporting and immutable snapshots.
- `test_simple_continuation.py` and `test_simple_inherited_cache.py`: exact continuation, unchanged inputs and cache ancestry.
- `test_simple_curriculum.py` and `test_simple_confirmation_*.py`: prospective independent-run tooling and assessment guardrails. These do not establish that the full confirmation experiment has run.
- `test_simple_physical_fitness.py`: the unselected prospective physical-motion scorer; it is not the accepted lambda-0 reward.

```powershell
$py = '.\.venv\Scripts\python.exe'

# Fast focused checks; no full evolution job is launched.
& $py -m pytest tests/test_neat_simple_fitness.py tests/test_simple_pilot_analysis.py -q --basetemp .cache/pc

# List all retained checks without executing them.
& $py -m pytest --collect-only -q
```

The short temporary path avoids Windows path-length failures in the snapshot fixtures.

## Shared and historical checks

Robot, sensor, track, episode, PID, viewer and memory checks protect the common simulator. Speed, fast, smooth, quality, follow-up and joint study checks protect the corresponding preserved launchers and checkpoint formats. Keep them with the implementations they verify. Python caches are ignored and are not research results.
