# Repeating the prepared development run in a clean extraction

The preparation-era ZIP deliberately stores the source tree under `runs/neat-quality-speed-v1/prepared/source/`. It does not install root-level Python packages. Use its frozen entrypoint directly after extracting into a **separate clean project directory**, installing the recorded Python 3.12.14/dependencies and creating `.venv` there:

```powershell
$py = ".\.venv\Scripts\python.exe"
$run = (Resolve-Path "runs/neat-quality-speed-v1").Path
$entry = Join-Path $run "prepared/source/run_neat_quality_study.py"

& $py $entry verify --run $run --frozen-source
& $py $entry train --run $run --workers 6 --frozen-source

# Only after an interrupted training command:
& $py $entry train --run $run --workers 6 --resume --frozen-source
```

This uses the immutable prepared source and inputs. Do not run `prepare` again to reproduce a frozen protocol, and do not extract over an active run. The preparation-era archive excludes the live population and training logs: its `train` command starts that stage from the frozen parent and evolution seed. Dependency versions and the Python/platform record are separate reproduction requirements.

The original ZIP and its hash are preserved. This note corrects the earlier shorthand restoration text that assumed root packages were already installed. The optional completed-stage archive now also materializes the frozen root packages and entrypoint, so its documented root launcher works from a clean extraction.
