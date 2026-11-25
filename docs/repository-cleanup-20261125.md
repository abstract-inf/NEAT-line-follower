# Repository backup and cleanup — 25 November 2026

The user requested commit/push of the existing repository before cleanup, then clarified that intentionally ignored files must not be copied into a backup folder. The backup includes only nonignored repository content. No ignored run, video, cache, environment, manuscript or downloaded-literature files were copied, archived or uploaded.

## Verified backup before deletion

Repository: `abstract-inf/NEAT-line-follower`; branch: `research/mujoco-neat-pid`. Main was not changed. All commits use Yazan AlAtout's personal `abstract-inf` noreply address; native GitHub authentication was independently verified as `abstract-inf` before pushing.

The backup consists of these commits:

| Commit | Description |
|---|---|
| `fc08765` | Add quality-constrained NEAT studies and full-speed PID tuning |
| `8e12501` | Implement and validate compact-fitness NEAT development |
| `43c474b` | Preserve research findings, frozen results and Endurance replay |

The remote branch was checked against local HEAD before deletion. A separately pushed tag, **`backup/pre-cleanup-20261125`**, points to full commit `43c474b3733adc2def83651ad5c0dec20b55438e`. Both remote refs matched that exact commit.

## Removed from the current checkout

Every removed file was verified byte-for-byte against its Git blob in the pushed backup before deletion. Cleanup removed **2,864 files / 191.97 MiB** from eight explicitly checked workspace targets:

- `tmp/pdfs/old-paper-review/`: generated page previews/text derived from the preserved old paper.
- Four superseded captures below `results/neat-simple-pilot-v1/snapshots/`:
  - `371585675d33832e1bb77e4bf783b294f1c49f992b3abc8f4fe5b68cd3b5076c`
  - `c4d8c51afece7188cdcea488bc025831432aa6c4655fe011af9cf70efb5638d1`
  - `cfaa61c068829fe4d426ae22c6a9e703a4615435010a46bd0e440222c290eaf2`
  - `e16abd60f74e5ea9e37993475cd2a5a9b1d15e1d2832aa879f215f1f39a58ad6`
- Three superseded captures below `results/neat-simple-pilot-50-v1/snapshots/`:
  - `0509852f429aac6d1c934c4d658e4a09e522b4ba16a39cd0373672ced620dd66`
  - `079dba11f4778f5a689974a690090fd2ea773e2de73ef1cbdd012e8552c7c00d`
  - `f9fd691cf0e72377c19d23aad560476dd937e30cb9462d87b0075d4989dbccf0`

These are superseded analyses, not independent experiment outcomes. Current report/model/decision references use the retained terminal captures:

- Twenty-generation pilot: `3864b04984d3709ad84f2fa3b24153b79e40230673450ade4538cdbc8446e74f`.
- Fifty-generation continuation: `5c4e5e5ea6f1f2461040cdaba557279f983f78da5daa11f47a382f5f373a673a`.

All failed-study evidence, required PID/prototype caches, models, raw runs and checkpoint/source trees remain at their original paths. All 37 automated test modules remain: they verify code still present in the repository. The root and results indexes now lead with the accepted lambda-0 result; earlier studies are grouped under expandable historical sections. The test index identifies the current checks. Future temporary PDF previews under `/tmp/` are ignored.

## Restore a removed path

Restore only the specific path required; no branch-wide reset or history rewrite is needed:

```powershell
# Example: recover an earlier capture from the published backup tag.
git restore --source=backup/pre-cleanup-20261125 --worktree -- results/neat-simple-pilot-50-v1/snapshots/f9fd691cf0e72377c19d23aad560476dd937e30cb9462d87b0075d4989dbccf0
```

This recovers repository files only. Intentionally ignored local data was excluded from the backup as requested. No new controller training or simulation was launched during cleanup.

## Post-cleanup checks

- Frozen continuation verification passed, preserving source/input/ancestry hashes.
- The retained terminal snapshot reproduced the full 50-generation numerical report byte for byte.
- All 37 existing automated test modules remain tracked.
- All current index links resolve locally and both simple-pilot snapshot folders retain their current report's terminal capture.
- Focused checks: **42 passed**, using `python -m pytest tests/test_neat_simple_fitness.py tests/test_simple_pilot_analysis.py -q --basetemp .cache/pc`. The short temporary path avoids the Windows path-length failure encountered with the sandbox's default pytest temporary directory; no runtime code was changed for that environment issue.
