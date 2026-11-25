# GitHub workflow for the new MuJoCo experiments

The 25 November cleanup first pushed all then-nonignored work and published `backup/pre-cleanup-20261125`, using the personal `abstract-inf` author identity and authentication. Intentionally ignored files were not copied or uploaded. See [the cleanup record](repository-cleanup-20261125.md) for removed paths and restoration; the [results index](../results/README.md) now opens with the accepted simple-fitness outcome.

Workflow recorded on 24 November 2026. The ignore rules and byte-preserving attributes have been implemented. The first-time setup below is retained as a reproducibility record; do not repeat it in a folder that already contains `.git`.

Use the existing repository, <https://github.com/abstract-inf/NEAT-line-follower>, with a new branch named **`research/mujoco-neat-pid`**. Its current default branch is `main`. Keep `main` as the original paper's implementation while the follow-up study develops on the research branch. When the new paper is finalized, tag the exact code commit and publish a matching data/model release.

At planning time this project folder had no `.git` directory. The first-time commands below attach such a folder to the existing history without checking the old code out over the working files. The research branch contains the current project layout; old tracked files absent from this folder appear as deletions on that branch. Review them before committing. They remain available in the original history and on remote `main`.

## What belongs in Git

| Content | Recommendation |
|---|---|
| `line_follower/`, `neat_smooth_study/`, root Python launchers and analysis scripts | Commit |
| `configs/`, `benchmarks/`, tests, dependency lock, package metadata | Commit, including small fixed track suites |
| `docs/`, `research/*.md`, analysis notebooks/scripts, selected small figures and tables | Commit |
| `LICENSE.txt`, `THIRD_PARTY_NOTICES.md`, project README | Preserve and commit |
| `runs/`, caches, virtual environments, temporary renders and raw archives | Ignore; archive research evidence separately |
| Selected frozen genomes and PID settings | Publish a small curated, versioned selection; preserve hashes and provenance |
| Full checkpoints, per-episode logs, source bundles, prepared inputs and videos | Keep backed up and distribute as release/data archives |
| Downloaded related-work PDFs | Keep local; commit bibliographic notes and links instead |

The local inspection found about **1,627 MiB / 42,178 files in `runs/`** and **332 MiB in `research/`**, including about 185 MiB of ZIP archives. These are a snapshot, not a fixed expected size. Avoid putting all of that into ordinary Git history.

Do not globally ignore `*.csv`, `*.json`, `*.pkl`, `*.png` or `*.svg`: curated metrics, track definitions, PID settings, selected small genomes and paper figures are useful repository content. Keep raw outputs under the ignored folders and explicitly curate what gets published.

## Implemented `.gitignore`

The project-root `.gitignore` contains the following. It extends the original rules without deleting local files.

```gitignore
# Python environments and caches
.venv/
.venv-*/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.cache/
*.egg-info/
build/
dist/

# Generated experiment data and historical raw archives
/runs/
/output/
/research/archive/
/research/artifacts/**/telemetry/

# Downloaded literature and local presentation
/Related Work/
/simple-presentation.pptx

# Rendered media and simulator exports
*.mp4
*.avi
*.mov
*.mjpg
*.mjb

# Local secrets and temporary files
.env
.env.*
!.env.example
*.tmp
*.bak

# Local OS/editor files
.DS_Store
Thumbs.db
Desktop.ini
/.idea/
/.vscode/
```

`research/artifacts/` is not blanket-ignored: its small figures and analysis data support existing reports. Its nested per-step `telemetry/` directories are excluded: they accounted for about 123 MiB at initial publication and remain available locally for a separate data archive. The paper PDF already present in the repository can remain; there is no global PDF exclusion.

## Preserve the frozen fingerprints

Create a root **`.gitattributes`** with:

```gitattributes
# Protocols fingerprint exact file bytes; preserve their original line endings.
* -text
```

This deliberately disables Git line-ending normalization for this repository. Several prepared protocols hash Python source, configuration and track files as bytes. CRLF/LF conversion during commit or checkout can make an otherwise identical experiment fail verification. Keep frozen source ZIPs, input files and protocol hashes unchanged; make future scientific changes in a new protocol version.

## First-time connection from this project folder

First save the `.gitignore` and `.gitattributes` above. Run this block only once, from the existing `NEAT Line Follower` folder. It is tailored to the inspected state with no Git repository attached.

```powershell
if (Test-Path -LiteralPath '.git') {
    throw 'A Git repository already exists here. Inspect git status and git remote -v instead of repeating first-time setup.'
}

git init -b research/mujoco-neat-pid
if ($LASTEXITCODE -ne 0) { throw 'Git initialization failed.' }

git remote add origin https://github.com/abstract-inf/NEAT-line-follower.git
if ($LASTEXITCODE -ne 0) { throw 'Adding the remote failed.' }

git fetch --filter=blob:none --no-tags origin main
if ($LASTEXITCODE -ne 0) { throw 'Fetching the original history failed.' }

# Attach this research branch to the old main commit, retaining working files.
git reset --mixed origin/main
if ($LASTEXITCODE -ne 0) { throw 'Attaching the original history failed.' }

# Preserve the existing repository license; it is missing from this local folder.
git restore --source=origin/main --worktree -- LICENSE.txt
if ($LASTEXITCODE -ne 0) { throw 'Restoring the repository license failed.' }

git branch --show-current
git status --short
```

The `--mixed` reset changes the branch pointer and index while leaving working files unchanged. Do not substitute `--hard`. The partial fetch avoids eagerly downloading every historical file blob, although later operations can download old contents as needed. The branch inherits the original history and pushes to a different remote branch.

## Review, commit and push

```powershell
git add -A

# Inspect anything already tracked by the old repository despite ignore rules.
git ls-files -ci --exclude-standard

git diff --cached --stat
git diff --cached --name-status
```

If the ignored-tracked check lists files, untrack just those paths while retaining their local copies:

```powershell
$ignoredTracked = @(git ls-files -ci --exclude-standard)
if ($ignoredTracked.Count -gt 0) {
    git rm --cached -r --ignore-unmatch -- $ignoredTracked
}
```

Check the sizes of staged additions/modifications:

```powershell
git diff --cached --name-only --diff-filter=ACM | ForEach-Object {
    $stagedFile = Get-Item -LiteralPath $_
    if ($stagedFile.Length -gt 50MB) {
        [PSCustomObject]@{ Path = $_; MiB = [math]::Round($stagedFile.Length / 1MB, 1) }
    }
}
```

Review any reported files and keep bulky generated data out of the commit. Then:

```powershell
git diff --cached --stat
git commit -m "Add reproducible MuJoCo NEAT and PID follow-up experiments"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; nothing has been pushed.' }

git push -u origin research/mujoco-neat-pid
```

This pushes the research branch. No force push or merge into `main` is needed. Use your GitHub account's normal Git authentication; a token must not be written into a remote URL or committed file.

## Keeping results reproducible

Ignoring `runs/` means a Git push **does not back up training results**. Preserve full run folders in a separate backup. Publish stable snapshots as GitHub Release attachments or a research data deposit such as Zenodo, with SHA-256 manifests and restoration instructions. Git LFS is another option for deliberately selected large versioned files, but it is not necessary for every changing checkpoint.

For the current smooth-driving pilot, a prepared continuation bundle must preserve at least the exact `runs/neat-smooth-study-v1/prepared/` tree, `audit/` results and `pilot/references/`, restoring them to the same relative paths. Include stage checkpoints/logs as well when publishing a resumable stage. Take snapshots at a completed stage or committed checkpoint; do not publish a mixture of files being updated by an active generation.

A clean code checkout alone cannot run the current `prepare` command from scratch: preparation reads historical runs, parent models and validation logs. Either publish those transitive historical inputs or supply the already-prepared frozen bundle. With the latter restored, begin with `run_neat_smooth_study.py verify` and the documented pilot commands; do not describe source-only cloning as sufficient reproduction. Existing report links into ignored run/archive folders also need the corresponding archive or a published permanent data link.

Small publication tables/figures, frozen models and a restoration README are curated under [`results/neat-smooth-study-v1/`](../results/neat-smooth-study-v1/README.md). That snapshot includes the completed audit episodes and parent references, an exact protocol copy, selected frozen inputs and SHA-256 provenance. A small immutable prepared-input ZIP (about 1.8 MiB) is deliberately included with this curated snapshot, allowing the audited pilot inputs to be restored without publishing all raw runs. Its contents exclude live generation checkpoints. Bulk historical data, telemetry and videos still need separate data archives. Do not relocate live frozen inputs simply to make them easier to commit.

For later code/document changes:

```powershell
git status
git add -A
git diff --cached --stat
git commit -m "Describe the change or newly documented result"
git push
```

Before a paper submission, tag the exact validated commit and link the matching immutable data/model release. Do not rewrite old-paper history to remove its existing large files as part of this branch setup.

## References and preparation check

- [Existing repository](https://github.com/abstract-inf/NEAT-line-follower).
- [Git reset modes](https://git-scm.com/docs/git-reset): `--mixed` preserves the working directory.
- [Git partial fetch](https://git-scm.com/docs/git-fetch): `--filter=blob:none` defers file-content downloads.
- [GitHub ignore rules](https://docs.github.com/en/get-started/git-basics/ignoring-files): ignore rules do not automatically untrack existing files.
- [GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github): ordinary Git warns above 50 MiB and blocks files above 100 MiB; releases/LFS are alternatives.

Before initial publication, the initialization/mixed-reset/commit steps were checked on disposable local repositories using a locally supplied reference commit, including preservation of working file bytes and an ignored checkpoint, ancestry from the old commit, retention of the old `main`, and exact CRLF preservation with the attributes above. The real repository was then fetched and the local research branch attached to the existing `main` commit. Publication must be verified by comparing the local commit with the remote research-branch head; `main` is not a push target in this workflow.
