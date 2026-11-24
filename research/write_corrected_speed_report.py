"""Regenerate the corrected speed report, presentation figures, and notebook."""

from collections import Counter
from pathlib import Path

import analyze_speed_results as plotting
from analyze_speed_results import (
    SETUP, build_notebook, describe, embedded, export_csv, group_episodes,
    label, page, pair, save, show_table, source_of,
)
from evaluate_speed_comparison import ROOT, RUN, digest, read_csv, read_json
from reassess_speed_results import OUT

CONDITIONS = (("speed-g98", .5), ("pid-filtered", .5), ("speed-g98", 1.5), ("pid-filtered", 1.5))


def completion_figure(groups):
    fig = page(1, "NEAT and recovery PID on the same 90 tracks",
               "Corrected evaluator. Controller gains and genomes are frozen across the matched test cohort.",
               "Each difficulty has 30 tracks. Numbers at bar ends show total finishes out of 90.", 5.7)
    axes = fig.add_gridspec(1, 2, left=.16, right=.95, top=.65, bottom=.17, wspace=.45).subplots()
    shades = ("#99bfc1", "#448e8a", "#20595f")
    for ax, (cap, neat) in zip(axes, ((.5, FINAL), (1.5, FINAL))):
        for y, policy in enumerate((neat, FILTERED)):
            r = describe(groups[policy, cap])
            left = 0
            for d, color in zip(DIFFICULTIES, shades):
                n = r[d]
                ax.barh(y, n, left=left, color=color, height=.48)
                if n >= 7:
                    ax.text(left+n/2, y, str(n), ha="center", va="center",
                            color=INK if d == "beginner" else "white", weight="bold")
                left += n
            ax.text(left+1.8, y, f"{left}/90", va="center", weight="bold", color=COLORS[policy])
        ax.set(xlim=(0, 105), xticks=(0, 30, 60, 90), yticks=(0, 1),
               yticklabels=[label(neat), "Recovery PID"], xlabel="Tracks finished", ylim=(1.6, -.6))
        ax.set_title(f"{cap:.2f} m/s cap", loc="left", pad=12)
        ax.set_axisbelow(True)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=9, color=c, label=d.title())
                        for d, c in zip(DIFFICULTIES, shades)],
               ncol=3, loc="upper left", bbox_to_anchor=(.06, .79), frameon=False)
    save(fig, "01-corrected-completion")


def timing_figure(groups):
    fig = page(2, "Completion time on tracks both controllers finished",
               "Each point pairs NEAT and PID on one identical track. The diagonal marks equal completion times.",
               "Points above the diagonal favor NEAT. Each panel uses its own shared-finish subset.", 5.9)
    axes = fig.add_gridspec(1, 2, left=.10, right=.95, top=.70, bottom=.17, wspace=.24).subplots()
    colors = dict(zip(DIFFICULTIES, ("#679da2", "#147867", "#ae7934")))
    for ax, (cap, neat) in zip(axes, ((.5, FINAL), (1.5, FINAL))):
        pairs = [(a, groups[FILTERED, cap][slot]) for slot, a in groups[neat, cap].items()
                 if a["done"] and groups[FILTERED, cap][slot]["done"]]
        maximum = max(max(a["elapsed"], b["elapsed"]) for a, b in pairs)*1.10
        ax.plot((0, maximum), (0, maximum), ls="--", color="#8da4a1", lw=1)
        for d in DIFFICULTIES:
            points = [(a["elapsed"], b["elapsed"]) for a, b in pairs if a["difficulty"] == d]
            if points:
                x, y = np.array(points).T
                ax.scatter(x, y, s=30, color=colors[d], alpha=.78, edgecolor=BG, linewidth=.5, label=d.title())
        ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel=f"{label(neat)} time (s)", ylabel="Recovery PID time (s)")
        ax.set_title(f"{cap:.2f} m/s  /  {len(pairs)} shared finishes", loc="left", pad=12)
        ax.set_axisbelow(True)
        ax.grid()
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "02-corrected-paired-time")


def error_figure(groups):
    fig = page(3, "Physical tracking error and authenticated progress",
               "All 90 episodes count in every mean, including the observed portions of failed runs.",
               "RMSE uses exact distance to the intended route polyline. Progress still requires ordered route gates.", 5.6)
    axes = fig.add_gridspec(1, 2, left=.24, right=.95, top=.69, bottom=.16, wspace=.25).subplots()
    names = []
    for y, (policy, cap) in enumerate(CONDITIONS):
        r = describe(groups[policy, cap])
        names.append(f"{label(policy)}  /  {cap:.2f}")
        for ax, key in zip(axes, ("rmse_mm", "progress_pct")):
            value = r[key]
            ax.barh(y, value, color=COLORS[policy], height=.52)
            ax.text(value, y, f"  {value:.1f}" + ("%" if key == "progress_pct" else ""),
                    va="center", fontsize=10, weight="bold")
    for ax, title in zip(axes, ("Mean lateral RMSE (mm)", "Mean route progress")):
        ax.set_title(title, loc="left", pad=12)
        ax.set(yticks=range(4), yticklabels=names if ax is axes[0] else [""]*4, ylim=(3.7, -.7))
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x")
    axes[0].set_xlim(0, max(describe(groups[p, c])["rmse_mm"] for p, c in CONDITIONS)*1.28)
    axes[1].set(xlim=(0, 124), xticks=(0, 25, 50, 75, 100))
    axes[1].xaxis.set_major_formatter(PercentFormatter(100))
    save(fig, "03-corrected-error-progress")


def correction_figure(groups, old):
    fig = page(4, "How the corrected evaluation changed completion counts",
               "Same frozen controllers and 90 geometries. The revised protocol also widens the corridor and recovery allowance.",
               "Original results use the old gate rule, 45 mm corridor and 1 s departure. Corrected uses 100 mm and 2 s.", 5.5)
    ax = fig.add_gridspec(1, 1, left=.27, right=.91, top=.70, bottom=.16).subplots()
    names = []
    for y, (p, cap) in enumerate(CONDITIONS):
        before = old[p, str(cap)]
        after = describe(groups[p, cap])["finished"]
        names.append(f"{label(p)}  /  {cap:.2f} m/s")
        ax.plot((before, after), (y, y), color="#b8c9c3", lw=5)
        ax.scatter(before, y, facecolor=BG, edgecolor="#6d8581", s=75, lw=1.8, zorder=3)
        ax.scatter(after, y, color=COLORS[p], s=75, zorder=4)
        ax.annotate(str(before), (before, y), (0, 12), textcoords="offset points", ha="center", fontsize=10)
        ax.annotate(str(after), (after, y), (0, -21), textcoords="offset points", ha="center",
                    color=COLORS[p], weight="bold", fontsize=10)
    ax.set(xlim=(0, 97), xticks=(0, 15, 30, 45, 60, 75, 90), xlabel="Tracks finished out of 90",
           yticks=range(4), yticklabels=names, ylim=(3.7, -.7))
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=8, markerfacecolor=BG,
                               markeredgecolor="#6d8581", label="Original evaluation"),
                        Line2D([], [], ls="", marker="o", ms=8, color=TEAL, label="Corrected evaluation")],
               ncol=2, loc="upper left", bbox_to_anchor=(.06, .79), frameon=False)
    save(fig, "04-evaluation-correction")


def load_data():
    results = OUT / "test-episodes.csv"
    assert digest(results) == read_json(OUT / "status.json")["sha256"]
    rows = read_csv(results)
    assert len(rows) == 540 and not any(r["error"] for r in rows)
    rows = [r for r in rows if (r["policy_id"], float(r["evaluation_speed_limit_m_s"])) in CONDITIONS]
    assert len(rows) == 360
    episodes = [{"policy": r["policy_id"], "cap": float(r["evaluation_speed_limit_m_s"]),
                 "slot": int(r["track_slot"]), "hash": r["track_hash"], "difficulty": r["difficulty"],
                 "done": r["completed"] == "True", "error": r["error"], "termination": r["termination"],
                 "progress": float(r["progress"]), "rmse": float(r["geometry_lateral_rmse_m"]),
                 "mae": float(r["geometry_lateral_mae_m"]), "elapsed": float(r["elapsed_s"]),
                 "forward": float(r["average_forward_speed_m_s"])} for r in rows]
    original = read_csv(RUN / "assessment/test-episodes.csv")
    original += read_csv(ROOT / "research/artifacts/neat-speed-results/supplemental-episodes.csv")
    old = Counter((r["policy_id"].replace("speed-trained", "speed-g98"), r["evaluation_speed_limit_m_s"])
                  for r in original if r["completed"] == "True")
    tracks = [r["geometry_hash"] for r in read_json(RUN / "suite/suite.json")["splits"]["test"]]
    return {"episodes": episodes, "tracks": tracks,
            "old": [{"policy": p, "cap": c, "finished": n} for (p, c), n in old.items()]}


def write_report(groups):
    summaries = {(p, c): describe(g) for (p, c), g in groups.items()}
    pairs = {(cap, neat): pair(groups[neat, cap], groups["pid-filtered", cap])
             for cap, neat in ((.5, "speed-g98"), (1.5, "speed-g98"))}
    primary_rows = []
    for p, cap in CONDITIONS:
        r = summaries[p, cap]
        name = "Recovery PID" if p == "pid-filtered" else "NEAT g98"
        primary_rows.append(f"| {name} | {cap:.2f} | {r['finished']}/90 | {r['beginner']}/30 | "
                            f"{r['intermediate']}/30 | {r['advanced']}/30 | {r['failure_pct']:.1f}% | "
                            f"{r['rmse_mm']:.2f} | {r['mae_mm']:.2f} | {r['progress_pct']:.1f}% |")
    paired_rows = []
    for (cap, neat), r in pairs.items():
        paired_rows.append(f"| {cap:.2f} | g98 | {r['both_finished']} | "
                           f"{r['first_only']} | {r['second_only']} | {r['neither']} | "
                           f"{r['first_time_s']:.2f} / {r['second_time_s']:.2f} | "
                           f"{r['first_rmse_mm']:.2f} / {r['second_rmse_mm']:.2f} |")
    low, high = summaries["speed-g98", .5], summaries["speed-g98", 1.5]
    pl, ph = summaries["pid-filtered", .5], summaries["pid-filtered", 1.5]
    cross = pair(groups["speed-g98", 1.5], groups["speed-g98", .5])
    failures = "\n".join(f"| {('Recovery PID' if p == 'pid-filtered' else p)} | {cap:.2f} | "
                          f"{summaries[p,cap]['off_route']} | {summaries[p,cap]['stalled']} | "
                          f"{summaries[p,cap]['other_failures']} |" for p, cap in CONDITIONS)
    report = f"""# NEAT speed results after the evaluator correction

Regenerated on 5 November 2026 from **360 matched test episodes** on the same 90 tracks. **Every NEAT result in this report uses the same frozen speed-trained g98 genome**, evaluated at either a 0.50 or 1.50 m/s wheel speed cap.

At the **0.50 m/s** cap, speed-trained NEAT **g98 finished {low['finished']}/90** and recovery PID finished **{pl['finished']}/90**. At **1.50 m/s**, the same NEAT **g98 finished {high['finished']}/90** and the same recovery PID finished **{ph['finished']}/90**.

The final speed-trained policy transfers imperfectly to the lower actuator cap. All {low['stalled']} low-cap NEAT failures ended after five seconds without new ordered route progress. The recovery PID has the higher completion count at 0.50 m/s, while g98 has the higher completion count at 1.50 m/s.

## Matched test results

Each condition uses the same 30 beginner, 30 intermediate, and 30 advanced geometries. The NEAT genome and PID tuning are fixed across caps. All 360 episodes completed evaluation without a technical error.

| Controller | Cap (m/s) | Finished | Beginner | Intermediate | Advanced | Failure rate | RMSE (mm) | MAE (mm) | Mean progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(primary_rows)}

![Corrected completion by controller, speed cap and difficulty](artifacts/neat-speed-results-corrected/01-corrected-completion.png)

## Time and error on shared finishes

The time and error means below use only tracks both members of the pair finished. Each row reports its own denominator.

| Cap (m/s) | NEAT | Both finished | NEAT only | PID only | Neither | Mean time, NEAT / PID (s) | RMSE, NEAT / PID (mm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(paired_rows)}

At 1.50 m/s, PID had a lower mean time on the {pairs[1.5, 'speed-g98']['both_finished']} shared finishes, while g98 completed {pairs[1.5, 'speed-g98']['first_only']} additional tracks. The shared-finish table keeps that timing comparison on identical geometry.

![Paired completion times on shared finishes](artifacts/neat-speed-results-corrected/02-corrected-paired-time.png)

Across the **{cross['both_finished']} tracks g98 finished at both caps**, it averaged **{cross['first_time_s']:.2f} s at 1.50 m/s**, versus **{cross['second_time_s']:.2f} s at 0.50 m/s**, using **{100*(1-cross['first_time_s']/cross['second_time_s']):.1f}% less time**. Its corresponding mean geometric RMSE was **{cross['first_rmse_mm']:.2f} mm at 1.50 m/s** and **{cross['second_rmse_mm']:.2f} mm at 0.50 m/s**. This comparison holds the genome fixed and changes the speed cap.

The network uses the same sensor and history input schema at both caps. The changed cap rescales its motor commands. These outcomes measure how the speed-trained policy transfers when the actuator cap changes.

## Tracking error and failure reasons

Lateral RMSE and MAE measure the exact shortest distance from the chassis center to the intended route polyline. They average per-episode errors with equal weight for each track and include failed episodes through termination. The metric follows the intended centerline through gaps and excludes branch paint. Ordered progress remains a separate measure, so driving backward along the line can have low geometric error while earning no additional progress.

![Geometric lateral error and authenticated route progress](artifacts/neat-speed-results-corrected/03-corrected-error-progress.png)

| Controller | Cap (m/s) | Off route | No new route progress | Other failures |
| --- | ---: | ---: | ---: | ---: |
{failures}

## What changed in evaluation

The earlier evaluator could miss a corner gate while the robot was still tracking the line, leaving its local route cursor behind. Version **progress-1.2** makes gate tolerance cover corner cuts within the accepted corridor and resets the departure timer whenever a line sensor is active. This reassessment uses the same corrected protocol as the recreated videos, with a **100 mm corridor half-width**, **two seconds** of continuous departure with no sensor contact, and **five seconds** without new ordered progress for a stall. All controllers use the same rules.

![Completion counts under the original and corrected protocols](artifacts/neat-speed-results-corrected/04-evaluation-correction.png)

The corridor and recovery allowance also changed from 45 mm and one second, so this figure measures the combined evaluator correction. The report's current error columns use geometric distance. The archived assessment's local gate-relative errors remain in the raw records for provenance.

## Frozen controllers and methodology

The recovery PID is the saved filtered baseline retained by the gains-only tuning experiment. Its P, I and D values are **0.7239850053**, **0.1681344890**, and **0.02158228171**, with base command **0.75**. Its derivative filter, bounded integral, error-dependent slowdown, 0.738 s last-command hold, and remembered-direction search all remain enabled. Corner supervision is disabled.

The original tuning robot used a **0.35 m/s** cap. The later search retained the original filtered gains. This study transfers that exact tuning to 0.50 and 1.50 m/s. The simulator's 600 Hz physics, 60 Hz control, robot geometry, and actuator limits match the speed experiment apart from the requested cap.

The g98 genome comes from the speed run that warm-started from g94 and evolved a nominal population of 100 for 100 generations, using four shared mixed-difficulty tracks per generation. The cap schedule rose through 0.50, 0.65, 0.80, 1.00, 1.25, and 1.50 m/s. Training and checkpoint selection used the original evaluator. g98 was selected by that run's validation rule before this reassessment. Both reported NEAT conditions load this exact selected artifact.

This is a retrospective comparison on a known test cohort from one evolution lineage. The reassessment retains the previously selected models. The [original protocol](../runs/neat-speed-100-20261104/protocol.json) and [frozen PID configuration](../runs/pid-gains-20261029/selected-pid_filtered.json) record their settings.

## Reproduce and present

- [Executed comparison notebook](NEAT_Speed_PID_Comparison.ipynb) embeds the corrected data, code, plots and tables.
- [Summary table](artifacts/neat-speed-results-corrected/test-summary.csv) and [paired results](artifacts/neat-speed-results-corrected/paired-summary.csv) contain the four reported conditions. The [raw episode archive](artifacts/neat-speed-results-corrected/test-episodes.csv) contains these 360 episodes plus 180 historical-reference evaluations of g94 and g00.
- [Evaluation freeze](artifacts/neat-speed-results-corrected/evaluation-freeze.json), [source snapshot](artifacts/neat-speed-results-corrected/source.zip), and [full telemetry](artifacts/neat-speed-results-corrected/telemetry/) preserve the inputs, gains, model hashes, and every replay.
- PNG and SVG versions of all four figures are in [the corrected figure folder](artifacts/neat-speed-results-corrected/).

From the repository root, regenerate the current report and notebook with the command below.

```powershell
./.venv/Scripts/python.exe research/write_corrected_speed_report.py
```
"""
    (ROOT / "research/NEAT_Speed_Results.md").write_text(report, encoding="utf-8")
    export_csv(OUT / "test-summary.csv", [{"policy": p, "cap_m_s": cap, **r} for (p, cap), r in summaries.items()])
    exported = [{"first": neat, "first_cap_m_s": cap, "second": "pid-filtered", "second_cap_m_s": cap, **r}
                for (cap, neat), r in pairs.items()]
    exported.append({"first": "speed-g98", "first_cap_m_s": 1.5, "second": "speed-g98",
                     "second_cap_m_s": .5, **cross})
    export_csv(OUT / "paired-summary.csv", exported)


def main():
    data = load_data()
    groups = group_episodes(data["episodes"], data["tracks"])
    plotting.OUT = OUT
    common = SETUP+"\n"+source_of(page, save, label, show_table)+f"\nCONDITIONS = {CONDITIONS!r}"
    grouping = source_of(group_episodes, describe, pair)+"\ngroups = group_episodes(data['episodes'], data['tracks'])"
    plotting.build_notebook(ROOT / "research/NEAT_Speed_PID_Comparison.ipynb", [
        ("markdown", """# Corrected NEAT speed comparison

Regenerated on 5 November 2026. Compare the same speed-trained g98 genome at 0.50 and 1.50 m/s with the same frozen recovery PID at each cap. The embedded 360 corrected episode summaries reproduce all four report figures. Run all cells with NumPy and Matplotlib in Jupyter.

The corrected evaluator uses a 100 mm corridor, two seconds of departure without sensor contact, corridor-consistent gates, and five seconds without new ordered progress. Error means use exact distance to the intended route polyline."""),
        ("code", common), ("data", embedded(data)), ("code", grouping),
        ("code", source_of(completion_figure)+"\ncompletion_figure(groups)"),
        ("code", source_of(timing_figure)+"\ntiming_figure(groups)"),
        ("code", source_of(error_figure)+"\nerror_figure(groups)"),
        ("code", source_of(correction_figure)+
         "\nold = {(r['policy'], r['cap']): r['finished'] for r in data['old']}\ncorrection_figure(groups, old)"),
        ("code", """rows = []
for p, cap in CONDITIONS:
    r = describe(groups[p, cap])
    rows.append([label(p), f"{cap:.2f}", f"{r['finished']}/90", f"{r['rmse_mm']:.2f}", f"{r['progress_pct']:.1f}%"])
show_table(["Controller", "Cap (m/s)", "Finished", "RMSE (mm)", "Progress"], rows)""")
    ])
    write_report(groups)
    print("Regenerated NEAT_Speed_Results.md, comparison notebook, four PNG/SVG figures and summary tables.")


if __name__ == "__main__":
    main()
