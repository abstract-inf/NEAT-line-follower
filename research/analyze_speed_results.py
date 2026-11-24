"""Build the speed study's figures, tables, and standalone notebooks from saved episodes."""

import base64
from collections import Counter
import csv
import gzip
import inspect
import json
from pathlib import Path, PureWindowsPath
from statistics import mean

import matplotlib

matplotlib.use("Agg")

from evaluate_speed_comparison import OUT, ROOT, RUN, digest, read_csv, read_json
from line_follower.tracks import TrackSpec

DIFFICULTIES = ("beginner", "intermediate", "advanced")

SETUP = '''import base64
from collections import Counter
import gzip
from io import BytesIO
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

EXPORT_DIR = None  # Set to a Path to also save PNG and SVG files.
BG, INK, MUTED, GRID = "#f6f5f0", "#17262c", "#4c6168", "#d7dfdd"
TEAL, BLUE, CLAY, OCHRE = "#087f72", "#416994", "#ba533c", "#a87516"
PARENT, EARLY, FINAL = "g94-parent", "speed-g00", "speed-g98"
PLAIN, FILTERED = "pid-plain", "pid-filtered"
COMPARATORS = (PARENT, EARLY, FINAL, PLAIN, FILTERED)
TRACE_POLICIES = (PARENT, FINAL, PLAIN, FILTERED)
LABELS = {PARENT: "NEAT g94 (parent)", EARLY: "NEAT g00", FINAL: "NEAT g98",
          PLAIN: "Plain PID", FILTERED: "Filtered PID + recovery"}
COLORS = dict(zip(COMPARATORS, (BLUE, "#795e92", TEAL, CLAY, OCHRE)))
DIFFICULTIES = ("beginner", "intermediate", "advanced")
CAPS = (.50, .65, .80, 1.00, 1.25, 1.50)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "svg.hashsalt": "neat-speed-results"
})
'''


def page(number, title, subtitle, footer, height=5.4):
    fig = plt.figure(figsize=(12, height))
    fig.text(.06, .96, f"SPEED STUDY  /  {number:02d}", va="top", color=TEAL, weight="bold", fontsize=10)
    fig.text(.06, .902, title, va="top", size=22, weight="bold")
    fig.text(.06, .824, subtitle, va="top", color=MUTED, size=11)
    fig.text(.06, .041, footer, va="bottom", color=MUTED, size=9)
    return fig


def save(fig, name):
    png = BytesIO()
    fig.savefig(png, format="png", dpi=160)
    if EXPORT_DIR is not None:
        path = Path(EXPORT_DIR)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.png").write_bytes(png.getvalue())
        fig.savefig(path / f"{name}.svg", metadata={"Date": None})
    display({"image/png": base64.b64encode(png.getvalue()).decode("ascii"), "text/plain": name},
            raw=True, metadata={"image/png": {"width": 1000}})
    plt.close(fig)


def label(policy):
    return LABELS.get(policy, f"NEAT {policy.removeprefix('speed-')}")


def group_episodes(records, tracks):
    grouped = {}
    for r in records:
        group = grouped.setdefault((r["policy"], r["cap"]), {})
        assert r["slot"] not in group and not r["error"]
        group[r["slot"]] = r
    for group in grouped.values():
        assert set(group) == set(range(90))
        assert [group[i]["hash"] for i in range(90)] == tracks
        assert Counter(r["difficulty"] for r in group.values()) == dict.fromkeys(DIFFICULTIES, 30)
    return grouped


def describe(group):
    rows = list(group.values())
    finished = [r for r in rows if r["done"]]
    reasons = Counter(r["termination"] for r in rows)
    return {"n": len(rows), "finished": len(finished), "failure_pct": 100*(1-len(finished)/len(rows)),
            **{d: sum(r["difficulty"] == d for r in finished) for d in DIFFICULTIES},
            "rmse_mm": mean(r["rmse"] for r in rows)*1000, "mae_mm": mean(r["mae"] for r in rows)*1000,
            "progress_pct": mean(r["progress"] for r in rows)*100,
            "observed_forward_m_s": mean(r["forward"] for r in rows),
            "finished_time_s": mean(r["elapsed"] for r in finished) if finished else None,
            "off_route": reasons["off_route"], "stalled": reasons["stalled"],
            "other_failures": len(rows)-len(finished)-reasons["off_route"]-reasons["stalled"]}


def pair(first, second):
    assert first.keys() == second.keys()
    rows = [(a, second[i]) for i, a in first.items()]
    shared = [(a, b) for a, b in rows if a["done"] and b["done"]]
    result = {"both_finished": len(shared),
              "first_only": sum(a["done"] and not b["done"] for a, b in rows),
              "second_only": sum(b["done"] and not a["done"] for a, b in rows),
              "neither": sum(not a["done"] and not b["done"] for a, b in rows)}
    for i, side in enumerate(("first", "second")):
        result[f"{side}_rmse_mm"] = mean(p[i]["rmse"] for p in shared)*1000 if shared else None
        result[f"{side}_time_s"] = mean(p[i]["elapsed"] for p in shared) if shared else None
    return result


def learning_plot(generations, choices):
    fig = page(1, "Learning across the six speed stages",
               "Each point is a generation champion evaluated on the same 30 validation tracks.",
               "Diamonds mark the checkpoint selected within each speed stage by completion, then validation fitness.", 5.9)
    ax = fig.add_gridspec(1, 1, left=.10, right=.96, bottom=.15, top=.72).subplots()
    colors = ["#e4ecea", "#edf1ec", "#e4ecea", "#edf1ec", "#e4ecea", "#edf1ec"]
    for cap, color in zip(CAPS, colors):
        phase = [(g, n) for g, c, n in generations if c == cap]
        x, y = np.array(phase).T
        ax.axvspan(x[0]-.5, x[-1]+.5, color=color, lw=0)
        ax.plot(x, y, ".-", color=BLUE, lw=1.1, ms=5, alpha=.75)
        ax.step(x, np.maximum.accumulate(y), where="post", color=TEAL, lw=1.6)
        ax.text(x.mean(), 31.7, f"{cap:.2f}", color=MUTED, ha="center", fontsize=10, weight="bold")
    for row in choices.values():
        gen, n = row["generation"], round(row["validation_completion"]*30)
        ax.scatter(gen, n, marker="D", s=45, color=TEAL, edgecolor=BG, zorder=4)
        align = "left" if gen == 0 else "right" if gen > 90 else "center"
        ax.annotate(f"g{gen:02d}", (gen, n), xytext=(0, -15), textcoords="offset points",
                    ha=align, color=INK, fontsize=9, weight="bold")
    ax.set(xlim=(-.5, 99.5), ylim=(0, 33.2), xlabel="Generation", ylabel="Validation tracks finished",
           yticks=range(0, 31, 5))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    fig.text(.10, .754, "WHEEL SPEED CAP (m/s)", color=MUTED, weight="bold", fontsize=9)
    save(fig, "01-speed-curriculum")


def transfer_plot(groups, choices):
    policies = [PARENT, *choices]
    matrix = np.array([[describe(groups[p, c])["finished"] if (p, c) in groups else np.nan
                        for c in CAPS] for p in policies])
    fig = page(2, "The best checkpoint depends on the speed cap",
               "Test tracks finished out of 90. Each evaluated cell uses the same geometry cohort.",
               "Outlined cells show the checkpoint's own training cap. A dash indicates an unevaluated condition.", 5.9)
    ax = fig.add_gridspec(1, 1, left=.23, right=.88, bottom=.15, top=.727).subplots()
    cmap = plt.get_cmap("GnBu").with_extremes(bad="#e8ebe7")
    mesh = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=90, aspect="auto")
    ax.xaxis.tick_top()
    ax.set(xticks=range(6), xticklabels=[f"{c:.2f} m/s" for c in CAPS],
           yticks=range(len(policies)), yticklabels=[label(p) for p in policies])
    ax.tick_params(length=0, pad=9)
    for i, p in enumerate(policies):
        for j, cap in enumerate(CAPS):
            n = matrix[i, j]
            ax.text(j, i, "–" if np.isnan(n) else f"{n:.0f}", ha="center", va="center",
                    color="white" if n >= 65 else INK if np.isfinite(n) else MUTED, weight="bold")
            if p in choices and cap == choices[p]["training_cap_m_s"]:
                ax.add_patch(plt.Rectangle((j-.48, i-.47), .96, .94, fill=False, lw=2.1,
                                           edgecolor="white" if n >= 65 else TEAL))
    cbar = fig.colorbar(mesh, cax=fig.add_axes([.905, .15, .015, .577]), ticks=[0, 30, 60, 90])
    cbar.ax.tick_params(labelsize=9)
    save(fig, "02-speed-transfer")


def ranking_plot(groups, choices):
    rank = [(p, describe(groups[p, 1.5])) for p in [PARENT, *choices]]
    rank.sort(key=lambda item: (-item[1]["finished"], item[1]["rmse_mm"]))
    fig = page(3, "g98 leads the 1.50 m/s checkpoint comparison",
               "Each NEAT checkpoint drives the same 90 test tracks at the same wheel target cap.",
               "RMSE averages per-episode lateral errors and includes the observed portion of failed runs.", 5.7)
    left, right = fig.add_gridspec(1, 2, left=.22, right=.94, bottom=.16, top=.72,
                                  wspace=.28, width_ratios=[1, 1]).subplots()
    colors = [TEAL if p == FINAL else BLUE if p == PARENT else "#769da6" for p, _ in rank]
    for ax, key, title, xlabel, xlim, fmt in (
            (left, "finished", "Courses completed", "Tracks finished", 94, "{:d} / 90"),
            (right, "rmse_mm", "Lateral tracking error", "Mean RMSE (mm)", 245, "{:.1f}")):
        values = [r[key] for _, r in rank]
        ax.barh(range(len(rank)), values, color=colors, height=.6)
        for i, value in enumerate(values):
            ax.text(value+xlim*.022, i, fmt.format(value), va="center", color=INK, weight="bold", fontsize=10)
        ax.set(yticks=range(len(rank)), yticklabels=[label(p) if ax is left else "" for p, _ in rank],
               xlim=(0, xlim), xlabel=xlabel)
        ax.set_title(title, loc="left", pad=12)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x")
    save(fig, "03-fast-checkpoint-ranking")


def completion_plot(groups):
    fig = page(4, "g00 leads at 0.50 m/s and g98 leads at 1.50 m/s",
               "The same 30 beginner, 30 intermediate, and 30 advanced tracks repeat in every condition.",
               "Bars divide each controller's completions by difficulty. Labels give the total out of 90.", 6.0)
    axes = fig.add_gridspec(1, 2, left=.225, right=.95, bottom=.16, top=.69, wspace=.13).subplots()
    shades = ("#98b9bc", "#47938f", "#20595f")
    for ax, cap in zip(axes, (.5, 1.5)):
        for i, p in enumerate(COMPARATORS):
            result = describe(groups[p, cap])
            offset = 0
            for d, color in zip(DIFFICULTIES, shades):
                n = result[d]
                ax.barh(i, n, left=offset, color=color, height=.55)
                if n >= 8:
                    ax.text(offset+n/2, i, n, ha="center", va="center",
                            color=INK if d == "beginner" else "white", weight="bold", fontsize=10)
                offset += n
            ax.text(offset+2, i, f"{offset}", ha="left", va="center",
                    color=COLORS[p], weight="bold", fontsize=11)
        ax.set(xlim=(0, 100), xticks=(0, 30, 60, 90), xlabel="Tracks finished", yticks=range(len(COMPARATORS)),
               yticklabels=[label(p) if ax is axes[0] else "" for p in COMPARATORS])
        ax.set_title(f"{cap:.2f} m/s cap", loc="left", pad=12)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x")
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=color, label=d.title(), markersize=8)
                        for d, color in zip(DIFFICULTIES, shades)],
               loc="upper left", bbox_to_anchor=(.06, .795), ncol=3, frameon=False)
    save(fig, "04-neat-pid-completion")


def quality_plot(groups):
    fig = page(5, "Tracking error and how far each controller got",
               "All 90 episodes count in each mean. The error and progress axes measure different outcomes.",
               "Progress uses ordered route gates. Error measures observed lateral deviation through termination.", 6.2)
    axes = fig.add_gridspec(1, 2, left=.09, right=.94, bottom=.16, top=.69, wspace=.26).subplots()
    for ax, metric, title, max_x, fmt in (
            (axes[0], "rmse_mm", "Mean lateral RMSE (mm)", 490, "{:.1f}"),
            (axes[1], "progress_pct", "Mean route progress", 118, "{:.0f}%")):
        for j, cap in enumerate((.5, 1.5)):
            for i, p in enumerate(COMPARATORS):
                value = describe(groups[p, cap])[metric]
                y = i + j*6.1
                ax.hlines(y, 0, value, color=COLORS[p], alpha=.27, lw=2)
                ax.scatter(value, y, color=COLORS[p], s=50, zorder=3)
                ax.text(value+max_x*.026, y, fmt.format(value), va="center", fontsize=10, weight="bold")
        ax.axhline(5.05, color=GRID)
        ax.text(max_x, -.70, "0.50 m/s", ha="right", va="center", color=MUTED, fontsize=9, weight="bold")
        ax.text(max_x, 5.64, "1.50 m/s", ha="right", va="center", color=MUTED, fontsize=9, weight="bold")
        ax.set(xlim=(0, max_x), ylim=(11, -1.22), yticks=[])
        ax.set_title(title, loc="left", pad=12)
        ax.set_axisbelow(True)
        ax.grid(axis="x")
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].xaxis.set_major_formatter(PercentFormatter(100))
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS[p], label=label(p))
                        for p in COMPARATORS],
               loc="upper left", bbox_to_anchor=(.06, .795), ncol=5, frameon=False)
    save(fig, "05-neat-pid-error-progress")


def trajectory_plot(examples, groups):
    fig = page(6, "Two routes show where the controllers diverged",
               "Recorded 1.50 m/s runs on the prelogged intermediate and advanced tracks 031 and 032.",
               "Grey is the route centerline. Open circle is start, square is finish, and × is off route.", 8.1)
    axes = fig.add_gridspec(2, 2, left=.10, right=.94, bottom=.12, top=.72, wspace=.27,
                            hspace=.43, height_ratios=[1.55, 1]).subplots()
    styles = {PARENT: "--", FINAL: "-", PLAIN: ":", FILTERED: "-."}
    for (slot, example), (xy, clock) in zip(examples.items(), axes.T):
        route = np.array(example["route"])
        xy.plot(*route.T, color="#a7b1ae", lw=4, solid_capstyle="round", zorder=1)
        xy.scatter(*route[0], facecolors=BG, edgecolors=INK, s=33, zorder=5)
        for p in TRACE_POLICIES:
            row = groups[p, 1.5][int(slot)]
            t, x, y, progress = np.array(example["traces"][p]).T
            marker = "s" if row["done"] else "x"
            xy.plot(x, y, ls=styles[p], color=COLORS[p], lw=1.8)
            xy.scatter(x[-1], y[-1], color=COLORS[p], marker=marker, s=36, zorder=4)
            clock.plot(np.r_[0, t], np.r_[0, progress*100], ls=styles[p], color=COLORS[p], lw=1.8)
            clock.scatter(t[-1], progress[-1]*100, color=COLORS[p], marker=marker, s=36, zorder=4)
        xy.set(xlabel="World X (m)", aspect="equal")
        xy.set_title(f"Track {int(slot):03d}  /  {example['difficulty'].title()}", loc="left", pad=9)
        clock.set(xlabel="Simulated time (s)", xlim=(0, None), ylim=(0, 106), yticks=(0, 50, 100))
        clock.yaxis.set_major_formatter(PercentFormatter(100))
        clock.set_axisbelow(True)
        clock.grid()
    axes[0, 0].set_ylabel("World Y (m)")
    axes[1, 0].set_ylabel("Route progress")
    fig.legend(handles=[Line2D([], [], ls=styles[p], lw=2, color=COLORS[p], label=label(p)) for p in TRACE_POLICIES],
               loc="upper left", bbox_to_anchor=(.06, .79), ncol=4, frameon=False)
    save(fig, "06-matched-trajectories")


def examples_from_logs(rows, entries):
    examples = {}
    for slot in (31, 32):
        track = TrackSpec.load(RUN / "suite" / entries[slot]["path"])
        traces = {}
        for row in rows:
            old_policy = row["policy_id"]
            if int(row["track_slot"]) != slot or float(row["evaluation_speed_limit_m_s"]) != 1.5 or not row["telemetry"]:
                continue
            if old_policy in ("g94-parent", "speed-trained"):
                path = RUN / "assessment" / f"{old_policy}-1.50" / "telemetry/test" / PureWindowsPath(row["telemetry"]).name
            else:
                path = ROOT / row["telemetry"]
            samples = read_csv(path)
            assert len(samples) == int(row["control_steps"])
            assert abs(float(samples[-1]["time_s"])-float(row["elapsed_s"])) < 1e-7
            traces[old_policy.replace("speed-trained", "speed-g98")] = [
                [round(float(r[k]), 7) for k in ("time_s", "x_m", "y_m", "progress")] for r in samples]
        examples[slot] = {"difficulty": track.difficulty, "route": track.route.points.round(6).tolist(), "traces": traces}
    return examples


def load_inputs():
    freeze = read_json(OUT / "comparison-freeze.json")
    status = read_json(OUT / "supplemental-status.json")
    original = RUN / "assessment/test-episodes.csv"
    supplemental = OUT / "supplemental-episodes.csv"
    suite = RUN / "suite/suite.json"
    assert digest(original) == freeze["original_assessment_sha256"]
    assert digest(supplemental) == status["sha256"] and digest(suite) == freeze["suite_sha256"]
    source = read_csv(original) + read_csv(supplemental)
    assert len(source) == 360 + status["episodes"] == 1980
    tracks = read_json(suite)["splits"]["test"]
    episodes = [{"policy": r["policy_id"].replace("speed-trained", "speed-g98"),
                 "cap": float(r["evaluation_speed_limit_m_s"]), "slot": int(r["track_slot"]),
                 "hash": r["track_hash"], "difficulty": r["difficulty"], "done": r["completed"] == "True",
                 "error": r["error"], "termination": r["termination"], "progress": float(r["progress"]),
                 "rmse": float(r["lateral_rmse_m"]), "mae": float(r["mean_absolute_lateral_error_m"]),
                 "forward": float(r["average_forward_speed_m_s"]), "elapsed": float(r["elapsed_s"])}
                for r in source]
    generations = [[int(r["generation"]), float(r["wheel_speed_cap_m_s"]),
                    round(float(r["validation_completion_rate"])*30)]
                   for r in read_csv(RUN / "training/generations.csv")]
    return {"tracks": [r["geometry_hash"] for r in tracks], "episodes": episodes,
            "choices": freeze["phase_checkpoints"], "generations": generations,
            "traces": examples_from_logs(source, tracks)}


def source_of(*functions):
    return "\n\n".join(inspect.getsource(f).strip() for f in functions)


def show_table(headers, rows):
    lines = [headers, ["---"]*len(headers), *rows]
    markdown = "\n".join("| " + " | ".join(row) + " |" for row in lines)
    display({"text/markdown": markdown, "text/plain": markdown}, raw=True, metadata={})


def build_notebook(path, blocks):
    def capture(bundle, *, raw, metadata):
        outputs.append({"output_type": "display_data", "data": bundle, "metadata": metadata})

    namespace = {"display": capture}
    cells = []
    execution = 0
    for kind, source in blocks:
        metadata = {}
        if kind == "data":
            metadata = {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]}
            kind = "code"
        cell = {"id": f"speed-{len(cells)+1:02d}", "cell_type": kind,
                "metadata": metadata, "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            outputs = []
            execution += 1
            exec(compile(source, f"{path.name}:{execution}", "exec"), namespace)
            namespace["EXPORT_DIR"] = OUT
            cell.update(execution_count=execution, outputs=outputs)
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                 "name": "python3"}, "language_info": {"name": "python", "file_extension": ".py"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")


def embedded(data):
    payload = base64.b64encode(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)).decode()
    chunks = "\n".join(f"    {payload[i:i+100]!r}" for i in range(0, len(payload), 100))
    return f"payload = (\n{chunks}\n)\ndata = json.loads(gzip.decompress(base64.b64decode(payload)))"


def export_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    data = load_inputs()
    # Notebook kernels already provide IPython display. The batch builder supplies its own collector.
    common = SETUP + "\n" + source_of(page, save, label, show_table)
    grouping = source_of(group_episodes, describe, pair) + "\n\ngroups = group_episodes(data['episodes'], data['tracks'])"
    neat = {k: v for k, v in data.items() if k != "traces"}
    neat["episodes"] = [r for r in data["episodes"] if not r["policy"].startswith("pid")]
    comparison = {k: data[k] for k in ("tracks", "traces")}
    comparison["episodes"] = [r for r in data["episodes"] if r["policy"] in
                              ("g94-parent", "speed-g00", "speed-g98", "pid-plain", "pid-filtered")]
    build_notebook(ROOT / "research/NEAT_Speed_Checkpoints.ipynb", [
        ("markdown", """# NEAT speed checkpoints

Reproduce figures 1–3 from the speed report. The compressed cell carries the matched per-episode data, generation history, and frozen selection metadata. Run all cells with Python 3, NumPy, and Matplotlib in Jupyter. The embedded results need no simulator, checkpoint files, or repository.

The 0.50 and 1.50 m/s columns measure speed transfer. Each intermediate checkpoint also ran at its own training cap. The six phase winners were selected from validation before the supplemental replays."""),
        ("code", common), ("data", embedded(neat)), ("code", grouping),
        ("markdown", """## Validation through the curriculum

Diamonds identify the phase winner. The teal line accumulates the best completion count within each phase."""),
        ("code", source_of(learning_plot)+"\n\nlearning_plot(data['generations'], data['choices'])"),
        ("markdown", """## Completion and tracking at matched caps

Compare each checkpoint in the 1.50 m/s column to hold the cap and tracks constant. Mean lateral RMSE includes each episode through its own termination."""),
        ("code", source_of(transfer_plot)+"\n\ntransfer_plot(groups, data['choices'])"),
        ("code", source_of(ranking_plot)+"\n\nranking_plot(groups, data['choices'])"),
        ("code", """rank = [(label(p), describe(group)) for (p, cap), group in groups.items() if cap == 1.50]
rank.sort(key=lambda item: (-item[1]["finished"], item[1]["rmse_mm"]))
show_table(["Checkpoint", "Finished", "Failure rate", "Mean RMSE (mm)", "Mean MAE (mm)"],
           [[name, f"{r['finished']}/{r['n']}", f"{r['failure_pct']:.1f}%", f"{r['rmse_mm']:.2f}",
             f"{r['mae_mm']:.2f}"] for name, r in rank])"""),
        ("markdown", """## Native-cap tradeoff on shared finishes

Pair g98 at 1.50 m/s with g00 at 0.50 m/s to compare actual completion times. The two controllers run at different wheel target caps."""),
        ("code", """r = pair(groups[FINAL, 1.50], groups[EARLY, .50])
show_table(["Shared finishes", "Mean time (s), g98 / g00", "Mean RMSE (mm), g98 / g00"],
           [[str(r['both_finished']), f"{r['first_time_s']:.2f} / {r['second_time_s']:.2f}",
             f"{r['first_rmse_mm']:.2f} / {r['second_rmse_mm']:.2f}"]])"""),
    ])
    build_notebook(ROOT / "research/NEAT_Speed_PID_Comparison.ipynb", [
        ("markdown", """# NEAT speed models versus two PID baselines

Reproduce figures 4–6 from the speed report. Embedded data include all 900 episode rows for these five controllers, plus trajectories and geometry for two logged examples. Run all cells with Python 3, NumPy, and Matplotlib in Jupyter.

Plain PID uses a fixed forward command and continues straight if all sensors lose the line. Filtered PID uses the same P/I/D gains and base command with filtering, slowdown, and remembered-direction recovery. All controllers use the same 90 tracks and simulator settings at each cap."""),
        ("code", common), ("data", embedded(comparison)), ("code", grouping),
        ("markdown", """## Completion by track difficulty"""),
        ("code", source_of(completion_plot)+"\n\ncompletion_plot(groups)"),
        ("markdown", """## Observed lateral error, progress, and example trajectories

All episodes count equally in the error and progress means. Track 031 and 032 traces illustrate two individual outcomes. The cohort-wide completion bars give their broader context."""),
        ("code", source_of(quality_plot)+"\n\nquality_plot(groups)"),
        ("code", source_of(trajectory_plot)+"\n\ntrajectory_plot(data['traces'], groups)"),
        ("markdown", """## Timing and error on shared finishes

Pair on track slot after verifying the geometry hash. Each row uses the tracks both controllers finished at that cap."""),
        ("code", """rows = []
for cap, neat in ((.50, EARLY), (1.50, FINAL)):
    for pid in (PLAIN, FILTERED):
        r = pair(groups[neat, cap], groups[pid, cap])
        rows.append([f"{cap:.2f}", label(neat), label(pid), str(r["both_finished"]),
                     f"{r['first_time_s']:.2f} / {r['second_time_s']:.2f}",
                     f"{r['first_rmse_mm']:.2f} / {r['second_rmse_mm']:.2f}"])
show_table(["Cap (m/s)", "NEAT", "PID", "Both finished", "Time (s), NEAT / PID",
            "RMSE (mm), NEAT / PID"], rows)"""),
    ])

    groups = group_episodes(data["episodes"], data["tracks"])
    export_csv(OUT / "test-summary.csv", [{"policy": p, "cap_m_s": c, **describe(group)}
                                         for (p, c), group in sorted(groups.items())])
    pairs = [(first, cap, second, cap)
             for cap, first in ((.5, "speed-g00"), (.5, "g94-parent"), (1.5, "speed-g98"))
             for second in ("pid-plain", "pid-filtered")]
    pairs.extend(("pid-filtered", cap, "pid-plain", cap) for cap in (.5, 1.5))
    pairs.extend(("speed-g98", 1.5, p, 1.5) for p in ["g94-parent", *data["choices"]] if p != "speed-g98")
    pairs.extend(("speed-g00", .5, p, .5) for p in ["g94-parent", *data["choices"]] if p != "speed-g00")
    pairs.extend(("speed-g98", 1.5, p, .5) for p in ("speed-g00", "g94-parent"))
    export_csv(OUT / "paired-summary.csv", [{"first": a, "first_cap_m_s": ac, "second": b, "second_cap_m_s": bc,
                                           **pair(groups[a, ac], groups[b, bc])} for a, ac, b, bc in pairs])
    print(f"Built two notebooks and speed study artifacts in {OUT}")


if __name__ == "__main__":
    main()
