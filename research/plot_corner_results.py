"""Plot the completed corner-training run from saved records, without simulations."""

from collections import Counter
import csv
from pathlib import Path, PureWindowsPath
from statistics import mean

from plot_neat_pid_comparison import (
    GRID, INK, MUTED, PAPER, ROOT, completed, digest, page, read_csv, read_json,
)
from line_follower.tracks import TrackMask, TrackSpec
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np


RUN = ROOT / "runs/neat-corner-100-20261104"
OUT = ROOT / "research/artifacts/neat-corner-results"
POLICIES = {
    "neat-fresh": ("NEAT g94", "#6743a4"),
    "neat-g109": ("NEAT g109", "#657281"),
    "pid-filtered": ("Filtered PID", "#3976a9"),
    "pid-supervised-original": ("Supervised PID", "#9e641b"),
    "pid-supervised-tuned": ("Tuned supervised PID", "#087e6c"),
}
LEVELS = ("beginner", "intermediate", "advanced")


def load_results():
    training = RUN / "training"
    assessment = RUN / "assessment"
    manifest = RUN / "suite/suite.json"
    status = read_json(assessment / "status.json")
    freeze = read_json(assessment / "selection-freeze.json")
    protocol = read_json(RUN / "protocol.json")
    assert status["complete"] and status["episodes"] == 450
    assert read_json(training / "status.json")["complete"]
    assert digest(manifest) == freeze["suite_sha256"] == protocol["suite_sha256"]
    assert digest(assessment / "frozen-best.pkl") == digest(training / "best.pkl") == freeze["new_neat_sha256"]
    assert digest(ROOT / "runs/28-10-2026-hard/best.pkl") == freeze["old_neat_sha256"]
    assert digest(assessment / "test-episodes.csv") == status["results_sha256"]

    suite = read_json(manifest)
    tracks = suite["splits"]["test"]
    hashes = [track["geometry_hash"] for track in tracks]
    assert len(set(hashes)) == 90 and hashes == freeze["test_hashes"]
    assert Counter(track["difficulty"] for track in tracks) == dict.fromkeys(LEVELS, 30)
    assert not set(hashes) & set(suite["protected_geometry_hashes"])
    assert not set(hashes) & {r["track_hash"] for r in read_csv(training / "training-episodes.csv")}
    for entry in tracks:
        assert TrackSpec.load(manifest.parent / entry["path"]).geometry_hash == entry["geometry_hash"]

    records = read_csv(assessment / "test-episodes.csv")
    index = {(r["track_hash"], r["policy_id"]): r for r in records}
    assert len(index) == len(records) == 450
    rows = {policy: [index[h, policy] for h in hashes] for policy in POLICIES}
    for slot, entry in enumerate(tracks):
        paired = [rows[policy][slot] for policy in POLICIES]
        for row in paired:
            assert not row["error"] and row["phase"] == row["split"] == "test"
            assert row["backend"] == "cpu" and float(row["evaluation_speed_limit_m_s"]) == .35
            assert int(row["track_slot"]) == slot and row["difficulty"] == entry["difficulty"]
            assert set(row["features"].split(";")) == set(entry["features"])
        for field in ("time_limit_s", "route_length_m", "track_name"):
            assert len({row[field] for row in paired}) == 1

    generations = read_csv(training / "generations.csv")
    curriculum = read_csv(training / "curriculum.csv")
    assert [int(row["generation"]) for row in generations] == list(range(100))
    selected = freeze["new_neat_metadata"]["generation"]
    winner = max(generations, key=lambda r: (float(r["validation_completion_rate"]), float(r["validation_score"])))
    assert int(winner["generation"]) == selected == 94
    return rows, tracks, generations, curriculum, selected


def export(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=190)
    plt.close(fig)


def mean_field(rows, field):
    return mean(float(row[field]) for row in rows)


def summarize(rows):
    summary = []
    for policy, records in rows.items():
        for subset in ("all", *LEVELS, "zigzag"):
            sample = [r for r in records if subset in {"all", r["difficulty"]}
                      or subset == "zigzag" and "zigzag" in r["features"].split(";")]
            summary.append({"policy": policy, "subset": subset, "tracks": len(sample),
                            "completed": sum(map(completed, sample)),
                            "mean_progress_pct": 100 * mean_field(sample, "progress"),
                            "mean_rmse_mm": 1000 * mean_field(sample, "lateral_rmse_m"),
                            "mean_mae_mm": 1000 * mean_field(sample, "mean_absolute_lateral_error_m")})
    paired = []
    for policy in list(POLICIES)[1:]:
        pairs = list(zip(rows["neat-fresh"], rows[policy], strict=True))
        outcomes = Counter((completed(new), completed(reference)) for new, reference in pairs)
        common = [(new, reference) for new, reference in pairs if completed(new) and completed(reference)]
        a, b = zip(*common)
        paired.append({"comparator": policy, "both_finish": outcomes[True, True],
                       "g94_only": outcomes[True, False], "comparator_only": outcomes[False, True],
                       "neither": outcomes[False, False],
                       "g94_mean_time_s": mean_field(a, "completion_time_s"),
                       "comparator_mean_time_s": mean_field(b, "completion_time_s"),
                       "g94_faster": sum(float(n["completion_time_s"]) < float(r["completion_time_s"]) for n, r in common),
                       "g94_mean_rmse_mm": 1000 * mean_field(a, "lateral_rmse_m"),
                       "comparator_mean_rmse_mm": 1000 * mean_field(b, "lateral_rmse_m")})
    export("test-summary.csv", summary)
    export("paired-summary.csv", paired)
    return summary


def learning_plot(generations, curriculum, selected):
    x = np.arange(len(generations))
    completion = 100 * np.array([float(r["validation_completion_rate"]) for r in generations])
    color = POLICIES["neat-fresh"][1]
    fig = page(1, "Validation improved, with swings between champions",
               "Each generation’s training winner ran on the same 30 validation tracks.",
               "Shading shows the four-track batch’s procedural course difficulty. Drill stages are shown below.")
    upper, lower = fig.add_gridspec(2, 1, left=.105, right=.96, top=.72, bottom=.18, hspace=.24,
                                   height_ratios=(3.3, 1)).subplots(sharex=True)
    for start, stop, tint, label in ((0, 20, "#eaf1f6", "Beginner"), (20, 55, "#f4eee3", "Intermediate"),
                                     (55, 100, "#f0eaf6", "Advanced")):
        for ax in (upper, lower):
            ax.axvspan(start-.5, stop-.5, color=tint, zorder=0)
        upper.text((start+stop-1)/2, 110, label, ha="center", va="center", color=MUTED, fontsize=9)
    upper.plot(x, completion, lw=1.2, color="#536f8e", alpha=.78, label="Generation champion")
    upper.step(x, np.maximum.accumulate(completion), where="post", lw=2.2, color=color, label="Best completion so far")
    upper.scatter([selected], [completion[selected]], color=color, s=56, edgecolor="white", zorder=5)
    upper.annotate(f"Selected g{selected}", (selected, completion[selected]), xytext=(-9, -12),
                   textcoords="offset points", ha="right", va="top", color=color, fontsize=9)
    upper.set(ylabel="Validation completion", ylim=(25, 116), yticks=(40, 60, 80, 100))
    upper.yaxis.set_major_formatter(PercentFormatter(100))
    upper.legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    lower.step(x, [int(r["stage"]) for r in curriculum], where="post", color="#4a506c", lw=1.7)
    lower.set(ylabel="Drill stage", xlabel="Generation", xlim=(-.5, 99.5), ylim=(-.5, 4.7), yticks=range(5))
    for ax in (upper, lower):
        ax.grid(axis="y", color=GRID, lw=.8)
    save(fig, "01-learning")


def completion_plot(summary):
    fig = page(2, "NEAT g94 finished 88 of 90 fresh tracks",
               "Five frozen controllers on the same routes. Every controller finished all 30 beginner tracks.",
               "Zigzag is a track tag. The 40 zigzag routes overlap the difficulty groups.")
    axes = fig.add_gridspec(1, 4, left=.075, right=.97, top=.65, bottom=.18, wspace=.26).subplots(sharey=True)
    fields = {(row["policy"], row["subset"]): row for row in summary}
    legend = [Patch(facecolor=color, label=label) for label, color in POLICIES.values()]
    fig.legend(handles=legend, ncol=5, loc="upper left", bbox_to_anchor=(.04, .762), frameon=False, fontsize=8.6)
    for ax, subset, title in zip(axes, ("all", "intermediate", "advanced", "zigzag"),
                                 ("Overall", "Intermediate", "Advanced", "With zigzags")):
        samples = [fields[policy, subset] for policy in POLICIES]
        bars = ax.bar(range(5), [100*r["completed"]/r["tracks"] for r in samples], width=.74,
                      color=[color for _, color in POLICIES.values()])
        ax.bar_label(bars, [str(r["completed"]) for r in samples], padding=2, fontsize=8)
        ax.set(title=f"{title}  ·  {samples[0]['tracks']} tracks", xticks=[], ylim=(0, 113), yticks=range(0, 101, 25))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("Tracks completed")
    save(fig, "02-test-completion")


def paired_plot(rows):
    common = [(a, b) for a, b in zip(rows["neat-fresh"], rows["pid-supervised-tuned"], strict=True)
              if completed(a) and completed(b)]
    fig = page(3, "NEAT g94 was faster on every shared finish",
               f"NEAT g94 vs. tuned supervised PID · {len(common)} tracks both finished.",
               "Each point is one track. RMSE is the lateral tracking error over the complete episode.")
    axes = fig.add_gridspec(1, 2, left=.085, right=.96, top=.71, bottom=.25, wspace=.26).subplots()
    difficulty = list(zip(LEVELS, ("o", "s", "^"), ("#485e91", "#9b6218", "#007666")))
    for ax, field, title, units, scale in zip(axes, ("completion_time_s", "lateral_rmse_m"),
                                              ("Completion time", "Lateral RMSE"), ("s", "mm"), (1, 1000)):
        max_value = max(float(row[field])*scale for pair in common for row in pair) * 1.06
        ax.plot([0, max_value], [0, max_value], color=MUTED, ls="--", lw=1)
        for level, marker, color in difficulty:
            subset = [(a, b) for a, b in common if a["difficulty"] == level]
            ax.scatter([float(a[field])*scale for a, _ in subset], [float(b[field])*scale for _, b in subset],
                       color=color, marker=marker, s=37, alpha=.85, edgecolor=PAPER, linewidth=.5)
        limits = (-.025*max_value, max_value)
        ax.set(title=title, xlabel=f"NEAT g94 ({units})", ylabel=f"Tuned supervised PID ({units})",
               xlim=limits, ylim=limits, aspect="equal")
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], marker=m, color=c, ls="", label=d.title()) for d, m, c in difficulty]
    handles.append(Line2D([], [], color=MUTED, ls="--", label="Equal"))
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(.5, .095), frameon=False, fontsize=9)
    save(fig, "03-paired-speed-error")


def trace_plot(rows, tracks):
    fig = page(4, "Two examples of zigzag tracks in the final test",
               "Top shows chassis trajectories on the painted route. Bottom shows progress through simulated time.",
               "○ Start   × Off route   ■ Finished. Grey shows paint, including gaps and branches.", size=(12, 7.3))
    axes = fig.add_gridspec(2, 2, left=.085, right=.96, top=.73, bottom=.14, hspace=.47, wspace=.25).subplots()
    policies = ("neat-fresh", "neat-g109", "pid-supervised-tuned")
    styles = ("-", ":", "--")
    for column, slot in enumerate((31, 32)):
        entry = tracks[slot]
        track = TrackSpec.load(RUN / "suite" / entry["path"])
        xy, clock = axes[:, column]
        half = track.arena_size / 2
        xy.imshow(TrackMask(track).field, cmap=ListedColormap(["#a8afb5", PAPER]),
                  extent=(-half, half, -half, half), interpolation="nearest", vmin=0, vmax=255)
        points = [track.route.points]
        for policy, style in zip(policies, styles):
            row = rows[policy][slot]
            trace = RUN / "assessment/telemetry/test" / PureWindowsPath(row["telemetry"]).name
            data = read_csv(trace)
            t, x, y, progress = np.array([[float(r[k]) for r in data] for k in ("time_s", "x_m", "y_m", "progress")])
            assert len(t) == int(row["control_steps"]) and np.isclose(t[-1], float(row["elapsed_s"]))
            color = POLICIES[policy][1]
            marker = "s" if completed(row) else "x"
            xy.plot(x, y, color=color, ls=style, lw=1.7)
            xy.scatter(x[-1], y[-1], color=color, marker=marker, s=36, zorder=4)
            clock.plot(np.r_[0, t], np.r_[0, progress*100], color=color, ls=style, lw=1.8)
            clock.scatter(t[-1], progress[-1]*100, color=color, marker=marker, s=36, zorder=4)
            points.append(np.column_stack((x, y)))
        x, y = np.concatenate(points).T
        xy.scatter(*track.route.points[0], facecolor=PAPER, edgecolor=INK, s=26, zorder=5)
        xy.set(title=f"Track {slot:03}  ·  {track.difficulty.title()}", xlabel="World X (m)",
               xlim=(min(x)-.15, max(x)+.15), ylim=(min(y)-.15, max(y)+.15), aspect="equal")
        clock.set(xlabel="Simulated time (s)", xlim=(0, None), ylim=(0, 105), yticks=(0, 50, 100))
        clock.yaxis.set_major_formatter(PercentFormatter(100))
        clock.grid(color=GRID)
    axes[0, 0].set_ylabel("World Y (m)")
    axes[1, 0].set_ylabel("Route progress")
    handles = [Line2D([], [], ls=style, lw=2, color=POLICIES[p][1], label=POLICIES[p][0])
               for p, style in zip(policies, styles)]
    fig.legend(handles=handles, ncol=3, loc="upper left", bbox_to_anchor=(.04, .808), frameon=False)
    save(fig, "04-zigzag-traces")


def main():
    rows, tracks, generations, curriculum, selected = load_results()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "font.size": 10,
                         "axes.facecolor": PAPER, "axes.labelcolor": INK, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.edgecolor": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                         "svg.fonttype": "none"})
    summary = summarize(rows)
    learning_plot(generations, curriculum, selected)
    completion_plot(summary)
    paired_plot(rows)
    trace_plot(rows, tracks)
    print(f"Saved four figures and two summary tables to {OUT}")


if __name__ == "__main__":
    main()
