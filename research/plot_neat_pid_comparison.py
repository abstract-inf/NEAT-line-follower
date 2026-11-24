"""Build figures and tables from the two saved, geometry-matched test cohorts."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.tracks import TrackMask, TrackSpec

RUNS = ROOT / "runs"
NEAT = RUNS / "neat-frozen-new90-20261102"
PID = RUNS / "pid-gains-20261029"
ORIGINAL = RUNS / "pid-research-20261029"
NEW_SUITE = PID / "suite/suite.json"
OLD_SUITE = ROOT / "benchmarks/suite/suite.json"
OUT = ROOT / "research/artifacts/neat-pid-comparison"
LEVELS = ("beginner", "intermediate", "advanced")
POLICIES = ("neat-g109", "gain-000", "gain-160", "gain-350")
NAMES = ("NEAT g109", "Filtered PID", "Supervised PID · original", "Supervised PID · tuned")
COLORS = ("#7441a5", "#3b6795", "#996018", "#087e6c")
OUTCOMES = ("both", "pid_only", "neat_only", "neither")
PAPER, INK, MUTED, GRID = "#faf9f6", "#182637", "#4c5c6d", "#e1e3e3"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed(row):
    return row["completed"] == "True"


def index_cohort(manifest, records):
    tracks = read_json(manifest)["splits"]["test"]
    hashes = {track["geometry_hash"] for track in tracks}
    rows = {(row["track_hash"], row["policy_id"]): row for row in records}
    assert len(hashes) == 90 and Counter(t["difficulty"] for t in tracks) == dict.fromkeys(LEVELS, 30)
    assert set(rows) == {(h, p) for h in hashes for p in POLICIES} and len(rows) == len(records)
    for track in tracks:
        sample = [rows[track["geometry_hash"], policy] for policy in POLICIES]
        for row in sample:
            assert row["difficulty"] == track["difficulty"] and row["track_seed"] == str(track["seed"])
            assert set(row["features"].split(";")) == set(track["features"])
            assert row["backend"] == "cpu" and float(row["evaluation_speed_limit_m_s"]) == 0.35
            assert not row["error"] and row["split"] == "test" and row["completed"] in {"True", "False"}
        for key in ("route_length_m", "time_limit_s", "track_name"):
            assert len({row[key] for row in sample}) == 1, (track["path"], key)
    return tracks, rows


def load_cohorts():
    neat, pid, old = [read_json(run / "run.json") for run in (NEAT, PID, ORIGINAL)]
    freeze, old_freeze = [read_json(run / "selection-freeze.json") for run in (PID, ORIGINAL)]
    assert freeze["overall"] == "gain-350"
    assert read_json(PID / "best.json") == freeze["parameters"]["gain-350"]
    assert old_freeze["parameters"]["pid-112"] == freeze["parameters"]["gain-000"]
    assert old_freeze["parameters"]["pid-111"] == freeze["parameters"]["gain-160"]
    assert neat["model_metadata"]["generation"] == 109
    assert digest(RUNS / "28-10-2026-hard/best.pkl") == neat["model_sha256"] == old_freeze["neat_model_sha256"]
    assert digest(NEAT / "frozen-model.pkl") == neat["model_sha256"]
    assert digest(NEW_SUITE) == neat["suite_sha256"] == freeze["suite_sha256"]
    assert digest(OLD_SUITE) == old["suite_sha256"] == old_freeze["suite_sha256"]
    assert digest(PID / "test-episodes.csv") == neat["reference_pid_csv_sha256"]
    for key in ("robot", "episode"):
        assert neat[key] == pid[key] == old[key]
    assert len({run["environment"]["source_sha256"] for run in (neat, pid, old)}) == 1

    latest = read_csv(NEAT / "test-episodes.csv") + read_csv(PID / "test-episodes.csv")
    aliases = {"pid-112": "gain-000", "pid-111": "gain-160"}
    original = [{**r, "policy_id": aliases.get(r["policy_id"], r["policy_id"])}
                for r in read_csv(ORIGINAL / "test-episodes.csv")]
    original += [r for r in read_csv(PID / "legacy-test-episodes.csv") if r["policy_id"] == "gain-350"]
    cohorts = {"latest": index_cohort(NEW_SUITE, latest), "original": index_cohort(OLD_SUITE, original)}
    assert not {h for h, _ in cohorts["latest"][1]} & {h for h, _ in cohorts["original"][1]}
    return cohorts


def summarize(cohorts):
    summary, paired = [], []
    for cohort, (tracks, rows) in cohorts.items():
        for policy in POLICIES:
            for level in ("all", *LEVELS):
                sample = [row for (_, p), row in rows.items()
                          if p == policy and (level == "all" or row["difficulty"] == level)]
                n = sum(map(completed, sample))
                summary.append(dict(
                    cohort=cohort, policy=policy, difficulty=level, tracks=len(sample), completed=n,
                    completion_pct=100 * n / len(sample),
                    mean_progress_pct=100 * mean(float(r["progress"]) for r in sample),
                    mean_rmse_mm=1000 * mean(float(r["lateral_rmse_m"]) for r in sample),
                    mean_mae_mm=1000 * mean(float(r["mean_absolute_lateral_error_m"]) for r in sample),
                    off_route=sum(r["termination"] == "off_route" for r in sample),
                    stalled=sum(r["termination"] == "stalled" for r in sample),
                ))
        for track in tracks:
            h = track["geometry_hash"]
            n = rows[h, POLICIES[0]]
            for policy in POLICIES[1:]:
                p = rows[h, policy]
                outcome = {(True, True): "both", (False, True): "pid_only",
                           (True, False): "neat_only", (False, False): "neither"}[completed(n), completed(p)]
                paired.append(dict(
                    cohort=cohort, track_slot=Path(track["path"]).stem, track_hash=h,
                    track_name=n["track_name"], difficulty=n["difficulty"], features=n["features"],
                    pid_policy=policy, outcome=outcome, neat_termination=n["termination"],
                    pid_termination=p["termination"], neat_progress_pct=100 * float(n["progress"]),
                    pid_progress_pct=100 * float(p["progress"]), neat_time_s=n["completion_time_s"],
                    pid_time_s=p["completion_time_s"], neat_rmse_mm=1000 * float(n["lateral_rmse_m"]),
                    pid_rmse_mm=1000 * float(p["lateral_rmse_m"]),
                ))
    write_csv("summary.csv", summary)
    write_csv("paired-tracks.csv", paired)
    return summary, [r for r in paired if r["cohort"] == "latest"]


def page(number, title, subtitle, footer, size=(12, 5.1)):
    fig = plt.figure(figsize=size, facecolor=PAPER)
    fig.text(.045, .958, f"MUJOCO LINE FOLLOWING  /  MATCHED BENCHMARK  /  {number:02}",
             fontsize=9, color=MUTED, weight="bold", va="top")
    fig.text(.045, .905, title, fontsize=23, weight="bold", va="top")
    fig.text(.045, .817 if size[1] < 6 else .842, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(.045, .039, footer, fontsize=9, color=MUTED, va="bottom")
    return fig


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=190)
    plt.close(fig)


def policy_legend(fig):
    handles = [Patch(color=color, label=label) for color, label in zip(COLORS, NAMES)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.038, .77), ncol=4, frameon=False)


def completion_plot(summary):
    fig = page(1, "Completion drops sharply for NEAT beyond beginner",
               "All four frozen controllers ran once per track on each cohort.",
               "30 tracks per difficulty. Tuned PID on the original cohort is a secondary benchmark.")
    axes = fig.add_gridspec(1, 2, left=.09, right=.955, top=.66, bottom=.17, wspace=.32,
                            width_ratios=(1.48, 1)).subplots()
    policy_legend(fig)
    data = {(r["cohort"], r["policy"], r["difficulty"]): r for r in summary}
    groups = [[("latest", level) for level in LEVELS], [(cohort, "all") for cohort in ("original", "latest")]]
    labels = [["Beginner", "Intermediate", "Advanced"], ["Original 90", "Latest 90"]]
    for ax, group, names, title in zip(axes, groups, labels, ("Latest cohort by difficulty", "Overall by cohort")):
        x = np.arange(len(group))
        for i, (policy, color) in enumerate(zip(POLICIES, COLORS)):
            sample = [data[cohort, policy, level] for cohort, level in group]
            bars = ax.bar(x + (i-1.5)*.2, [r["completion_pct"] for r in sample], width=.17, color=color)
            ax.bar_label(bars, [str(r["completed"]) for r in sample], padding=3, fontsize=8)
        ax.set(title=title, xticks=x, xticklabels=names, ylim=(0, 113), yticks=range(0, 101, 25))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("Tracks completed")
    save(fig, "01-completion")


def pairing_plot(paired, rows):
    fig = page(2, "Tuned PID finishes 46 tracks that NEAT misses",
               "Latest cohort · 90 identical routes per controller. Each row on the left accounts for all 90 tracks.",
               "Progress is the greatest valid fraction of the ordered route reached before termination.")
    a, b = fig.add_gridspec(1, 2, left=.235, right=.96, top=.69, bottom=.24, wspace=.38,
                           width_ratios=(1.17, 1)).subplots()
    categories = ("Both finish", "Only PID", "Only NEAT", "Neither finishes")
    colors = ("#39485c", COLORS[-1], COLORS[0], "#cfd5da")
    for i, policy in enumerate(POLICIES[1:]):
        counts = Counter(r["outcome"] for r in paired if r["pid_policy"] == policy)
        left = 0
        for key, color in zip(OUTCOMES, colors):
            count = counts[key]
            if count:
                a.barh(i, count, left=left, height=.53, color=color)
                if count >= 5:
                    a.text(left + count / 2, i, str(count), ha="center", va="center",
                           color=INK if key == "neither" else "white", fontsize=10, weight="bold")
                else:
                    a.text(left + count / 2, i - .34, str(count), ha="center", va="bottom",
                           color=color, fontsize=9, weight="bold")
            left += count
    a.set(title="Track by track vs. NEAT", xlim=(0, 90), xticks=(0, 30, 60, 90),
          yticks=(0, 1, 2), yticklabels=NAMES[1:], ylim=(2.5, -.62), xlabel="Tracks")
    handles = [Patch(color=c, label=l) for c, l in zip(colors, categories)]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.5, .093), fontsize=9)
    for policy, name, color, style in zip(POLICIES, NAMES, COLORS, ("-", ":", "--", "-")):
        progress = np.sort([float(r["progress"]) for (_, p), r in rows.items() if p == policy])
        x = np.unique(np.r_[0., progress, 1.])
        y = len(progress) - np.searchsorted(progress, x, side="left")
        b.step(x * 100, y, where="pre", color=color, ls=style, lw=2, label=name)
    b.set(title="How far each controller gets", xlabel="Route progress reached", ylabel="Tracks reaching threshold",
          xlim=(0, 100), ylim=(0, 94), yticks=(0, 30, 60, 90))
    b.xaxis.set_major_formatter(PercentFormatter(100))
    b.grid(color=GRID)
    b.legend(loc="lower left", fontsize=8, frameon=False)
    save(fig, "02-paired-outcomes")


def common_success_plot(paired):
    common = [r for r in paired if r["pid_policy"] == "gain-350" and r["outcome"] == "both"]
    faster = sum(float(r["neat_time_s"]) < float(r["pid_time_s"]) for r in common)
    lower_error = sum(r["pid_rmse_mm"] < r["neat_rmse_mm"] for r in common)
    fig = page(3, "NEAT is faster when both finish; tuned PID tracks closer",
               f"Latest cohort · {len(common)} shared finishes, shown as one dot per track.",
               f"NEAT was faster on {faster}/{len(common)}. Tuned PID had lower RMSE on "
               f"{lower_error}/{len(common)}. Failed episodes are excluded from this figure.")
    axes = fig.add_gridspec(1, 2, left=.10, right=.95, top=.69, bottom=.255, wspace=.3).subplots()
    markers = ("o", "s", "^")
    for ax, metric, title, unit in zip(axes, ("time_s", "rmse_mm"), ("Completion time", "Lateral RMSE"), ("s", "mm")):
        upper = 1.07 * max(float(r[f"{p}_{metric}"]) for r in common for p in ("neat", "pid"))
        ax.plot((0, upper), (0, upper), color="#667482", lw=1, ls="--", label="Equal")
        for level, marker in zip(LEVELS, markers):
            sample = [r for r in common if r["difficulty"] == level]
            ax.scatter([float(r[f"neat_{metric}"]) for r in sample], [float(r[f"pid_{metric}"]) for r in sample],
                       marker=marker, s=43, color="#176787", alpha=.84, edgecolor=PAPER, linewidth=.5)
        limits = (-.025 * upper, upper)
        ax.set(title=title, xlabel=f"NEAT g109 ({unit})", ylabel=f"Tuned supervised PID ({unit})",
               xlim=limits, ylim=limits, aspect="equal")
        ax.set_axisbelow(True)
        ax.grid(color=GRID)
    handles = [Line2D([], [], color="#176787", marker=m, ls="none", label=level.title())
               for level, m in zip(LEVELS, markers)]
    handles += [Line2D([], [], color="#667482", ls="--", label="Equal")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(.5, .093), frameon=False)
    save(fig, "03-speed-and-error")


def trace_plot(tracks, rows):
    fig = page(4, "Three tracks show what the completion counts hide",
               "Latest cohort · one saved trace per difficulty. Top: chassis paths. Bottom: ordered-route progress.",
               "Markers: ○ start, × off route, ■ completed. Grey shows the painted line, including gaps and branches.",
               size=(12, 7.6))
    axes = fig.add_gridspec(2, 3, left=.085, right=.965, top=.729, bottom=.135,
                            wspace=.34, hspace=.52, height_ratios=(1.2, 1)).subplots()
    pid_traces = {r["track_hash"]: r for r in read_csv(PID / "test-trace-replays.csv") if r["policy_id"] == "gain-350"}
    for column, slot in enumerate((0, 31, 62)):
        route = tracks[slot]
        track = TrackSpec.load(NEW_SUITE.parent / route["path"])
        h = route["geometry_hash"]
        assert h == track.geometry_hash
        a, b = axes[:, column]
        mask = TrackMask(track)
        half = track.arena_size / 2
        a.imshow(mask.field, cmap=ListedColormap(["#a8afb5", PAPER]),
                 extent=(-half, half, -half, half), interpolation="nearest", zorder=0, vmin=0, vmax=255)
        points = [track.route.points]
        for policy, color, style in ((POLICIES[-1], COLORS[-1], "-"), (POLICIES[0], COLORS[0], "--")):
            row = rows[h, policy]
            trace = pid_traces[h] if policy == "gain-350" else row
            for key in ("completed", "termination", "progress", "elapsed_s", "lateral_rmse_m"):
                assert trace[key] == row[key], (slot, policy, key)
            trace_dir = PID / "telemetry/test-traces" if policy == "gain-350" else NEAT / "telemetry/test"
            path = trace_dir / f"{policy}-{slot:03}.csv"
            assert Path(trace["telemetry"]).name == path.name
            data = read_csv(path)
            t, x, y, progress = np.array([[float(r[k]) for r in data]
                                          for k in ("time_s", "x_m", "y_m", "progress")])
            assert len(t) == int(row["control_steps"]) and np.isclose(t[-1], float(row["elapsed_s"]))
            marker = "s" if completed(row) else "x"
            a.plot(x, y, color=color, ls=style, lw=1.6)
            a.scatter(x[-1], y[-1], color=color, marker=marker, s=31, zorder=4)
            b.plot(np.r_[0, t], np.r_[0, progress * 100], color=color, ls=style, lw=1.8)
            b.scatter(t[-1], progress[-1]*100, color=color, marker=marker, s=31, zorder=4)
            points.append(np.column_stack((x, y)))
        x, y = np.concatenate(points).T
        a.scatter(*track.route.points[0], facecolor=PAPER, edgecolor=INK, marker="o", s=23, zorder=4)
        a.set(title=f"{track.difficulty.title()} · track {slot:03}", xlim=(min(x)-.15, max(x)+.15),
              ylim=(min(y)-.15, max(y)+.15), xlabel="World X (m)", aspect="equal")
        b.set(xlabel="Simulated time (s)", xlim=(0, None), ylim=(0, 106), yticks=(0, 50, 100))
        b.yaxis.set_major_formatter(PercentFormatter(100))
        b.grid(color=GRID)
    axes[0, 0].set_ylabel("World Y (m)")
    axes[1, 0].set_ylabel("Route progress")
    handles = [Line2D([], [], color=COLORS[0], ls="--", lw=2, label=NAMES[0]),
               Line2D([], [], color=COLORS[-1], lw=2, label=NAMES[-1])]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.04, .805), ncol=2, frameon=False)
    save(fig, "04-track-traces")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.facecolor": PAPER, "axes.labelcolor": INK, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.edgecolor": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                         "svg.fonttype": "none"})
    cohorts = load_cohorts()
    summary, paired = summarize(cohorts)
    tracks, rows = cohorts["latest"]
    completion_plot(summary)
    pairing_plot(paired, rows)
    common_success_plot(paired)
    trace_plot(tracks, rows)
    print(f"Saved four figures and two data tables to {OUT}")


if __name__ == "__main__":
    main()
