"""Generate publication figures and a results note from the frozen assessment."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from line_follower.controllers import load_model
from run_neat_speed_study import BASELINES, CAPS, OUTPUT, SEEDS, digest, read_json

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "research/artifacts/neat-five-seed-paper-figures"
STAGES = ("low", "ramp")
COLORS = {"low": "#0072B2", "ramp": "#D55E00", "pid": "#3C4651"}
LABELS = {"low": "Low-speed NEAT", "ramp": "Ramp NEAT"}
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73", "#8D579E", "#5C6223")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_assessment(study):
    assessment = study / "assessment"
    directory = assessment / "results"
    status = read_json(directory / "status.json")
    freeze = read_json(assessment / "frozen/selection-freeze.json")
    manifest = study / "prepared/suite/suite.json"
    if (not status["complete"] or digest(manifest) != freeze["suite_sha256"]
            or any(digest(directory / name) != sha for name, sha in status["files"].items())):
        raise ValueError("Assessment fingerprints disagree")
    for record in freeze["policies"].values():
        if digest(assessment / "frozen" / record["file"]) != record["sha256"]:
            raise ValueError("Frozen controller fingerprint disagrees")
    entries = read_json(manifest)["splits"]["test"]
    tracks = [entry["geometry_hash"] for entry in entries]
    groups = {(cap, policy): {} for cap in CAPS for policy in freeze["policies"]}
    for row in csv_rows(directory / "test-episodes.csv"):
        cap, policy, track = float(row["evaluation_speed_limit_m_s"]), row["policy_id"], row["track_hash"]
        group = groups[cap, policy]
        if row["error"] or track in group:
            raise ValueError("Duplicate or technically invalid final-test row")
        group[track] = {"finished": row["completed"] == "True",
                        "time": float(row["completion_time_s"]) if row["completion_time_s"] else None,
                        "rmse": 1000 * float(row["geometry_lateral_rmse_m"]),
                        "jitter": float(row["command_jitter"]),
                        "progress": float(row["progress"])}
    if any(set(group) != set(tracks) for group in groups.values()):
        raise ValueError("Every controller must share the exact test tracks")
    if status["episodes"] != len(groups) * len(tracks):
        raise ValueError("Unexpected test episode count")
    return groups, freeze, tracks


def seed_policy(seed, stage):
    return f"seed-{seed:02d}-{stage}"


def completed(group):
    return sum(row["finished"] for row in group.values())


def shared_times(first, second):
    return [(a["time"], second[track]["time"]) for track, a in first.items()
            if a["finished"] and second[track]["finished"]]


def pair_similarity(first, second):
    agreement = sum(row["finished"] == second[key]["finished"] for key, row in first.items())
    times = shared_times(first, second)
    gap = float(np.median([200 * abs(x-y)/(x+y) for x, y in times])) if times else np.nan
    return agreement, gap, len(times)


def stage_legend(fig, with_pid=False):
    handles = [Line2D([], [], color=COLORS[stage], marker=marker, linestyle="",
                      label=LABELS[stage]) for stage, marker in zip(STAGES, ("o", "D"), strict=True)]
    if with_pid:
        handles.append(Line2D([], [], color=COLORS["pid"], linestyle="--", label="Frozen PID at this cap"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)


def plot_completion(groups, count):
    fig, axes = plt.subplots(1, len(CAPS), figsize=(8.0, 3.65), sharey=True, layout="constrained")
    for axis, cap in zip(axes, CAPS, strict=True):
        for seed in SEEDS:
            values = [completed(groups[cap, seed_policy(seed, stage)]) for stage in STAGES]
            axis.plot([seed-.14, seed+.14], values, color="#ABB5C0", linewidth=1, zorder=2)
            for stage, offset, value, marker in zip(STAGES, (-.14, .14), values, ("o", "D"), strict=True):
                axis.scatter(seed+offset, value, s=43, color=COLORS[stage], marker=marker, zorder=3)
        axis.axhline(completed(groups[cap, BASELINES[cap]]), linestyle="--", color=COLORS["pid"], linewidth=1)
        axis.set(title=f"{cap:.2f} m/s cap", xlabel="Evolution seed", xticks=SEEDS, xlim=(.6, 5.4),
                 ylim=(0, count+5), yticks=np.linspace(0, count, 7))
        axis.grid(axis="y", color="#DEE4EB", linewidth=.6)
    axes[0].set_ylabel(f"Held-out tracks finished (of {count})")
    fig.suptitle("Held-out completion across five independent seeds")
    stage_legend(fig, with_pid=True)
    return fig


def plot_similarity(groups, count):
    fig, axes = plt.subplots(2, len(CAPS), figsize=(7.7, 6.7), layout="constrained")
    matrices = {}
    for column, cap in enumerate(CAPS):
        agreement = np.empty((len(SEEDS), len(SEEDS)), dtype=int)
        gap = np.full_like(agreement, np.nan, dtype=float)
        matches = np.zeros_like(agreement)
        for i, a in enumerate(SEEDS):
            first = groups[cap, seed_policy(a, "ramp")]
            for j, b in enumerate(SEEDS):
                second = groups[cap, seed_policy(b, "ramp")]
                agreement[i, j], gap[i, j], matches[i, j] = pair_similarity(first, second)
        matrices[cap] = agreement, gap, matches
    agreement_floor = min(count-5, min(matrix[0].min() for matrix in matrices.values()) // 5 * 5)
    gap_ceiling = max(1, max(float(np.nanmax(matrix[1])) for matrix in matrices.values()))
    for column, cap in enumerate(CAPS):
        agreement, gap, matches = matrices[cap]
        for row, (matrix, cmap, color_range, annotations) in enumerate((
            (agreement, "Blues", (agreement_floor, count), [[str(value) for value in line] for line in agreement]),
            (gap, "Oranges", (0, gap_ceiling),
             [[f"{value:.1f}%\n(n={n})" if n else "no shared\nfinishes"
               for value, n in zip(line, sizes, strict=True)] for line, sizes in zip(gap, matches, strict=True)]),
        )):
            axis = axes[row, column]
            image = axis.imshow(matrix, cmap=cmap, vmin=color_range[0], vmax=color_range[1])
            for i, line in enumerate(annotations):
                for j, label in enumerate(line):
                    rgba = image.cmap(image.norm(matrix[i, j]))
                    brightness = sum(a*b for a, b in zip(rgba[:3], (.299, .587, .114), strict=True))
                    axis.text(j, i, label, ha="center", va="center", fontsize=8,
                              color="white" if brightness < .47 else "#16212C")
            axis.set(xticks=range(len(SEEDS)), yticks=range(len(SEEDS)),
                     xticklabels=SEEDS, yticklabels=SEEDS, xlabel="Ramp seed",
                     ylabel="Ramp seed" if column == 0 else "",
                     title=f"{cap:.2f} m/s | " + ("matching outcomes" if row == 0 else "median paired time gap"))
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.suptitle("Pairwise similarity of the five ramp policies")
    return fig


def plot_speed_tracking(groups):
    fig, axes = plt.subplots(2, len(CAPS), figsize=(8.2, 6.1), layout="constrained")
    axes[0, 1].sharey(axes[0, 0])
    for column, cap in enumerate(CAPS):
        reference = groups[cap, BASELINES[cap]]
        top, bottom = axes[:, column]
        top.axhline(0, linestyle="--", linewidth=1, color=COLORS["pid"])
        for stage, offset, marker in zip(STAGES, (-.12, .12), ("o", "D"), strict=True):
            for seed in SEEDS:
                group = groups[cap, seed_policy(seed, stage)]
                times = shared_times(group, reference)
                if times:
                    difference = 100 * (mean(a for a, _ in times)/mean(b for _, b in times) - 1)
                    top.scatter(seed+offset, difference, color=COLORS[stage], marker=marker, s=38, zorder=3)
                bottom.scatter(mean(row["jitter"] for row in group.values()),
                               mean(row["rmse"] for row in group.values()),
                               color=COLORS[stage], marker=marker, s=38, alpha=.85)
        top.set(title=f"{cap:.2f} m/s cap", xlabel="Evolution seed", xticks=SEEDS, xlim=(.6, 5.4))
        bottom.scatter(mean(row["jitter"] for row in reference.values()),
                       mean(row["rmse"] for row in reference.values()),
                       marker="*", s=100, color=COLORS["pid"], zorder=3)
        bottom.set(xlabel="Mean command jitter", ylabel="Geometric RMSE (mm)" if column == 0 else "")
        bottom.xaxis.set_major_locator(MaxNLocator(4))
        for axis in (top, bottom):
            axis.grid(axis="y", color="#DEE4EB", linewidth=.6)
    axes[0, 0].set_ylabel("Mean finish time vs. PID (%)\non shared finishes")
    handles = [Line2D([], [], color=COLORS[stage], marker=marker, linestyle="", label=LABELS[stage])
               for stage, marker in zip(STAGES, ("o", "D"), strict=True)]
    handles.append(Line2D([], [], color=COLORS["pid"], marker="*", linestyle="--", label="Frozen PID"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Completion time, tracking error, and wheel-command jitter")
    return fig


def plot_learning(study):
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.65), sharey=True, layout="constrained")
    for axis, stage in zip(axes, STAGES, strict=True):
        for seed, color in zip(SEEDS, SEED_COLORS, strict=True):
            rows = csv_rows(study / f"seed-{seed:02d}/{stage}/generations.csv")
            generations = [int(row["generation"]) for row in rows]
            scores = [100 * float(row["validation_completion_rate"]) for row in rows]
            axis.plot(generations, scores, linewidth=1, alpha=.86, color=color, label=f"Seed {seed}")
        caps = [float(row["wheel_speed_cap_m_s"]) for row in rows]
        changes = [0] + [i for i in range(1, len(caps)) if caps[i] != caps[i-1]] + [len(caps)]
        for i, (start, end) in enumerate(zip(changes[:-1], changes[1:], strict=True)):
            if i % 2 == 0:
                axis.axvspan(start, end, color="#ECEFF3", zorder=0)
        axis.set(title="Low-speed stage" if stage == "low" else "Ramp stage",
                 xlabel="Generation (zero-based)", xlim=(0, len(caps)), ylim=(0, 104))
        axis.grid(axis="y", color="#DEE4EB", linewidth=.55)
        speed_axis = axis.secondary_xaxis("top")
        speed_axis.set_xticks([(a+b)/2 for a, b in zip(changes[:-1], changes[1:], strict=True)],
                              [f"{caps[a]:.2f}" for a in changes[:-1]])
        speed_axis.tick_params(length=0, labelsize=7)
        speed_axis.set_xlabel("Wheel-speed cap (m/s)")
    axes[0].set_ylabel("Validation completion (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Validation learning curves by stage")
    return fig


def plot_ramp_fitness(study):
    logs = [csv_rows(study / f"seed-{seed:02d}/ramp/generations.csv") for seed in SEEDS]
    phases = [(int(row["generation"]), float(row["wheel_speed_cap_m_s"])) for row in logs[0]]
    if any([(int(row["generation"]), float(row["wheel_speed_cap_m_s"])) for row in log] != phases
           for log in logs[1:]):
        raise ValueError("Ramp generations and speed caps differ across seeds")
    generations, caps = map(np.asarray, zip(*phases, strict=True))
    boundaries = [0, *(np.flatnonzero(np.diff(caps)) + 1), len(caps)]
    segments = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, layout="constrained")
    panels = (("training_mean", "Population mean\ntraining fitness", COLORS["low"]),
              ("validation_score", "Champion\nvalidation fitness", COLORS["ramp"]))
    for axis, (column, label, color) in zip(axes, panels, strict=True):
        scores = np.array([[float(row[column]) for row in log] for log in logs])
        avg, sd = scores.mean(axis=0), scores.std(axis=0, ddof=1)
        for start, end in segments:
            segment = slice(start, end)
            axis.plot(generations[segment], avg[segment], color=color, linewidth=1.4)
            axis.fill_between(generations[segment], (avg-sd)[segment], (avg+sd)[segment],
                              color=color, alpha=.17, linewidth=0)
            if start:
                axis.axvline(generations[start]-.5, color="#A1ABB4", linestyle=":", linewidth=.8)
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#DEE4EB", linewidth=.6)
    speed_axis = axes[0].secondary_xaxis("top")
    speed_axis.set_xticks([(generations[a]+generations[b-1])/2 for a, b in segments],
                          [f"{caps[a]:.2f}" for a, _ in segments])
    speed_axis.tick_params(length=0, labelsize=8)
    speed_axis.set_xlabel("Wheel-speed cap (m/s)")
    axes[-1].set(xlabel="Ramp generation (zero-based)", xlim=(generations[0]-.5, generations[-1]+.5))
    fig.suptitle("Ramp fitness across five seeds")
    fig.legend(handles=[Line2D([], [], color="#3C4651", label="Seed mean"),
                        Line2D([], [], color="#3C4651", linewidth=8, alpha=.17, label="±1 sample SD")],
               loc="outside lower center", ncol=2, frameon=False)
    return fig


def expressed_paths(network):
    incoming = {node: [origin for origin, weight in links if weight and response]
                for node, _, _, _, response, links in network.node_evals}
    paths = []
    for output in network.output_nodes:
        pending, visited, edges = [output], set(), set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            for origin in incoming.get(node, ()):
                edges.add((origin, node))
                pending.append(origin)
        paths.append((visited, edges))
    return paths


def plot_architecture(study, freeze):
    input_names = [f"Sensor {i+1}" for i in range(15)] + ["Line center", "Last center", "Edge memory",
                                                           "Line-loss age", "Previous drive", "Previous steer"]
    profiles, labels, complexity = [], [], []
    for seed in SEEDS:
        for stage in STAGES:
            saved = freeze["policies"][seed_policy(seed, stage)]
            controller, _ = load_model(study / "assessment/frozen" / saved["file"])
            network = controller.network
            (drive, drive_edges), (steer, steer_edges) = expressed_paths(network)
            profiles.append([int(node in drive) + 2*int(node in steer) for node in network.input_nodes])
            hidden = (drive | steer) - set(network.input_nodes) - set(network.output_nodes)
            complexity.append((len(hidden), len(drive_edges | steer_edges)))
            labels.append(f"S{seed} {stage} · g{saved['generation']}")
    fig, (axis, counts) = plt.subplots(1, 2, figsize=(9.05, 5.2), sharey=True,
                                    gridspec_kw={"width_ratios": (8, 1.7)}, layout="constrained")
    colors = ("#E8EBEF", "#0072B2", "#D55E00", "#7B559C")
    axis.imshow(profiles, cmap=ListedColormap(colors), vmin=-.5, vmax=3.5, aspect="auto")
    axis.set(xticks=range(len(input_names)), yticks=range(len(profiles)), yticklabels=labels)
    axis.set_xticklabels(input_names, rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
    axis.tick_params(length=0)
    for y in (1.5, 3.5, 5.5, 7.5):
        axis.axhline(y, color="white", linewidth=1.5)
    counts.set(xlim=(0, 2), xticks=(.5, 1.5))
    counts.set_xticklabels(["Hidden\nnodes", "Links"], fontsize=8)
    counts.tick_params(left=False, labelleft=False, length=0)
    counts.xaxis.tick_top()
    counts.spines[:].set_visible(False)
    for y, (nodes, edges) in enumerate(complexity):
        counts.text(.5, y, str(nodes), ha="center", va="center")
        counts.text(1.5, y, str(edges), ha="center", va="center")
    keys = ["Unused", "Drive", "Steering", "Both outputs"]
    fig.legend(handles=[Line2D([], [], linestyle="", marker="s", color=color, label=key)
                        for color, key in zip(colors, keys, strict=True)],
               loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Input paths in the selected networks before and after the ramp")
    return fig, complexity, profiles


def write_summary(output, groups, freeze, count, study, complexity, profiles):
    lines = ["# NEAT five-seed held-out assessment", "",
             f"The assessment used the same {count} held-out tracks for all {len(freeze['policies'])} frozen policies at each speed cap.",
             "The low and ramp models are paired by evolutionary seed. Means and sample SDs describe variation across five lineages on this shared suite.", "",
             "## Held-out completion", "",
             "| Wheel-speed cap | Stage | Seeds 1–5, each out of 90 | Mean | Sample SD | Frozen PID |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for cap in CAPS:
        for stage in STAGES:
            counts = [completed(groups[cap, seed_policy(seed, stage)]) for seed in SEEDS]
            lines.append(f"| {cap:.2f} m/s | {LABELS[stage]} | {', '.join(map(str, counts))} | "
                         f"{mean(counts)/count:.1%} | {100 * stdev(counts)/count:.1f} pp | "
                         f"{completed(groups[cap, BASELINES[cap]])}/{count} |")
    lines.extend(["", "### Ramp performance by seed", "",
                  "| Cap | Seed | Finished | Shared PID finishes | NEAT / PID mean time on shared finishes | Geometric RMSE | Command jitter |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for cap in CAPS:
        reference = groups[cap, BASELINES[cap]]
        for seed in SEEDS:
            group = groups[cap, seed_policy(seed, "ramp")]
            times = shared_times(group, reference)
            timing = f"{mean(a for a, _ in times):.2f} / {mean(b for _, b in times):.2f} s" if times else "—"
            lines.append(f"| {cap:.2f} m/s | {seed} | {completed(group)} | {len(times)} | {timing} | "
                         f"{mean(row['rmse'] for row in group.values()):.1f} mm | "
                         f"{mean(row['jitter'] for row in group.values()):.4f} |")
    cap = CAPS[-1]
    high = [completed(groups[cap, seed_policy(seed, "ramp")]) for seed in SEEDS]
    leave_one = [mean(high[:i] + high[i+1:])/count for i in range(len(high))]
    pairs = [pair_similarity(groups[cap, seed_policy(a, "ramp")], groups[cap, seed_policy(b, "ramp")])
             for a in SEEDS for b in SEEDS if a < b]
    agreement, gaps, _ = zip(*pairs, strict=True)
    low_edges, ramp_edges = ([edges for _, edges in complexity[index::2]] for index in (0, 1))
    ramp_drive_count = sum(any(value & 1 for value in profile) for profile in profiles[1::2])
    lines.extend(["", "## Interpretation", "",
                  f"At {cap:.2f} m/s, the ramp winners ranged from {min(high)}/{count} to {max(high)}/{count} completions. "
                  f"Leaving out one seed at a time gave four-seed mean rates from {min(leave_one):.1%} to {max(leave_one):.1%}. "
                  "This describes seed sensitivity on the shared test cohort.",
                  "",
                  f"At that cap, pairs of ramp winners agreed on {min(agreement)}–{max(agreement)} of {count} track outcomes. Their median "
                  f"absolute time gaps on shared finishes ranged from {min(gaps):.1f}% to {max(gaps):.1f}%.",
                  "",
                  f"The low-speed winners had {min(low_edges)}–{max(low_edges)} reachable links, and the ramp winners had "
                  f"{min(ramp_edges)}–{max(ramp_edges)}. {ramp_drive_count} of the five ramp networks had a structural path from an input to the drive output.",
                  "",
                  "The PID lines use the gain files frozen in this assessment protocol. At 1.50 m/s this is the earlier high-speed tune from 15 November. "
                  "The later 1.00 and 1.50 m/s PID follow-up used its own held-out cohort and has separate results.",
                  "", "## Figure captions", "",
                  "1. **[Held-out completion](01-held-out-completion.svg).** A line links the two winners from the same seed. The dashed line gives the cap-specific frozen PID reference.",
                  "2. **[Ramp policy similarity](02-ramp-policy-similarity.svg).** The top row counts tracks with matching finish or failure outcomes. The bottom row gives the median absolute time difference divided by the two times' mean on shared finishes. Each cell lists its shared denominator. Color scales are shared across caps within each row.",
                  "3. **[Speed and tracking](03-speed-and-tracking.svg).** Time differences use each policy's matched finishes with PID. Negative values indicate shorter times. Geometric RMSE and normalized command jitter average all observed episodes.",
                  "4. **[Validation learning curves](04-validation-learning-curves.svg).** Every line shows one seed's generation champions. Speed labels apply to the shaded or white intervals beneath them.",
                  "5. **[Network input paths](05-network-input-paths.svg).** Color marks structural paths through nonzero links in the expressed feedforward network. The two right columns count reachable hidden nodes and links. The architecture describes the network's possible dependencies.",
                  "6. **[Ramp fitness across seeds](06-ramp-fitness-across-seeds.svg).** The upper panel averages population mean training fitness and the lower panel averages champion validation fitness. Shading shows one sample SD across the five seeds. Each cap uses its own line segment.",
                  "", "Figures are available as SVG and 300 dpi PNG.", "",
                  f"Raw data directory: `{study / 'assessment/results'}`.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    groups, freeze, tracks = read_assessment(args.study)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "svg.fonttype": "none",
                         "text.color": "#16212C", "axes.labelcolor": "#16212C"})
    architecture, complexity, profiles = plot_architecture(args.study, freeze)
    figures = (
        ("01-held-out-completion", plot_completion(groups, len(tracks))),
        ("02-ramp-policy-similarity", plot_similarity(groups, len(tracks))),
        ("03-speed-and-tracking", plot_speed_tracking(groups)),
        ("04-validation-learning-curves", plot_learning(args.study)),
        ("05-network-input-paths", architecture),
        ("06-ramp-fitness-across-seeds", plot_ramp_fitness(args.study)),
    )
    for name, fig in figures:
        for extension in ("svg", "png"):
            fig.savefig(args.output / f"{name}.{extension}", dpi=300)
        plt.close(fig)
    write_summary(args.output, groups, freeze, len(tracks), args.study, complexity, profiles)
    print(f"Saved {len(figures)} figures and the results note in {args.output}")


if __name__ == "__main__":
    main()
