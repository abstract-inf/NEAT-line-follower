"""Draw the exact enabled topology and runtime activity of the saved g98 genome."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.controllers import load_model, read_model_artifact


def main():
    model = ROOT / "runs/neat-speed-100-20261104/assessment/frozen-best.pkl"
    artifact = read_model_artifact(model)
    genome, config = artifact["genome"], artifact["config"]
    controller, _ = load_model(model)
    outputs = config.genome_config.output_keys
    inputs = config.genome_config.input_keys
    hidden = [409, 423, 347, 211, 532]
    assert set(hidden) == set(genome.nodes)-set(outputs)
    executed = {n for n, *_ in controller.network.node_evals}
    effective = {1}
    for n, _, _, _, _, links in reversed(controller.network.node_evals):
        if n in effective:
            effective.update(source for source, _ in links)
    connections = [k for k, c in genome.connections.items() if c.enabled]
    used = {k for k in connections if k[0] in effective and k[1] in effective}
    assert len(inputs) == 21 and len(connections) == 30 and len(used) == 12
    positions = {key: (0, 8.5-i*.405) for i, key in enumerate(inputs)}
    positions.update({409: (4.0, 6.8), 423: (4.0, 4.9), 347: (4.0, 3.0),
                      211: (4.0, 1.1), 532: (4.0, -.8), 1: (8.0, 6.0), 0: (8.0, 1.5)})
    bg, ink, muted = "#f6f5f0", "#172d2d", "#526763"
    teal, grey, amber = "#147d70", "#bbc5c0", "#ad7c2b"
    fig, ax = plt.subplots(figsize=(15.5, 11))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set(xlim=(-3.0, 10.25), ylim=(-1.8, 9.65))
    ax.axis("off")
    fig.subplots_adjust(left=.025, right=.985, top=.84, bottom=.11)
    fig.text(.055, .958, "G98  /  EVOLVED NEURAL NETWORK", color=teal, fontsize=11, weight="bold")
    fig.text(.055, .91, "21 inputs · 5 stored hidden neurons · 2 output channels",
             fontsize=24, color=ink, weight="bold")
    fig.text(.055, .873, "Exact saved topology. Feedforward network with tanh activations and sparse skip connections.",
             fontsize=12, color=muted)
    ax.text(0, 9.33, "INPUTS", ha="center", color=muted, fontsize=12, weight="bold")
    ax.text(4, 9.33, "HIDDEN NEURONS", ha="center", color=muted, fontsize=12, weight="bold")
    ax.text(8, 9.33, "OUTPUTS", ha="center", color=muted, fontsize=12, weight="bold")
    for source, target in sorted(connections, key=lambda k: k in used):
        start, end = positions[source], positions[target]
        direct = source in inputs and target in outputs
        curve = -.12 if direct and target == 1 else .10 if direct else 0
        active = (source, target) in used
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                     connectionstyle=f"arc3,rad={curve}", shrinkA=9, shrinkB=17,
                                     color=teal if active else grey,
                                     lw=1.45 if active else .9, alpha=.78 if active else .6,
                                     linestyle="-" if active else "--", zorder=1))
    labels = [f"Sensor {i:02d}" for i in range(15)] + [
        "Current centroid", "Last visible centroid", "Edge memory",
        "Line-loss duration", "Previous forward", "Previous steering"]
    for i, (key, name) in enumerate(zip(inputs, labels), 1):
        x, y = positions[key]
        color = teal if key in effective else muted
        ax.add_patch(Circle((x, y), .115, facecolor=bg, edgecolor=color, lw=1.4, zorder=3))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7.5, color=color, zorder=4)
        ax.text(x-.23, y, name, ha="right", va="center", fontsize=10, color=ink)
    for key in hidden:
        x, y = positions[key]
        color = teal if key in effective else amber if key in executed else muted
        fill = "#e2efea" if key in effective else "#f1e8d7" if key in executed else "#e8ece8"
        ax.add_patch(Circle((x, y), .29, facecolor=fill, edgecolor=color, lw=1.8, zorder=3))
        ax.text(x, y, f"H{key}", ha="center", va="center", fontsize=10, weight="bold", color=color, zorder=4)
        status = "tanh" if key in effective else "computed · unused" if key in executed else "not evaluated"
        ax.text(x, y-.46, status, ha="center", va="top", fontsize=9, color=color, zorder=4,
                bbox={"facecolor": bg, "edgecolor": "none", "pad": 1.5})
    for key, name, status in ((1, "Steering", "Computed tanh output"),
                              (0, "Drive", "Not evaluated · stays at 0")):
        x, y = positions[key]
        color = teal if key == 1 else muted
        ax.add_patch(Circle((x, y), .34, facecolor="#d8eee4" if key == 1 else "#e8ece8",
                            edgecolor=color, lw=2.2, zorder=3))
        ax.text(x, y, "s" if key == 1 else "d", ha="center", va="center", fontsize=18,
                weight="bold", color=color, zorder=4)
        ax.text(x+.50, y+.04, name, fontsize=13, weight="bold", color=color, va="center")
        ax.text(x, y-.53, status, ha="center", va="top", fontsize=10, color=color,
                bbox={"facecolor": bg, "edgecolor": "none", "pad": 2})
    handles = [Line2D([], [], color=teal, lw=2, label="12 connections contributing to steering"),
               Line2D([], [], color=grey, lw=1.5, ls="--", label="18 other enabled stored connections")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.05, .055), ncol=2,
               frameon=False, fontsize=11, labelcolor=ink)
    fig.text(.055, .033, "Neuron IDs come from the saved genome. The runtime loader evaluates three hidden neurons; two influence steering.",
             color=muted, fontsize=10)
    out = ROOT / "research/artifacts/neat-g98-architecture"
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(out / f"g98-architecture.{extension}", dpi=180, facecolor=bg,
                    metadata={"Date": None} if extension == "svg" else None)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
