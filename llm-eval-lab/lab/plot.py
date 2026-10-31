"""Plot a GSM8K vs GSM-Symbolic run. Writes PNGs into <run_dir>/plots/:

    difficulty.png      accuracy per model across GSM8K -> Symbolic -> P1 -> P2, with 95% CIs
    gap.png             accuracy drop vs GSM8K on the same original questions, with 95% CIs
    variant_spread.png  accuracy of each variant set vs the GSM8K original (one panel per model)
    pareto.png          accuracy vs latency on GSM-Symbolic main

Usage:
    python -m lab.plot results/runs/<run_id>
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .analyze import ORIGINALS, summarize

# Categorical slots 1-3 of the reference palette (validated all-pairs for colorblind safety).
# Aqua is below 3:1 contrast on the surface, so every series also gets a direct text label.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

DATASET_ORDER = [ORIGINALS, "gsm_symbolic_main", "gsm_symbolic_p1", "gsm_symbolic_p2"]
DATASET_LABELS = {
    ORIGINALS: "GSM8K",
    "gsm_symbolic_main": "Symbolic",
    "gsm_symbolic_p1": "Symbolic P1",
    "gsm_symbolic_p2": "Symbolic P2",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": TEXT_SECONDARY,
    "font.size": 10,
})


def colors(models: list[str]) -> dict[str, str]:
    """Fixed order: a model keeps its color as long as the config's model order is unchanged."""
    if len(models) > len(SERIES):
        raise ValueError(f"palette supports {len(SERIES)} models; facet or drop some")
    return dict(zip(models, SERIES))


def plot_difficulty(summary, color, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    end_labels = []
    for model, datasets in summary.items():
        names = [d for d in DATASET_ORDER if d in datasets]
        xs = [DATASET_ORDER.index(d) for d in names]
        accs = [datasets[d]["acc"] for d in names]
        ys = [a[0] for a in accs]
        ax.fill_between(xs, [a[1] for a in accs], [a[2] for a in accs], color=color[model],
                        alpha=0.12, linewidth=0)
        ax.plot(xs, ys, color=color[model], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=model)
        end_labels.append((ys[-1], xs[-1], f"{model}  {ys[-1]:.0%}"))
    # Direct end labels, nudged apart vertically so tied models don't overprint
    placed = []
    for y, x, text in sorted(end_labels):
        y_label = max(y, placed[-1] + 0.05) if placed else y
        placed.append(y_label)
        ax.annotate(text, (x, y), xytext=(x + 0.08, y_label), textcoords="data",
                    va="center", color=TEXT_SECONDARY)
    ax.set_xticks(range(len(DATASET_ORDER)), [DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_xlim(-0.2, len(DATASET_ORDER) - 1 + 0.9)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy as questions get harder", pad=12)
    ax.legend(loc="lower left", title="Band = 95% bootstrap CI", title_fontsize=9,
              alignment="left")
    fig.tight_layout()
    fig.savefig(out / "difficulty.png", dpi=160)
    plt.close(fig)


def plot_gap(summary, color, out: Path) -> None:
    variants = [d for d in DATASET_ORDER[1:]
                if any(datasets.get(d, {}).get("gap") for datasets in summary.values())]
    if not variants:
        print("gap.png skipped: no variant dataset shares questions with GSM8K in this run")
        return
    models = list(summary)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.9 * len(variants) * len(models) / 2))
    rows, labels = [], []
    for d in variants:
        for model in models:
            if gap := summary[model].get(d, {}).get("gap"):
                rows.append((model, gap))
                labels.append(f"{DATASET_LABELS[d]} · {model}")
    ys = list(range(len(rows)))[::-1]
    for y, (model, (g, lo, hi)) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=color[model], linewidth=2, solid_capstyle="round")
        ax.plot(g, y, "o", color=color[model], markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{g:+.1%}", (hi, y), xytext=(8, 0), textcoords="offset points",
                    va="center", color=TEXT_SECONDARY)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_yticks(ys, labels)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("GSM8K accuracy − variant accuracy (same original questions)")
    ax.set_title("Accuracy drop vs GSM8K (right of 0 = worse on variants)")
    fig.tight_layout()
    fig.savefig(out / "gap.png", dpi=160)
    plt.close(fig)


def plot_variant_spread(summary, color, out: Path, dataset: str = "gsm_symbolic_main") -> None:
    models = [m for m in summary if summary[m].get(dataset, {}).get("set_accs")]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 1, figsize=(8, 1.6 * len(models) + 0.8),
                             sharex=True, squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        set_accs = summary[model][dataset]["set_accs"]
        ax.hist(set_accs, bins=[i / 50 for i in range(51)], color=color[model],
                edgecolor=SURFACE, linewidth=2)
        if orig := summary[model].get(ORIGINALS):
            ax.axvline(orig["acc"][0], color=TEXT, linewidth=2, linestyle="--")
            ax.annotate(f"GSM8K {orig['acc'][0]:.0%}", (orig["acc"][0], 1), xycoords=("data", "axes fraction"),
                        xytext=(6, -4), textcoords="offset points", va="top", color=TEXT)
        ax.set_ylabel(model, rotation=0, ha="right", va="center", color=TEXT_SECONDARY)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlim(0, 1)
    axes[-1, 0].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[-1, 0].set_xlabel("Accuracy of one variant set (same questions, different names and numbers)")
    axes[0, 0].set_title(f"How much accuracy moves across variant sets ({DATASET_LABELS[dataset]})")
    fig.tight_layout()
    fig.savefig(out / "variant_spread.png", dpi=160)
    plt.close(fig)


def plot_pareto(summary, color, out: Path, dataset: str = "gsm_symbolic_main") -> None:
    points = [(m, s[dataset]["latency_s"], s[dataset]["acc"]) for m, s in summary.items() if dataset in s]
    if not points:
        return
    # A model is on the frontier if no other model is both faster and at least as accurate
    frontier = sorted(
        (p for p in points
         if not any(q[1] <= p[1] and q[2][0] >= p[2][0] and q != p for q in points)),
        key=lambda p: p[1],
    )
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot([p[1] for p in frontier], [p[2][0] for p in frontier], color=TEXT_SECONDARY,
            linewidth=1, linestyle="--", label="Pareto frontier")
    for model, latency, (acc, lo, hi) in points:
        ax.plot([latency, latency], [lo, hi], color=color[model], linewidth=2, solid_capstyle="round")
        ax.plot(latency, acc, "o", color=color[model], markersize=10, markeredgecolor=SURFACE,
                markeredgewidth=2)
        ax.annotate(model, (latency, acc), xytext=(10, 10), textcoords="offset points",
                    va="bottom", color=TEXT_SECONDARY)
    ax.set_xlim(0, max(p[1] for p in points) * 1.35)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Mean latency per question (s) — only comparable if the GPU wasn't throttling")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs speed ({DATASET_LABELS[dataset]})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    summary = summarize(args.run_dir)
    color = colors(list(summary))
    out = args.run_dir / "plots"
    out.mkdir(exist_ok=True)
    plot_difficulty(summary, color, out)
    plot_gap(summary, color, out)
    plot_variant_spread(summary, color, out)
    plot_pareto(summary, color, out)
    print(f"Plots: {out}")


if __name__ == "__main__":
    main()
