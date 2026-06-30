"""
Generates figures from sweep_results.json:
  Fig 1: Heatmap — final analogical accuracy averaged over seeds, for each
          (architecture, LR) pair. Shows Abstractor's wider emergence window.
  Fig 2: Learning curves — analogical accuracy vs training step for all runs,
          coloured by architecture and LR. Shows when emergence happens.
  Fig 3: Summary table — fraction of (seed, LR) configs where analogy
          "emerged" (accuracy > 0.5) per architecture.
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "sweep_results.json")
FIGS_DIR     = os.path.join(os.path.dirname(__file__), "results", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

EMERGENCE_THRESHOLD = 0.5   # analogical accuracy above this = "emerged"

def load():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def final_acc(run, key="test_analogical"):
    return run["final"].get(key, 0.0)


# ── Figure 1: Heatmap ────────────────────────────────────────────────────────

def plot_heatmap(data):
    archs = sorted(set(r["arch"] for r in data))
    lrs   = sorted(set(r["lr"]   for r in data))

    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4),
                             sharey=True)
    if len(archs) == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        runs  = [r for r in data if r["arch"] == arch]
        # mean over seeds
        mat   = np.zeros((len(lrs), 1))
        for i, lr in enumerate(lrs):
            vals = [final_acc(r) for r in runs if r["lr"] == lr]
            mat[i, 0] = np.mean(vals)

        im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto",
                       cmap="RdYlGn", origin="upper")
        ax.set_title(arch.capitalize(), fontsize=13, fontweight="bold")
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lrs])
        ax.set_ylabel("Learning Rate")
        ax.set_xticks([])
        for i, lr in enumerate(lrs):
            vals = [final_acc(r) for r in runs if r["lr"] == lr]
            ax.text(0, i, f"{np.mean(vals):.2f}",
                    ha="center", va="center", fontsize=11,
                    color="black" if np.mean(vals) < 0.8 else "white")

    plt.colorbar(im, ax=axes[-1], label="Analogical Accuracy")
    plt.suptitle("Final Analogical Accuracy (mean over 3 seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "heatmap_analogy.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGS_DIR, "heatmap_analogy.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: heatmap_analogy")


# ── Figure 2: Learning curves ────────────────────────────────────────────────

def plot_learning_curves(data):
    archs      = sorted(set(r["arch"] for r in data))
    lrs        = sorted(set(r["lr"]   for r in data))
    lr_colors  = plt.cm.viridis(np.linspace(0, 1, len(lrs)))
    lr_color_map = dict(zip(lrs, lr_colors))

    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 4), sharey=True)
    if len(archs) == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        for lr in lrs:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            all_steps = sorted(set(h["step"] for r in runs for h in r["history"]))
            # mean curve over seeds
            mean_acc = []
            for step in all_steps:
                vals = []
                for r in runs:
                    pt = next((h for h in r["history"] if h["step"] == step), None)
                    if pt:
                        vals.append(pt.get("test_analogical", 0.0))
                mean_acc.append(np.mean(vals) if vals else np.nan)

            ax.plot(all_steps, mean_acc,
                    color=lr_color_map[lr],
                    label=f"lr={lr:.0e}",
                    linewidth=2, alpha=0.85)
            # shade individual seeds
            for r in runs:
                steps = [h["step"] for h in r["history"]]
                accs  = [h.get("test_analogical", 0.0) for h in r["history"]]
                ax.plot(steps, accs,
                        color=lr_color_map[lr],
                        linewidth=0.5, alpha=0.3)

        ax.axhline(EMERGENCE_THRESHOLD, color="gray", linestyle="--",
                   linewidth=1, label=f"threshold={EMERGENCE_THRESHOLD}")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Analogical Accuracy")
        ax.set_title(arch.capitalize(), fontsize=13, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Analogical Accuracy vs Training Steps", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "learning_curves.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGS_DIR, "learning_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: learning_curves")


# ── Figure 3: Emergence fraction bar chart ───────────────────────────────────

def plot_emergence_fraction(data):
    archs = sorted(set(r["arch"] for r in data))
    lrs   = sorted(set(r["lr"]   for r in data))

    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(len(lrs))
    width = 0.35
    colors = {"vanilla": "#4C72B0", "abstractor": "#DD8452"}

    for i, arch in enumerate(archs):
        fracs = []
        for lr in lrs:
            runs  = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            frac  = sum(1 for r in runs if final_acc(r) >= EMERGENCE_THRESHOLD) / max(1, len(runs))
            fracs.append(frac)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, fracs, width,
                      label=arch.capitalize(), color=colors.get(arch, None), alpha=0.85)
        for bar, frac in zip(bars, fracs):
            if frac > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{frac:.0%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel(f"Fraction of Seeds with Analogy Emerged (>{EMERGENCE_THRESHOLD:.0%})")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.set_title("Emergence Rate by Architecture and Learning Rate", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "emergence_fraction.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGS_DIR, "emergence_fraction.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: emergence_fraction")


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(data):
    archs = sorted(set(r["arch"] for r in data))
    lrs   = sorted(set(r["lr"]   for r in data))

    print("\n=== Final Analogical Accuracy (mean ± std over seeds) ===")
    header = "LR         " + "  ".join(f"{a:>12}" for a in archs)
    print(header)
    print("-" * len(header))
    for lr in lrs:
        row = f"{lr:.0e}     "
        for arch in archs:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            vals = [final_acc(r) for r in runs]
            row += f"  {np.mean(vals):.3f}±{np.std(vals):.3f}"
        print(row)

    print("\n=== Emergence Rate (fraction of seeds with acc > 0.5) ===")
    print(header)
    print("-" * len(header))
    for lr in lrs:
        row = f"{lr:.0e}     "
        for arch in archs:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            frac = sum(1 for r in runs if final_acc(r) >= EMERGENCE_THRESHOLD) / max(1, len(runs))
            row += f"  {'YES' if frac > 0.5 else 'no ':>4} ({frac:.0%})  "
        print(row)

    # Overall emergence rate
    print("\n=== Overall emergence rate ===")
    for arch in archs:
        runs  = [r for r in data if r["arch"] == arch]
        frac  = sum(1 for r in runs if final_acc(r) >= EMERGENCE_THRESHOLD) / max(1, len(runs))
        n_params = runs[0]["n_params"] if runs else "?"
        print(f"  {arch:12s}: {frac:.0%} ({sum(1 for r in runs if final_acc(r) >= EMERGENCE_THRESHOLD)}/{len(runs)})   params={n_params:,}")


def main():
    data = load()
    print(f"Loaded {len(data)} runs")
    print_summary(data)
    plot_heatmap(data)
    plot_learning_curves(data)
    plot_emergence_fraction(data)
    print(f"\nFigures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
