"""
Makes all paper figures from multiseed_comparison.json (wd=0.01, corrected setup).
Also reads sweep_results.json (wd=0, zero-shot) for the negative control.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGS_DIR    = os.path.join(RESULTS_DIR, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

EMERGENCE_THRESHOLD = 0.5
COLORS = {"vanilla": "#4C72B0", "abstractor": "#DD8452"}
LRATE  = [1e-4, 3e-4, 1e-3, 3e-3]
ARCHS  = ["vanilla", "abstractor"]


def load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def final_acc(r, key="ana_acc"):
    return r.get(key, r.get("final", {}).get("test_analogical", 0.0))


# ── Fig 1: Emergence heatmap (mean over seeds, wd=0.01) ────────────────────

def fig_heatmap(data, outname="fig1_heatmap"):
    lrs = LRATE
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        mat = []
        for lr in lrs:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            mat.append(np.mean([final_acc(r) for r in runs]) if runs else 0.0)
        mat = np.array(mat).reshape(-1, 1)
        im  = ax.imshow(mat, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn", origin="upper")
        ax.set_title(arch.capitalize(), fontsize=13, fontweight="bold")
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lrs])
        ax.set_ylabel("Learning Rate")
        ax.set_xticks([])
        for i, v in enumerate(mat.flatten()):
            ax.text(0, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                    color="black" if v < 0.7 else "white")
    plt.colorbar(im, ax=axes[-1], label="Analogy Accuracy")
    plt.suptitle("Final Analogical Accuracy (mean over seeds, wd=0.01)", fontsize=12)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS_DIR, f"{outname}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {outname}")


# ── Fig 2: Emergence rate bar chart ────────────────────────────────────────

def fig_emergence_rate(data, outname="fig2_emergence_rate"):
    lrs = LRATE
    x   = np.arange(len(lrs))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, arch in enumerate(ARCHS):
        fracs = []
        for lr in lrs:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            frac = sum(1 for r in runs if final_acc(r) >= EMERGENCE_THRESHOLD) / max(1, len(runs))
            fracs.append(frac)
        bars = ax.bar(x + (i - 0.5) * w, fracs, w, label=arch.capitalize(),
                      color=COLORS[arch], alpha=0.85)
        for bar, frac in zip(bars, fracs):
            if frac > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{frac:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])
    ax.set_xlabel("Learning Rate"); ax.set_ylabel(f"Fraction of seeds emerged (>{EMERGENCE_THRESHOLD:.0%})")
    ax.set_ylim(0, 1.2); ax.legend()
    ax.set_title("Emergence Rate: Abstractor vs Vanilla (wd=0.01)")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS_DIR, f"{outname}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {outname}")


# ── Fig 3: Dirichlet Energy ────────────────────────────────────────────────

def fig_dirichlet(data, outname="fig3_dirichlet_energy"):
    lrs = LRATE
    x   = np.arange(len(lrs))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, arch in enumerate(ARCHS):
        means = []
        errs  = []
        for lr in lrs:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            des  = [r.get("de", float("nan")) for r in runs]
            des  = [d for d in des if not np.isnan(d)]
            means.append(np.mean(des) if des else 0)
            errs.append(np.std(des) if len(des) > 1 else 0)
        ax.bar(x + (i - 0.5)*w, means, w, yerr=errs,
               label=arch.capitalize(), color=COLORS[arch], alpha=0.85,
               error_kw={"capsize": 4})
    ax.set_xticks(x); ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Dirichlet Energy ↓ (lower = better geometric alignment)")
    ax.legend(); ax.set_title("Geometric Alignment: Dirichlet Energy (wd=0.01)")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS_DIR, f"{outname}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {outname}")


# ── Fig 4: Learning curves (analogical accuracy vs step) ─────────────────

def fig_learning_curves(data, outname="fig4_learning_curves"):
    lrs = LRATE
    lr_colors = dict(zip(lrs, plt.cm.viridis(np.linspace(0, 1, len(lrs)))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        for lr in lrs:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            if not runs: continue
            curves = [r.get("curve", []) for r in runs]
            all_steps = sorted(set(s for c in curves for s, _ in c))
            means = []
            for step in all_steps:
                vals = [a for c in curves for s, a in c if s == step]
                means.append(np.mean(vals) if vals else np.nan)
            ax.plot(all_steps, means, color=lr_colors[lr],
                    label=f"lr={lr:.0e}", linewidth=2)
            for c in curves:
                steps = [s for s, _ in c]; accs = [a for _, a in c]
                ax.plot(steps, accs, color=lr_colors[lr], linewidth=0.5, alpha=0.3)
        ax.axhline(EMERGENCE_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Steps"); ax.set_ylabel("Analogical Accuracy")
        ax.set_title(arch.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)
    plt.suptitle("Analogical Accuracy vs Training Steps (wd=0.01)", fontsize=12)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS_DIR, f"{outname}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {outname}")


# ── Summary table ────────────────────────────────────────────────────────

def print_table(data):
    lrs = LRATE
    print("\n=== Analogical Accuracy (mean ± std) — wd=0.01 ===")
    print(f"{'LR':>8}  {'vanilla':>15}  {'abstractor':>15}  {'winner':>10}")
    print("-" * 55)
    for lr in lrs:
        vals = {}
        for arch in ARCHS:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
            v = [final_acc(r) for r in runs]
            vals[arch] = v
        v_mu = np.mean(vals["vanilla"]) if vals["vanilla"] else 0
        a_mu = np.mean(vals["abstractor"]) if vals["abstractor"] else 0
        v_std = np.std(vals["vanilla"]) if vals["vanilla"] else 0
        a_std = np.std(vals["abstractor"]) if vals["abstractor"] else 0
        winner = "abstractor" if a_mu > v_mu else ("vanilla" if v_mu > a_mu else "tie")
        print(f"{lr:>8.0e}  {v_mu:.3f}±{v_std:.3f}   {a_mu:.3f}±{a_std:.3f}   {winner}")

    print("\n=== Dirichlet Energy (mean ± std) — lower = better ===")
    print(f"{'LR':>8}  {'vanilla':>15}  {'abstractor':>15}  {'winner':>10}")
    print("-" * 55)
    for lr in lrs:
        for arch in ARCHS:
            runs = [r for r in data if r["arch"] == arch and r["lr"] == lr]
        de_v = np.mean([r.get("de",0) for r in data if r["arch"]=="vanilla" and r["lr"]==lr]) if any(r["lr"]==lr for r in data if r["arch"]=="vanilla") else 0
        de_a = np.mean([r.get("de",0) for r in data if r["arch"]=="abstractor" and r["lr"]==lr]) if any(r["lr"]==lr for r in data if r["arch"]=="abstractor") else 0
        winner = "abstractor" if de_a < de_v else ("vanilla" if de_v < de_a else "tie")
        print(f"{lr:>8.0e}  {de_v:>15.4f}  {de_a:>15.4f}  {winner}")


def fig_grokking_comparison(data, target_lr=3e-4, outname="fig5_grokking_comparison"):
    """Compare vanilla vs abstractor learning curves at a specific LR."""
    fig, ax = plt.subplots(figsize=(7, 4))
    found_any = False
    for arch in ARCHS:
        runs = [r for r in data if r["arch"] == arch
                and abs(r["lr"] - target_lr) < target_lr * 0.01]
        if not runs:
            continue
        for r in runs:
            curve = r.get("curve", [])
            if not curve:
                continue
            steps = [s for s, _ in curve]
            accs  = [a for _, a in curve]
            seed  = r.get("seed", 0)
            label = f"{arch.capitalize()} s={seed}" if len(runs) > 1 else arch.capitalize()
            ax.plot(steps, accs, color=COLORS[arch],
                    alpha=0.5 if len(runs) > 1 else 1.0,
                    linewidth=1.5 if len(runs) == 1 else 1.0,
                    label=label)
            found_any = True
    if not found_any:
        plt.close()
        print(f"No data for lr={target_lr:.0e}, skipping {outname}")
        return
    ax.axhline(EMERGENCE_THRESHOLD, color="gray", linestyle="--", linewidth=1,
               label="Emergence threshold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Analogical Test Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Grokking: Vanilla vs Abstractor (lr={target_lr:.0e}, wd=0.01)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS_DIR, f"{outname}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {outname}")


def main():
    main_data = load("multiseed_comparison.json")
    if not main_data:
        main_data = load("fast_comparison.json")
    # Always merge with all_comparison_data.json (comparison job results) if available
    comparison_data = load("all_comparison_data.json")
    if comparison_data:
        existing_keys = {(r["arch"], r["lr"], r.get("seed", 0)) for r in main_data}
        new_records = [r for r in comparison_data
                       if (r["arch"], r["lr"], r.get("seed", 0)) not in existing_keys]
        main_data = main_data + new_records
        if new_records:
            print(f"Merged {len(new_records)} records from all_comparison_data.json")
    if not main_data:
        print("No data found. Run experiments first.")
        return
    print(f"Total {len(main_data)} runs")
    print_table(main_data)
    fig_heatmap(main_data)
    fig_emergence_rate(main_data)
    fig_dirichlet(main_data)
    fig_learning_curves(main_data)
    # Grokking comparison at each LR where we have both archs
    for lr in LRATE:
        both = [r for r in main_data
                if abs(r["lr"] - lr) < lr * 0.01
                and r.get("curve", [])]
        archs_present = {r["arch"] for r in both}
        if len(archs_present) == 2:
            fig_grokking_comparison(main_data, target_lr=lr,
                                    outname=f"fig5_grokking_lr{lr:.0e}")
    print(f"\nAll figures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
