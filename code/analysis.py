"""
Geometric alignment analysis after the sweep.

Minegishi et al. show that analogical reasoning requires geometric alignment
of relational structure in the embedding space — formally, that the E1 entity
embeddings and the E2 entity embeddings are related by a near-isometry
(a functor-like map in embedding space).

We measure this via Dirichlet Energy (DE): for a functor F: E1 → E2,
  DE(F) = mean_{(i,j): edge(E1_i, E1_j)} || (e2_i - e2_j) - (e1_i - e1_j) ||²
where e1_i = embed(E1 entity i) and e2_i = embed(E2 entity i).

Lower DE → better geometric alignment → better conditions for analogy.

We also measure embedding similarity: how well the E2 embeddings predict
the E1 embeddings after a linear transform (analogical probing).
"""

import json, sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import build_knowledge_graph, generate_facts, get_vocab_size, FactDataset
from models import make_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGS_DIR    = os.path.join(RESULTS_DIR, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

N_ENTITIES  = 20
N_RELATIONS = 1000
DEVICE      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_model(arch, lr, wd, seed, graph, functor, facts, vocab_size,
                n_steps=15_000, batch_size=32):
    torch.manual_seed(seed)
    train_facts = facts["train_atomic"] + facts["train_analogical"]
    model = make_model(arch, vocab_size, seq_len=3).to(DEVICE)
    ds = FactDataset(train_facts, max_input_len=2)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    it   = iter(loader)
    for step in range(n_steps):
        try: inp, tgt = next(it)
        except StopIteration:
            it = iter(loader); inp, tgt = next(it)
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        logits = model(inp)
        loss   = crit(logits[:, -1, :], tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def get_entity_embeddings(model, n_entities):
    """Extract E1 and E2 entity token embeddings."""
    N = n_entities
    e1_idx = torch.arange(N).to(DEVICE)
    e2_idx = torch.arange(N, 2*N).to(DEVICE)
    with torch.no_grad():
        emb_e1 = model.tok_emb(e1_idx).cpu().numpy()   # [N, d]
        emb_e2 = model.tok_emb(e2_idx).cpu().numpy()   # [N, d]
    return emb_e1, emb_e2


def dirichlet_energy(emb_e1, emb_e2, graph, functor, n_entities):
    """
    Measure geometric alignment between E1 and E2 embeddings.
    For each edge (i, j) in E1, compare the relational vector in E1
    to the corresponding vector in E2 (via the functor mapping).
    Lower = better alignment.
    """
    N = n_entities
    energies = []
    for (src, tgt) in graph:
        if src >= N:   # only E1 edges
            continue
        i, j = src, tgt
        if i >= N or j >= N:
            continue
        fi, fj = functor[i], functor[j]   # functor images
        # relational vector in E1
        v1 = emb_e1[i] - emb_e1[j]       # [d]
        # corresponding relational vector in E2
        v2 = emb_e2[fi] - emb_e2[fj]     # [d]
        energies.append(np.sum((v1 - v2) ** 2))
    return float(np.mean(energies)) if energies else float("nan")


def analogy_probing_acc(emb_e1, emb_e2, n_entities):
    """
    Linear probe: train a linear map W: R^d → R^d such that W @ e1_i ≈ e2_i.
    Report the fraction of entities correctly retrieved (nearest-neighbour).
    """
    N = n_entities
    E1 = torch.tensor(emb_e1, dtype=torch.float32)
    E2 = torch.tensor(emb_e2, dtype=torch.float32)
    W  = nn.Linear(E1.shape[1], E2.shape[1], bias=True)
    opt = torch.optim.Adam(W.parameters(), lr=1e-2)
    for _ in range(2000):
        loss = ((W(E1) - E2) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred  = W(E1)   # [N, d]
        # for each E1 entity, find nearest E2 neighbour
        dists = torch.cdist(pred, E2)   # [N, N]
        nn_idx = dists.argmin(dim=-1)   # [N]
        targets = torch.arange(N)       # E2_i is the functor image of E1_i
        acc = (nn_idx == targets).float().mean().item()
    return acc


def run_analysis(arch_list, lrs, wd=0.01, seed=0):
    graph, functor = build_knowledge_graph(N_ENTITIES, N_RELATIONS, seed=42)
    facts = generate_facts(graph, functor, N_ENTITIES, N_RELATIONS, ood_ratio=0.3, seed=seed)
    vocab_size = get_vocab_size(N_ENTITIES, N_RELATIONS)

    results = []
    for arch in arch_list:
        for lr in lrs:
            print(f"Training {arch} lr={lr:.0e} wd={wd}...", flush=True)
            model = train_model(arch, lr, wd, seed, graph, functor, facts,
                                vocab_size, n_steps=15_000)
            emb_e1, emb_e2 = get_entity_embeddings(model, N_ENTITIES)
            de  = dirichlet_energy(emb_e1, emb_e2, graph, functor, N_ENTITIES)
            pba = analogy_probing_acc(emb_e1, emb_e2, N_ENTITIES)

            # direct analogical accuracy on test set
            model.eval()
            test_ds  = FactDataset(facts["test_analogical"], max_input_len=2)
            test_ldr = DataLoader(test_ds, batch_size=256)
            c = t = 0
            with torch.no_grad():
                for inp, tgt in test_ldr:
                    inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
                    pred = model(inp)[:, -1, :].argmax(-1)
                    c += (pred == tgt).sum().item(); t += tgt.size(0)
            ana_acc = c / t if t > 0 else 0

            results.append({
                "arch": arch, "lr": lr, "wd": wd, "seed": seed,
                "dirichlet_energy": de,
                "probing_acc": pba,
                "analogy_acc": ana_acc,
            })
            print(f"  DE={de:.4f}  probing={pba:.3f}  analogy_acc={ana_acc:.3f}")

    return results


def plot_analysis(results):
    archs = sorted(set(r["arch"] for r in results))
    lrs   = sorted(set(r["lr"]   for r in results))
    x     = np.arange(len(lrs))
    width = 0.35
    colors = {"vanilla": "#4C72B0", "abstractor": "#DD8452"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for metric, ax, title, better in [
        ("dirichlet_energy", axes[0], "Dirichlet Energy (↓ = better alignment)", "lower"),
        ("probing_acc",      axes[1], "Probing Accuracy (↑ = better alignment)", "higher"),
        ("analogy_acc",      axes[2], "Direct Analogical Accuracy (↑)",          "higher"),
    ]:
        for i, arch in enumerate(archs):
            vals = [next(r[metric] for r in results if r["arch"]==arch and r["lr"]==lr)
                    for lr in lrs]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=arch.capitalize(), color=colors.get(arch), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])
        ax.set_xlabel("Learning Rate")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

    plt.suptitle("Geometric Alignment & Analogical Reasoning (wd=0.01, 1 seed)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "alignment_analysis.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGS_DIR, "alignment_analysis.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: alignment_analysis")


if __name__ == "__main__":
    results = run_analysis(
        arch_list=["vanilla", "abstractor"],
        lrs=[1e-4, 3e-4, 1e-3, 3e-3],
        wd=0.01,
        seed=0,
    )
    plot_analysis(results)
    with open(os.path.join(RESULTS_DIR, "alignment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nAlignment analysis complete.")
