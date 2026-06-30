"""
Sweep: architecture × learning_rate × ood_ratio × seed
Logs accuracy on atomic / compositional / analogical (train and test splits)
every eval_interval steps. Saves results to results/sweep_results.json.

Experiment design mirrors Minegishi et al. (2602.01992):
  - N=10 entities per category, R=1000 relations
  - 1-layer, 1-head transformer, d=128
  - Adam optimiser, fixed 30k training steps
  - Loss on final token only (cross-entropy)

Our additions:
  - Architecture: vanilla vs abstractor
  - LR sweep: [1e-4, 3e-4, 1e-3, 3e-3]
  - OOD ratio: [0.1, 0.3]
  - 3 seeds
"""

import json
import os
import time
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim

from data import build_knowledge_graph, generate_facts, get_vocab_size, FactDataset
from models import make_model, count_params
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
N_ENTITIES   = 20     # 20 per category; ood=0.3 → 6 analogical test facts
N_RELATIONS  = 1000
D_MODEL      = 128
N_HEADS      = 1
N_LAYERS     = 1
BATCH_SIZE   = 32
N_STEPS      = 50_000
EVAL_INTERVAL = 5_000
SEQ_LEN_ATOMIC = 2   # input length (s, r); target is position 2
MAX_SEQ_LEN    = 3   # full sequence length

ARCHITECTURES = ["vanilla", "abstractor"]
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]
OOD_RATIOS     = [0.3]   # 30% held out: 6 test examples from 20 analogy facts
SEEDS          = [0, 1, 2]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "sweep_results.json")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def evaluate(model, facts_dict, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for split, facts in facts_dict.items():
            if not facts:
                continue
            ds = FactDataset(facts, max_input_len=SEQ_LEN_ATOMIC)
            loader = DataLoader(ds, batch_size=256, shuffle=False)
            correct = total = 0
            for inp, tgt in loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)          # [B, T, V]
                pred   = logits[:, -1, :].argmax(dim=-1)
                correct += (pred == tgt).sum().item()
                total   += tgt.size(0)
            results[split] = correct / total if total > 0 else 0.0
    model.train()
    return results


def run_one(arch, lr, ood_ratio, seed, vocab_size, graph, functor):
    torch.manual_seed(seed)

    facts = generate_facts(graph, functor, N_ENTITIES, N_RELATIONS,
                           ood_ratio=ood_ratio, seed=seed)

    # Training data: atomic facts + in-distribution analogical facts
    # (model must see some functor examples to learn functor concept;
    #  OOD analogical facts are the held-out test set)
    train_facts = facts["train_atomic"] + facts["train_analogical"]
    eval_splits = {k: v for k, v in facts.items()
                   if k not in ("train_atomic", "train_analogical")}

    train_ds  = FactDataset(train_facts, max_input_len=SEQ_LEN_ATOMIC)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_iter   = iter(train_loader)

    model     = make_model(arch, vocab_size, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    model     = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    history = []   # list of {step, acc_...}
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        try:
            inp, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inp, tgt   = next(train_iter)

        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        logits   = model(inp)          # [B, T, V]
        loss     = criterion(logits[:, -1, :], tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            accs = evaluate(model, eval_splits, DEVICE)
            record = {"step": step, "train_loss": loss.item(), **accs}
            history.append(record)

    elapsed = time.time() - t0
    final_accs = evaluate(model, eval_splits, DEVICE)
    return {
        "arch":      arch,
        "lr":        lr,
        "ood_ratio": ood_ratio,
        "seed":      seed,
        "n_params":  count_params(model),
        "history":   history,
        "final":     final_accs,
        "elapsed_s": elapsed,
    }


# ── Main sweep ───────────────────────────────────────────────────────────────

def main():
    vocab_size = get_vocab_size(N_ENTITIES, N_RELATIONS)
    print(f"Vocab size: {vocab_size}")

    # Build one shared graph (same structure across all runs for this data seed)
    graph, functor = build_knowledge_graph(N_ENTITIES, N_RELATIONS, seed=42)

    combos = list(product(ARCHITECTURES, LEARNING_RATES, OOD_RATIOS, SEEDS))
    total  = len(combos)
    print(f"Total runs: {total}")

    all_results = []

    # Load checkpoint if exists
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        done_keys = {(r["arch"], r["lr"], r["ood_ratio"], r["seed"]) for r in all_results}
        print(f"Resuming: {len(done_keys)}/{total} runs already done")
    else:
        done_keys = set()

    for i, (arch, lr, ood_ratio, seed) in enumerate(combos):
        key = (arch, lr, ood_ratio, seed)
        if key in done_keys:
            continue

        tag = f"[{i+1}/{total}] arch={arch} lr={lr} ood={ood_ratio} seed={seed}"
        print(f"\n{tag}")

        result = run_one(arch, lr, ood_ratio, seed, vocab_size, graph, functor)

        final = result["final"]
        print(f"  analogy_test={final.get('test_analogical', 0):.3f}  "
              f"comp_test={final.get('test_compositional', 0):.3f}  "
              f"elapsed={result['elapsed_s']:.1f}s")

        all_results.append(result)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
