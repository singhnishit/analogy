"""
Synthetic analogy task from Minegishi et al. (2602.01992).

Knowledge graph over two disjoint entity sets E1, E2 (each of size N).
Relations R are randomly assigned; each entity has unique outgoing edge labels.
Functor f: E1_i -> E2_i is a fixed bijection (identity-like structural mapping).

Three fact types (all as 3-token sequences [s, r, t]):
  - Atomic:       (e_s, r, e_t)           in-distribution
  - Compositional:(e_s, r1*r2, e_t)       2-hop, OOD
  - Analogical:   (e1_s, functor, e2_s)   cross-category, OOD

Model predicts the final token (e_t) given the first two.
Loss applied only to the final token.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def build_knowledge_graph(n_entities, n_relations, seed=0):
    """
    Returns:
      graph: dict {(src, tgt): relation_id} for both E1 and E2
      functor: array of shape (n_entities,) where functor[i] = j means E1_i maps to E2_j
               (identity functor: functor[i] = i)
    """
    rng = np.random.default_rng(seed)
    # Entity indices: E1 = [0..N-1], E2 = [N..2N-1]
    N = n_entities

    graph = {}  # (src, dst) -> relation_id

    # Assign relations for E1. Each source entity gets distinct outgoing labels.
    for src in range(N):
        available_relations = rng.permutation(n_relations)
        rel_idx = 0
        for tgt in range(N):
            if src != tgt:
                graph[(src, tgt)] = int(available_relations[rel_idx])
                rel_idx += 1

    # E2 uses the SAME relation tokens as E1 on corresponding edges.
    # Functor F: E1_i -> E2_i is structure-preserving:
    #   E1_i --(r)--> E1_j  =>  E2_i --(r)--> E2_j  (same relation token r)
    # This shared fingerprint is what makes analogy learnable.
    for src in range(N):
        for tgt in range(N):
            if src != tgt:
                graph[(N + src, N + tgt)] = graph[(src, tgt)]

    # Functor: E1_i -> E2_i (identity structural mapping)
    functor = np.arange(N)  # functor[i] = i means E1_i maps to E2_i

    return graph, functor


def generate_facts(graph, functor, n_entities, n_relations, ood_ratio=0.1, seed=0):
    """
    Generate atomic, compositional, and analogical facts.
    Returns train and test sets as lists of (input_ids, target_id).

    Vocab layout:
      [0..2N-1]          entity tokens (E1: 0..N-1, E2: N..2N-1)
      [2N..2N+R-1]       relation tokens
      [2N+R]             functor token
    """
    rng = np.random.default_rng(seed)
    N = n_entities
    R = n_relations
    functor_token = 2 * N + R

    # --- Atomic facts ---
    atomic = []
    for (src, tgt), rel in graph.items():
        rel_token = 2 * N + rel
        atomic.append(([src, rel_token], tgt))

    rng.shuffle(atomic)

    # --- Compositional facts (2-hop within same category) ---
    comp = []
    for cat_offset in [0, N]:
        for src in range(cat_offset, cat_offset + N):
            for mid in range(cat_offset, cat_offset + N):
                if src == mid:
                    continue
                for tgt in range(cat_offset, cat_offset + N):
                    if mid == tgt or src == tgt:
                        continue
                    if (src, mid) in graph and (mid, tgt) in graph:
                        r1_tok = 2 * N + graph[(src, mid)]
                        r2_tok = 2 * N + graph[(mid, tgt)]
                        comp.append(([src, r1_tok, r2_tok], tgt))

    rng.shuffle(comp)

    # --- Analogical facts ---
    # E1_i + functor -> E2_functor[i]
    analogical = []
    for i in range(N):
        src = i                   # E1 entity
        tgt = N + functor[i]      # E2 entity
        analogical.append(([src, functor_token], tgt))

    # --- OOD splits ---
    # Analogical: 90% in training (so model learns the functor concept),
    # 10% held out for OOD test. Mirrors Minegishi et al.'s protocol.
    n_ood_ana = max(1, int(len(analogical) * ood_ratio))
    train_analogical = analogical[n_ood_ana:]
    test_analogical  = analogical[:n_ood_ana]

    # Compositional: split by ood_ratio (10-30% withheld for test)
    n_ood_comp = max(1, int(len(comp) * ood_ratio))
    train_comp = comp[n_ood_comp:]
    test_comp  = comp[:n_ood_comp]

    return {
        "train_atomic":        atomic,
        "train_compositional": train_comp,
        "train_analogical":    train_analogical,
        "test_compositional":  test_comp,
        "test_analogical":     test_analogical,
    }


class FactDataset(Dataset):
    def __init__(self, facts, max_input_len=3):
        """
        facts: list of ([token, ...], target)
        Pads inputs to max_input_len.
        """
        self.inputs  = []
        self.targets = []
        for inp, tgt in facts:
            padded = inp + [0] * (max_input_len - len(inp))
            self.inputs.append(padded)
            self.targets.append(tgt)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx],  dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


def get_vocab_size(n_entities, n_relations):
    return 2 * n_entities + n_relations + 1  # +1 for functor token


def make_loaders(facts, batch_size=32):
    """Build DataLoaders for all splits. Training uses atomic + optional train splits."""
    loaders = {}
    train_facts = facts["train_atomic"]
    ds = FactDataset(train_facts, max_input_len=3)
    loaders["train"] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for key in ["train_compositional", "train_analogical", "test_compositional", "test_analogical"]:
        if facts[key]:
            ds = FactDataset(facts[key], max_input_len=3)
            loaders[key] = DataLoader(ds, batch_size=batch_size, shuffle=False)
        else:
            loaders[key] = None

    return loaders
