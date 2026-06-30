"""
Two model variants for the analogy emergence experiment:

1. VanillaTransformer  — standard causal GPT-2-style (emb -> self-attn -> FFN)
2. AbstractorTransformer — same, but each block also has a Relational Cross-Attention
   (RCA) sublayer between self-attention and FFN.

RCA (from Altabaa et al. 2304.00195):
  - Q and K are derived from the input X (same as self-attention)
  - V comes from LEARNED POSITION SYMBOLS S (not from X)
  => output encodes relational structure using symbolic values,
     disentangling relational patterns from object-level features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.qkv(x).split(C, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K, V = reshape(Q), reshape(K), reshape(V)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(~self.mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class RelationalCrossAttention(nn.Module):
    """
    Computes attention weights from input (Q, K from X) but uses LEARNED
    POSITION SYMBOLS as values. This disentangles relational structure
    (which positions attend to which) from object content.
    """
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.wq      = nn.Linear(d_model, d_model, bias=False)
        self.wk      = nn.Linear(d_model, d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        # Learned position symbols — the key architectural distinction
        self.symbols = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.wq(x)
        K = self.wk(x)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K = reshape(Q), reshape(K)
        # Symbols as values (broadcast over batch)
        S = self.symbols[:T].unsqueeze(0).expand(B, -1, -1)
        S = S.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(~self.mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out  = (attn @ S).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class VanillaBlock(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class AbstractorBlock(nn.Module):
    """
    Block with an extra RCA sublayer between self-attention and FFN.
    This gives the model a structural/symbolic pathway alongside the
    content pathway from self-attention.
    """
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.rca  = RelationalCrossAttention(d_model, n_heads, seq_len)
        self.ln3  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.rca(self.ln2(x))    # relational symbolic path
        x = x + self.ffn(self.ln3(x))
        return x


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------

class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=1, n_layers=1, seq_len=3):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks  = nn.ModuleList([
            VanillaBlock(d_model, n_heads, seq_len) for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: output logits = h @ tok_emb.T so prediction uses embedding geometry
        self.head.weight = self.tok_emb.weight
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h    = self.ln_f(h)
        return self.head(h)   # [B, T, vocab_size]


class AbstractorTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=1, n_layers=1, seq_len=3):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks  = nn.ModuleList([
            AbstractorBlock(d_model, n_heads, seq_len) for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: output logits = h @ tok_emb.T so prediction uses embedding geometry
        self.head.weight = self.tok_emb.weight
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h    = self.ln_f(h)
        return self.head(h)


def make_model(arch, vocab_size, d_model=128, n_heads=1, n_layers=1, seq_len=3):
    if arch == "vanilla":
        return VanillaTransformer(vocab_size, d_model, n_heads, n_layers, seq_len)
    elif arch == "abstractor":
        return AbstractorTransformer(vocab_size, d_model, n_heads, n_layers, seq_len)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
