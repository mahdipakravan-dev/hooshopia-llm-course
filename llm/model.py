"""A GPT-style transformer, from scratch.

    python -m llm.model --data data/clean --tokenizer tokenizer.json --steps 500   # quick demo vs bigram

token ids ─► embedding ─► [ Block × n_layer ] ─► RMSNorm ─► lm_head ─► logits for the next token
                             │
                             ├─ x = x + Attention(RMSNorm(x))   tokens exchange information
                             └─ x = x + MLP(RMSNorm(x))         each token is processed on its own
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256     # longest context the model can see
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    """Rescale each vector to unit root-mean-square, then by a learned per-channel weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


# ---------------------------------------------------------------- rotary position embedding (RoPE)
def rope_cache(head_dim: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_len, head_dim/2). Pair i of a vector at position t is rotated
    by angle t * base^(-2i/head_dim): fast rotation for the first pairs, slow for the last."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs of x (..., T, head_dim) by the angle of their position.
    After rotation, q·k depends only on the distance between the two positions."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.type_as(x)


# ---------------------------------------------------------------- attention
def attention_manual(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """softmax(q·kᵀ / √d) · v, written out. q: (B, h, T, d), k/v: (B, h, S, d).
    The causal mask stops a token from looking at tokens that come after it."""
    T, S = q.size(-2), k.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))          # (B, h, T, S)
    if causal:
        allowed = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
        scores = scores.masked_fill(~allowed, float("-inf"))
    return F.softmax(scores.float(), dim=-1).type_as(q) @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.use_sdpa = True   # PyTorch's fused kernel; same math as attention_manual, much faster

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): every head attends independently
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0.0)
        else:
            y = attention_manual(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                   # heads back side by side
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)   # residual: each layer adds to x, never replaces it
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying: same matrix to read and write tokens
        cos, sin = rope_cache(cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():      # keep the residual stream from growing with depth
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:   # targets = -100 are ignored (used by SFT in part 7)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            probs = F.softmax(logits[:, -1, :].float() / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


def main() -> None:
    """Train a tiny GPT next to the bigram on the same data, to see attention pay off."""
    import argparse

    from .batches import encode_corpus, estimate_loss, get_batch
    from .bigram import BigramLM
    from .data import iter_split
    from .env import get_device
    from .tokenizer import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--block-size", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = get_device()
    tok = Tokenizer.load(args.tokenizer)
    data = encode_corpus(tok, iter_split(args.data, "train"))
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=4, n_head=4, n_embd=128)
    for name, model, lr in [("bigram", BigramLM(tok.vocab_size), 1e-2), ("gpt", GPT(cfg), 3e-3)]:
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for _ in range(args.steps):
            _, loss = model(*get_batch(data, 16, args.block_size, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        n = sum(p.numel() for p in model.parameters())
        print(f"{name:>6}: {n:>9,} params | train loss {estimate_loss(model, data, 16, args.block_size, device=device):.3f}")


if __name__ == "__main__":
    main()
