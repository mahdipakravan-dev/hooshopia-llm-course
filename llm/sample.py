"""Generating text: how the next token is picked, and the KV cache that makes it fast.

    python -m llm.sample --ckpt checkpoints/small/ckpt.pt --prompt "هوش مصنوعی"
    python -m llm.sample --ckpt ... --prompt "..." --temperatures 0 0.7 1.3     # same prompt, three temperatures
    python -m llm.sample --ckpt ... --prompt "..." --bench                       # with vs without KV cache
"""
import argparse
import time
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

from .model import GPT, KVCache


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float = 1.0) -> torch.Tensor:
    """Set the logits of tokens we never want to pick to -inf.
    top_k: keep only the k most likely tokens.
    top_p: keep the smallest set of most-likely tokens whose probabilities add up to top_p."""
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p         # the most likely token is never dropped
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return logits


def pick(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float = 1.0,
         generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) -> next token ids (B, 1). temperature 0 = always the most likely token (greedy)."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = filter_logits(logits.float() / temperature, top_k, top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=generator)


@torch.no_grad()
def stream(model: GPT, prompt_ids: list[int], max_new_tokens: int = 200, temperature: float = 0.8,
           top_k: int | None = 50, top_p: float = 0.95, stop_ids: Iterable[int] = (), use_cache: bool = True,
           seed: int | None = None) -> Iterator[int]:
    """Yield new token ids one by one (batch size 1)."""
    model.eval()
    device = next(model.parameters()).device
    block = model.cfg.block_size
    stop_ids = set(stop_ids)
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    tokens = list(prompt_ids)[-block:]

    def prefill(ctx: list[int]):
        cache = KVCache(model.cfg.n_layer) if use_cache else None
        logits, _ = model(torch.tensor([ctx], device=device), cache=cache)
        return logits[:, -1], cache

    logits, cache = prefill(tokens)
    for _ in range(max_new_tokens):
        nxt = pick(logits, temperature, top_k, top_p, g).item()
        yield nxt
        if nxt in stop_ids:
            return
        tokens.append(nxt)
        if not use_cache:
            logits, _ = prefill(tokens[-block:])
        elif cache.seq_len >= block:            # context full: start over from the most recent half
            logits, cache = prefill(tokens[-(block // 2):])
        else:                                    # only the new token goes through the model
            logits, _ = model(torch.tensor([[nxt]], device=device), cache=cache)
            logits = logits[:, -1]


def generate_text(model: GPT, tok, prompt: str, **kw) -> str:
    return tok.decode(list(stream(model, tok.encode(prompt), **kw)))


def main() -> None:
    from .checkpoint import load_model
    from .env import get_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.8])
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num", type=int, default=1, help="samples per temperature")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--bench", action="store_true", help="time generation with and without the KV cache")
    args = ap.parse_args()

    device = get_device()
    model, tok = load_model(args.ckpt, device)
    prompt_ids = tok.encode(args.prompt) or [tok.eot_id]

    if args.bench:
        for use_cache in (False, True):
            t0 = time.time()
            n = sum(1 for _ in stream(model, prompt_ids, args.max_new_tokens, temperature=0, use_cache=use_cache))
            print(f"kv cache {'on ' if use_cache else 'off'}: {n / (time.time() - t0):7.1f} tokens/s")
        return

    for t in args.temperatures:
        for i in range(args.num):
            ids = stream(model, prompt_ids, args.max_new_tokens, t, args.top_k, args.top_p,
                         stop_ids={tok.eot_id}, use_cache=not args.no_cache, seed=args.seed + i)
            print(f"--- temperature {t} #{i + 1}\n{args.prompt}{tok.decode(list(ids))}\n")


if __name__ == "__main__":
    main()
