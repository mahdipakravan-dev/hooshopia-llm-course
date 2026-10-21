"""Evaluate one or more checkpoints and write a side-by-side report.

    python -m llm.evaluate --ckpt checkpoints/small/ckpt.pt checkpoints/sft/ckpt.pt \
        --tokens data/tokens --prompts evals/prompts.jsonl --out reports/eval.md

Two kinds of numbers:
- val loss and perplexity: how well the model predicts text it never trained on.
  Perplexity = exp(loss) ≈ "how many tokens the model is hesitating between". Untrained ≈ vocab size.
- answers to a fixed list of prompts: the only way to see if the assistant is actually useful.
  The last prompts are deliberately unanswerable; a good model says it doesn't know.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from .chat_cli import reply
from .checkpoint import load_model
from .env import get_device


@torch.no_grad()
def perplexity(model, data: np.ndarray, device: str = "cpu", max_tokens: int | None = None,
               batch_size: int = 8) -> tuple[float, float]:
    """Average next-token loss over consecutive, non-overlapping windows of the whole split
    (deterministic, unlike the random batches used during training). Returns (loss, perplexity)."""
    model.eval()
    T = model.cfg.block_size
    n = len(data) if max_tokens is None else min(len(data), max_tokens + 1)
    starts = list(range(0, n - 1, T))
    full = [s for s in starts if s + T <= n - 1]
    batches = [(full[i:i + batch_size], T) for i in range(0, len(full), batch_size)]
    if len(full) < len(starts):                        # the last, shorter window on its own
        batches.append(([starts[-1]], n - 1 - starts[-1]))
    total, count = 0.0, 0
    for chunk, L in batches:
        x = torch.from_numpy(np.stack([data[s:s + L].astype(np.int64) for s in chunk])).to(device)
        y = torch.from_numpy(np.stack([data[s + 1:s + 1 + L].astype(np.int64) for s in chunk])).to(device)
        loss = model(x, y)[1].item()
        total, count = total + loss * x.numel(), count + x.numel()
    loss = total / count
    return loss, math.exp(loss)


def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["prompt"] for line in f if line.strip()]


def evaluate(ckpts: list[str], prompts: list[str], tokens_dir: str | None, device: str,
             max_new_tokens: int = 120, max_eval_tokens: int | None = 2_000_000) -> dict:
    results = {}
    for path in ckpts:
        model, tok = load_model(path, device)
        r = {"params": model.num_params(), "answers": []}
        if tokens_dir:
            from .pretrain import open_tokens
            r["val_loss"], r["val_ppl"] = perplexity(model, open_tokens(tokens_dir, "val"), device, max_eval_tokens)
        for p in prompts:
            text = "".join(reply(model, tok, [{"role": "user", "content": p}],
                                 max_new_tokens=max_new_tokens, temperature=0))
            r["answers"].append(text.strip())
        results[path] = r
    return results


def to_markdown(results: dict, prompts: list[str]) -> str:
    names = list(results)
    lines = ["# Evaluation report", "", "| checkpoint | params | val loss | val perplexity |", "|---|---|---|---|"]
    for n in names:
        r = results[n]
        loss = f"{r['val_loss']:.4f}" if "val_loss" in r else "-"
        ppl = f"{r['val_ppl']:.1f}" if "val_ppl" in r else "-"
        lines.append(f"| `{n}` | {r['params'] / 1e6:.2f}M | {loss} | {ppl} |")
    lines += ["", "## Answers (greedy)", ""]
    for i, p in enumerate(prompts):
        lines.append(f"### {i + 1}. {p}")
        for n in names:
            ans = results[n]["answers"][i].replace("\n", " ") or "(empty)"
            lines.append(f"- `{n}`: {ans}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--tokens", help="data/tokens directory, for val loss and perplexity")
    ap.add_argument("--prompts", default="evals/prompts.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", default="reports/eval.md")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    results = evaluate(args.ckpt, prompts, args.tokens, get_device(), args.max_new_tokens)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(to_markdown(results, prompts))
    for n, r in results.items():
        extra = f" | val loss {r['val_loss']:.4f} | ppl {r['val_ppl']:.1f}" if "val_loss" in r else ""
        print(f"{n}: {r['params'] / 1e6:.2f}M params{extra}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
