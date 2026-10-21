"""Supervised fine-tuning (SFT): turn the pretrained text-continuer into an assistant.

    python -m llm.sft --ckpt checkpoints/small/ckpt.pt --data data/sample/sft.jsonl --out checkpoints/sft

Data: one conversation per line, {"messages": [{"role": "user", "content": ...},
                                               {"role": "assistant", "content": ...}, ...]}

Same training loop as pretraining, with three differences:
- the data is conversations in the chat format (llm/chat.py)
- the loss only counts the assistant's tokens (targets elsewhere are -100)
- a much smaller learning rate, so the model keeps what it learned in pretraining
"""
import argparse
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass

import torch

from .chat import build_example
from .checkpoint import load_checkpoint, save_checkpoint
from .env import get_device
from .tokenizer import Tokenizer


@dataclass
class SFTConfig:
    ckpt: str = "checkpoints/small/ckpt.pt"
    data: str = "data/sample/sft.jsonl"
    out_dir: str = "checkpoints/sft"
    steps: int = 1000
    batch_size: int = 16
    lr: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_every: int = 10          # every N-th conversation is held out
    eval_every: int = 100
    seed: int = 1337


def load_conversations(path: str) -> list[list[dict]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def collate(examples, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (x, y) to the longest one. Padded targets are -100 so they don't count."""
    T = max(len(x) for x, _ in examples)
    X = torch.full((len(examples), T), pad_id, dtype=torch.long)
    Y = torch.full((len(examples), T), -100, dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        X[i, :len(x)], Y[i, :len(y)] = x, y
    return X, Y


@torch.no_grad()
def masked_loss(model, examples, pad_id: int, device: str, batch_size: int = 16) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(examples), batch_size):
        x, y = collate(examples[i:i + batch_size], pad_id)
        loss = model(x.to(device), y.to(device))[1]
        k = (y != -100).sum().item()
        total, n = total + loss.item() * k, n + k
    model.train()
    return total / max(n, 1)


def train_sft(cfg: SFTConfig, device: str | None = None) -> dict:
    device = device or get_device()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model, _ = load_checkpoint(cfg.ckpt, device)
    tok_path = os.path.join(os.path.dirname(cfg.ckpt), "tokenizer.json")
    tok = Tokenizer.load(tok_path)

    convs = load_conversations(cfg.data)
    examples = [build_example(tok, c, model.cfg.block_size) for c in convs]
    examples = [e for e in examples if (e[1] != -100).any()]
    val = [e for i, e in enumerate(examples) if i % cfg.val_every == 0]
    train = [e for i, e in enumerate(examples) if i % cfg.val_every != 0] or val
    print(f"{len(train)} train / {len(val)} val conversations | params {model.num_params() / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    history = {"step": [], "val": []}
    os.makedirs(cfg.out_dir, exist_ok=True)
    shutil.copy(tok_path, os.path.join(cfg.out_dir, "tokenizer.json"))
    model.train()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            v = masked_loss(model, val, tok.eot_id, device)
            history["step"].append(step); history["val"].append(v)
            print(f"step {step:>5} | val (answers only) {v:.4f}")
        if step == cfg.steps:
            break
        lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup_steps) * 0.5 * (1 + math.cos(math.pi * step / cfg.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = collate(random.sample(train, min(cfg.batch_size, len(train))), tok.eot_id)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    save_checkpoint(os.path.join(cfg.out_dir, "ckpt.pt"), model, step=cfg.steps, sft_cfg=asdict(cfg))
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, v in asdict(SFTConfig()).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--out", dest="out_dir_alias")
    ap.add_argument("--device")
    args = vars(ap.parse_args())
    device = args.pop("device")
    alias = args.pop("out_dir_alias")
    if alias:
        args["out_dir"] = alias
    train_sft(SFTConfig(**args), device)


if __name__ == "__main__":
    main()
