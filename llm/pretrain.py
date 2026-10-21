"""Pretraining: teach the GPT to predict the next token on the whole corpus.

    python -m llm.pretrain --config configs/tiny.json                 # smoke test, runs on CPU
    python -m llm.pretrain --config configs/small.json                # single GPU
    python -m llm.pretrain --config configs/small.json --set max_steps=1000 batch_size=16
    python -m llm.pretrain --config configs/small.json --set resume=true

What is new compared with the bigram loop
- data is read from disk (memmap) instead of RAM
- AdamW with weight decay only on matrices, gradient clipping
- learning rate: linear warmup, then cosine decay to min_lr
- mixed precision (bfloat16 on GPU) and gradient accumulation for large effective batches
- periodic evaluation, best-checkpoint saving and resume
"""
import argparse
import contextlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .env import get_device
from .model import GPT, GPTConfig


@dataclass
class TrainConfig:
    data_dir: str = "data/tokens"
    tokenizer: str = "tokenizer.json"
    out_dir: str = "checkpoints/pretrain"
    # model
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    # optimization
    batch_size: int = 32
    grad_accum: int = 1            # effective batch = batch_size * grad_accum sequences
    max_steps: int = 5000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # bookkeeping
    eval_every: int = 250
    eval_iters: int = 50
    log_every: int = 10
    dtype: str = "auto"            # auto | bf16 | fp16 | fp32
    compile: bool = False
    seed: int = 1337
    resume: bool = False


def load_config(path: str | None, overrides: list[str]) -> TrainConfig:
    cfg = TrainConfig()
    if path:
        with open(path) as f:
            for k, v in json.load(f).items():
                if k.startswith("_"):      # comments
                    continue
                if not hasattr(cfg, k):
                    raise KeyError(f"unknown config key: {k}")
                setattr(cfg, k, v)
    types = {f.name: f.type for f in fields(TrainConfig)}
    for item in overrides:
        k, v = item.split("=", 1)
        t = types[k]
        setattr(cfg, k, v.lower() in ("1", "true", "yes") if t in (bool, "bool") else
                (int(v) if t in (int, "int") else float(v) if t in (float, "float") else v))
    return cfg


# ---------------------------------------------------------------- data
def open_tokens(data_dir: str, split: str) -> np.memmap:
    with open(os.path.join(data_dir, "meta.json")) as f:
        dtype = json.load(f)["dtype"]
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=dtype, mode="r")


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str,
              generator: torch.Generator | None = None):
    block_size = min(block_size, len(data) - 2)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator).tolist()
    x = torch.from_numpy(np.stack([data[i:i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in ix]))
    if device == "cuda":   # pinned memory -> asynchronous copy to the GPU
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- optimization helpers
def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to cfg.lr, then cosine decay to cfg.min_lr at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def make_optimizer(model: GPT, cfg: TrainConfig, device: str) -> torch.optim.AdamW:
    """Weight decay on weight matrices and embeddings only, never on norms."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=device == "cuda")


def autocast_dtype(cfg: TrainConfig, device: str):
    if device != "cuda" or cfg.dtype == "fp32":
        return None
    if cfg.dtype == "fp16" or (cfg.dtype == "auto" and not torch.cuda.is_bf16_supported()):
        return torch.float16
    return torch.bfloat16


@torch.no_grad()
def evaluate(model, data, cfg: TrainConfig, device: str, ctx) -> float:
    g = torch.Generator().manual_seed(0)   # same batches every time
    model.eval()
    total = 0.0
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size, device, g)
        with ctx():
            total += model(x, y)[1].item()
    model.train()
    return total / cfg.eval_iters


# ---------------------------------------------------------------- training
def train(cfg: TrainConfig, device: str | None = None) -> dict:
    device = device or get_device()
    torch.manual_seed(cfg.seed)
    train_data, val_data = open_tokens(cfg.data_dir, "train"), open_tokens(cfg.data_dir, "val")
    with open(os.path.join(cfg.data_dir, "meta.json")) as f:
        vocab_size = json.load(f)["vocab_size"]

    os.makedirs(cfg.out_dir, exist_ok=True)
    shutil.copy(cfg.tokenizer, os.path.join(cfg.out_dir, "tokenizer.json"))
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")

    start_step, best_val = 0, float("inf")
    if cfg.resume and os.path.exists(ckpt_path):
        model, state = load_checkpoint(ckpt_path, device)
        optimizer = make_optimizer(model, cfg, device)
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_val = state["step"], state["best_val"]   # that step's update was not done yet
        print(f"resumed from step {state['step']}")
    else:
        model = GPT(GPTConfig(vocab_size=vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
                              n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout)).to(device)
        optimizer = make_optimizer(model, cfg, device)

    amp = autocast_dtype(cfg, device)
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp else contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    fwd = torch.compile(model) if cfg.compile else model

    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.block_size
    print(f"device {device} | params {model.num_params() / 1e6:.2f}M | precision {amp or torch.float32} | "
          f"{tokens_per_step:,} tokens/step | train {len(train_data):,} tokens")

    history = {"step": [], "train": [], "val": []}
    t0 = time.time()
    for step in range(start_step, cfg.max_steps + 1):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            tr = evaluate(fwd, train_data, cfg, device, ctx)
            va = evaluate(fwd, val_data, cfg, device, ctx)
            history["step"].append(step); history["train"].append(tr); history["val"].append(va)
            print(f"eval step {step:>6} | train {tr:.4f} | val {va:.4f}")
            if va < best_val or step == cfg.max_steps:
                best_val = min(best_val, va)
                save_checkpoint(ckpt_path, model, optimizer, step=step, best_val=best_val, train_cfg=asdict(cfg))
        if step == cfg.max_steps:
            break

        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, device)
            with ctx():
                _, loss = fwd(x, y)
            scaler.scale(loss / cfg.grad_accum).backward()   # gradients add up over micro-batches
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            n = cfg.log_every if step > start_step else 1
            print(f"step {step:>6} | loss {loss.item():.4f} | lr {lr:.2e} | grad norm {norm:.2f} | "
                  f"{n * tokens_per_step / max(dt, 1e-9):,.0f} tok/s")

    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device")
    args = ap.parse_args()
    train(load_config(args.config, args.set), args.device)


if __name__ == "__main__":
    main()
