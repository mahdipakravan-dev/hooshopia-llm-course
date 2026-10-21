"""Train the bigram baseline: our first training loop.

    python -m llm.train_bigram --data data/clean --tokenizer tokenizer.json

The loop is the same one every model in this course uses:
    batch -> forward -> loss -> backward -> optimizer step
"""
import argparse
import math
import time

import torch

from .batches import encode_corpus, estimate_loss, get_batch
from .bigram import BigramLM
from .data import iter_split
from .env import get_device
from .tokenizer import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    tok = Tokenizer.load(args.tokenizer)
    train = encode_corpus(tok, iter_split(args.data, "train"))
    val = encode_corpus(tok, iter_split(args.data, "val"))
    print(f"device {device} | vocab {tok.vocab_size} | train {len(train):,} tokens | val {len(val):,} tokens")

    model = BigramLM(tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"random guessing would give loss ln({tok.vocab_size}) = {math.log(tok.vocab_size):.3f}")

    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            tr = estimate_loss(model, train, args.batch_size, args.block_size, device=device)
            va = estimate_loss(model, val, args.batch_size, min(args.block_size, len(val) - 2), device=device)
            print(f"step {step:>5} | train {tr:.3f} | val {va:.3f} | {time.time() - t0:.1f}s")
        if step == args.steps:
            break
        x, y = get_batch(train, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    start = torch.tensor([[tok.eot_id]], device=device)
    print("\nsample:\n" + tok.decode(model.generate(start, 80)[0, 1:].tolist()))


if __name__ == "__main__":
    main()
