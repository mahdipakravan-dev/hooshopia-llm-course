"""Tokenize the clean corpus once and store it on disk as a flat array of token ids.

    python -m llm.tokenize_data --data data/clean --tokenizer tokenizer.json --out data/tokens

Training then reads random windows straight from disk (numpy memmap), so the corpus can be far
larger than RAM. Documents are separated by <|endoftext|>.
"""
import argparse
import json
import os

import numpy as np

from .data import iter_split
from .tokenizer import Tokenizer


def dtype_for(vocab_size: int):
    return np.uint16 if vocab_size <= 2**16 else np.uint32


def tokenize_split(tok: Tokenizer, data_dir: str, split: str, out_path: str, flush_every: int = 1 << 20) -> int:
    dtype = dtype_for(tok.vocab_size)
    total, buf = 0, []
    with open(out_path, "wb") as f:
        for doc in iter_split(data_dir, split):
            buf.extend(tok.encode_ordinary(doc))
            buf.append(tok.eot_id)
            if len(buf) >= flush_every:
                np.asarray(buf, dtype=dtype).tofile(f)
                total += len(buf)
                buf = []
        np.asarray(buf, dtype=dtype).tofile(f)
        total += len(buf)
    return total


def tokenize_all(tokenizer_path: str, data_dir: str, out_dir: str) -> dict:
    tok = Tokenizer.load(tokenizer_path)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"vocab_size": tok.vocab_size, "dtype": np.dtype(dtype_for(tok.vocab_size)).name,
            "eot_id": tok.eot_id}
    for split in ("train", "val"):
        meta[f"{split}_tokens"] = tokenize_split(tok, data_dir, split, os.path.join(out_dir, f"{split}.bin"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out", default="data/tokens")
    args = ap.parse_args()
    meta = tokenize_all(args.tokenizer, args.data, args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
