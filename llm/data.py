"""Build a clean training corpus: read -> normalize -> filter -> deduplicate -> shard.

    # from local files (.txt: documents separated by a blank line, .jsonl: {"text": ...} per line)
    python -m llm.data --input data/sample/raw --out data/clean

    # from a Hugging Face dataset (needs `pip install datasets`), e.g. Persian Wikipedia
    python -m llm.data --hf wikimedia/wikipedia --hf-config 20231101.fa --limit 200000 --out data/clean

Output: data/clean/train_00000.jsonl, ..., data/clean/val.jsonl and data/clean/stats.json
"""
import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

from .normalize import normalize, persian_ratio

_BLANK_LINE = re.compile(r"\n\s*\n")


def read_documents(path: str) -> Iterator[str]:
    """Yield raw documents from a file or (recursively, in sorted order) a directory."""
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for name in sorted(files):
                yield from read_documents(os.path.join(root, name))
        return
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)["text"]
    elif path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            yield from (doc for doc in _BLANK_LINE.split(f.read()) if doc.strip())


def read_hf(name: str, config: str | None, split: str = "train", field: str = "text",
            limit: int | None = None) -> Iterator[str]:
    """Stream documents from a Hugging Face dataset without downloading all of it."""
    from datasets import load_dataset  # optional dependency

    ds = load_dataset(name, config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield row[field]


@dataclass
class Stats:
    seen: int = 0
    too_short: int = 0
    not_persian: int = 0
    duplicate: int = 0
    kept: int = 0
    chars: int = 0


def clean_documents(docs: Iterable[str], stats: Stats, min_chars: int = 40,
                    min_persian: float = 0.6) -> Iterator[str]:
    """Normalize every document and drop the ones that would only teach the model noise."""
    seen_hashes: set[bytes] = set()
    for raw in docs:
        stats.seen += 1
        doc = normalize(raw)
        if len(doc) < min_chars:
            stats.too_short += 1
            continue
        if persian_ratio(doc) < min_persian:
            stats.not_persian += 1
            continue
        h = hashlib.sha1(doc.encode("utf-8")).digest()  # exact duplicates, after normalization
        if h in seen_hashes:
            stats.duplicate += 1
            continue
        seen_hashes.add(h)
        stats.kept += 1
        stats.chars += len(doc)
        yield doc


def write_shards(docs: Iterable[str], out_dir: str, shard_docs: int = 50_000,
                 val_every: int = 100) -> dict:
    """Write documents to jsonl shards. Every `val_every`-th document goes to val.jsonl
    (the first one always does, so even a tiny corpus has a validation set)."""
    os.makedirs(out_dir, exist_ok=True)
    val = open(os.path.join(out_dir, "val.jsonl"), "w", encoding="utf-8")
    shard, shard_idx, in_shard, counts = None, 0, 0, {"train": 0, "val": 0, "shards": 0}
    try:
        for i, doc in enumerate(docs):
            line = json.dumps({"text": doc}, ensure_ascii=False) + "\n"
            if i % val_every == 0:
                val.write(line)
                counts["val"] += 1
                continue
            if shard is None or in_shard >= shard_docs:
                if shard:
                    shard.close()
                shard = open(os.path.join(out_dir, f"train_{shard_idx:05d}.jsonl"), "w", encoding="utf-8")
                shard_idx, in_shard = shard_idx + 1, 0
                counts["shards"] += 1
            shard.write(line)
            in_shard += 1
            counts["train"] += 1
    finally:
        val.close()
        if shard:
            shard.close()
    return counts


def iter_split(data_dir: str, split: str) -> Iterator[str]:
    """Read back the cleaned corpus: split is "train" or "val"."""
    names = sorted(n for n in os.listdir(data_dir) if n.startswith(split) and n.endswith(".jsonl"))
    for name in names:
        yield from read_documents(os.path.join(data_dir, name))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="file or directory of .txt / .jsonl files")
    src.add_argument("--hf", help="Hugging Face dataset name")
    ap.add_argument("--hf-config")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--hf-field", default="text")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--min-persian", type=float, default=0.6)
    ap.add_argument("--shard-docs", type=int, default=50_000)
    ap.add_argument("--val-every", type=int, default=100)
    args = ap.parse_args()

    docs = read_documents(args.input) if args.input else \
        read_hf(args.hf, args.hf_config, args.hf_split, args.hf_field, args.limit)
    stats = Stats()
    counts = write_shards(clean_documents(docs, stats, args.min_chars, args.min_persian),
                          args.out, args.shard_docs, args.val_every)
    report = {**asdict(stats), **counts}
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for k, v in report.items():
        print(f"{k:>12}  {v:,}")


if __name__ == "__main__":
    main()
