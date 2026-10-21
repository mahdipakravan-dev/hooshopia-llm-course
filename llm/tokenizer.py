"""Byte-level BPE tokenizer, from scratch.

    python -m llm.tokenizer train  --data data/clean --vocab-size 8192 --out tokenizer.json
    python -m llm.tokenizer encode --tokenizer tokenizer.json "سلام دنیا"
    python -m llm.tokenizer stats  --tokenizer tokenizer.json --data data/clean

How it works
1. Pre-tokenize: split text into chunks (words with their leading space, numbers, punctuation,
   whitespace). Merges never cross a chunk boundary, so "سلام" and " سلام" stay separate pieces.
2. Start from the 256 byte values: any text, in any language, can always be encoded.
3. Repeatedly find the most frequent adjacent pair of tokens and merge it into a new token.
4. Encoding applies the learned merges in the order they were learned.
"""
import argparse
import heapq
import json
import re
from collections import Counter, defaultdict
from typing import Iterable

# A ZWNJ (نیم‌فاصله) is part of a Persian word, so "می‌خواهم" is one chunk.
PATTERN = r" ?(?:[^\W\d_]|‌)+| ?\d{1,3}| ?(?:[^\w\s‌]|_)+|\s+(?!\S)|\s+"

# Reserved from the start, so the chat format in part 7 doesn't need a new tokenizer.
SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"]


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None, special_tokens: list[str] | None = None,
                 pattern: str = PATTERN):
        self.pattern = pattern
        self._split = re.compile(pattern)
        self.merges = [tuple(m) for m in (merges or [])]          # in learning order
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}  # pair -> new token id
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.ranks.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        base = 256 + len(self.merges)
        self.special = {tok: base + i for i, tok in enumerate(special_tokens or SPECIAL_TOKENS)}
        self.special_inv = {i: tok for tok, i in self.special.items()}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in self.special) + ")")
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, special_tokens: list[str] | None = None,
              pattern: str = PATTERN, verbose: bool = False) -> "Tokenizer":
        special_tokens = special_tokens or SPECIAL_TOKENS
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges >= 0, "vocab_size must be at least 256 + number of special tokens"
        split = re.compile(pattern)

        chunk_counts = Counter()
        for text in texts:
            chunk_counts.update(split.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        # pair -> total frequency, and pair -> which words contain it
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]  # max-heap; ties -> smallest pair
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg, best = heapq.heappop(heap)
            if pair_counts.get(best, 0) != -neg:   # stale entry, the count changed since it was pushed
                continue
            if -neg < 2:                           # merging a pair seen once teaches nothing
                break
            new_id = 256 + len(merges)
            merges.append(best)
            touched = set()
            for wi in where.pop(best):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                w = _merge(w, best, new_id)
                words[wi] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    touched.add(p)
            pair_counts.pop(best, None)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                elif p in pair_counts:
                    del pair_counts[p]
            if verbose and len(merges) % 500 == 0:
                print(f"merge {len(merges):>6}/{n_merges}")
        return cls(merges, special_tokens, pattern)

    # ------------------------------------------------------------------ encode / decode
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    @property
    def eot_id(self) -> int:
        return self.special["<|endoftext|>"]

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, self.ranks[pair])
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text; special-token strings inside it are treated as normal text."""
        out: list[int] = []
        for chunk in self._split.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not allow_special:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if part in self.special:
                out.append(self.special[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.special_inv[i].encode("utf-8") if i in self.special_inv else self.vocab[i]
                        for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Human-readable form of one token (for showing how text is split)."""
        return self.special_inv.get(i) or self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ save / load
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "pattern": self.pattern, "merges": self.merges,
                       "special_tokens": list(self.special)}, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls([tuple(m) for m in d["merges"]], d["special_tokens"], d["pattern"])


def main() -> None:
    from .data import iter_split

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/clean")
    t.add_argument("--vocab-size", type=int, default=8192)
    t.add_argument("--max-docs", type=int, help="train on the first N documents only (faster)")
    t.add_argument("--out", default="tokenizer.json")
    e = sub.add_parser("encode")
    e.add_argument("--tokenizer", default="tokenizer.json")
    e.add_argument("text")
    s = sub.add_parser("stats")
    s.add_argument("--tokenizer", default="tokenizer.json")
    s.add_argument("--data", default="data/clean")
    args = ap.parse_args()

    if args.cmd == "train":
        docs = iter_split(args.data, "train")
        if args.max_docs:
            docs = (d for i, d in enumerate(docs) if i < args.max_docs)
        tok = Tokenizer.train(docs, args.vocab_size, verbose=True)
        tok.save(args.out)
        print(f"saved {args.out}: vocab {tok.vocab_size} ({len(tok.merges)} merges)")
    elif args.cmd == "encode":
        tok = Tokenizer.load(args.tokenizer)
        ids = tok.encode(args.text)
        print(ids)
        print(" | ".join(tok.token_str(i) for i in ids))
        print(f"{len(args.text)} chars -> {len(ids)} tokens")
    else:
        tok = Tokenizer.load(args.tokenizer)
        chars = tokens = 0
        for doc in iter_split(args.data, "val"):
            chars += len(doc)
            tokens += len(tok.encode_ordinary(doc))
        print(f"val: {chars:,} chars, {tokens:,} tokens, {chars / max(tokens, 1):.2f} chars/token")


if __name__ == "__main__":
    main()
