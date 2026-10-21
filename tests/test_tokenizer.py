import os
import re

from llm.data import Stats, clean_documents, read_documents
from llm.normalize import ZWNJ
from llm.tokenizer import PATTERN, SPECIAL_TOKENS, Tokenizer

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "raw")


def sample_docs():
    return list(clean_documents(read_documents(SAMPLE), Stats()))


def test_pretokenizer_covers_every_character():
    text = f"سلام، دنیا! می{ZWNJ}خواهم ۱۲۳۴۵ x_y  \n\tok"
    chunks = re.findall(PATTERN, text)
    assert "".join(chunks) == text
    assert f" می{ZWNJ}خواهم" in chunks  # ZWNJ stays inside the word


def test_untrained_tokenizer_is_plain_bytes():
    tok = Tokenizer()
    assert tok.vocab_size == 256 + len(SPECIAL_TOKENS)
    assert tok.encode("سلام") == list("سلام".encode("utf-8"))


def test_toy_merges():
    tok = Tokenizer.train(["aaab aaab aaab"], vocab_size=256 + len(SPECIAL_TOKENS) + 2)
    a, b = ord("a"), ord("b")
    assert tok.merges[0] == (a, a)                # the most frequent pair goes first
    # then ("aa","a") and ("a","b") both appear 3 times; ties go to the smaller pair
    assert tok.merges[1] == (a, b)
    assert tok.decode(tok.encode("aaab")) == "aaab"


def test_roundtrip_and_compression_on_sample():
    docs = sample_docs()
    tok = Tokenizer.train(docs, vocab_size=1024)
    for doc in docs:
        assert tok.decode(tok.encode(doc)) == doc
    text = "\n".join(docs)
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2.5  # far fewer tokens than bytes


def test_special_tokens(tmp_path):
    tok = Tokenizer.train(sample_docs(), vocab_size=600)
    ids = tok.encode("<|user|>سلام<|end|>")
    assert ids[0] == tok.special["<|user|>"] and ids[-1] == tok.special["<|end|>"]
    assert tok.decode(ids) == "<|user|>سلام<|end|>"
    assert tok.special["<|user|>"] not in tok.encode("<|user|>", allow_special=False)
    assert max(tok.special.values()) == tok.vocab_size - 1

    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = Tokenizer.load(path)
    assert tok2.merges == tok.merges and tok2.special == tok.special
    assert tok2.encode("یک متن تازه برای تست") == tok.encode("یک متن تازه برای تست")


def naive_train(texts, n_merges):
    """Slow reference: recount every pair from scratch before each merge."""
    from collections import Counter

    from llm.tokenizer import _merge

    words = Counter()
    for t in texts:
        words.update(re.findall(PATTERN, t))
    seqs = {w: list(w.encode("utf-8")) for w in words}
    merges = []
    for k in range(n_merges):
        counts = Counter()
        for w, s in seqs.items():
            for p in zip(s, s[1:]):
                counts[p] += words[w]
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        if counts[best] < 2:
            break
        merges.append(best)
        seqs = {w: _merge(s, best, 256 + k) for w, s in seqs.items()}
    return merges


def test_fast_training_matches_naive_reference():
    docs = sample_docs()
    n = 300
    assert Tokenizer.train(docs, 256 + len(SPECIAL_TOKENS) + n).merges == naive_train(docs, n)


def test_training_is_deterministic():
    docs = sample_docs()
    assert Tokenizer.train(docs, 500).merges == Tokenizer.train(docs, 500).merges
