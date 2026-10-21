import json
import os

from llm.data import Stats, clean_documents, iter_split, read_documents, write_shards
from llm.normalize import ZWNJ, normalize, persian_ratio

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "raw")


def test_arabic_letters_become_persian():
    assert normalize("پايتون يكي") == "پایتون یکی"


def test_digits_diacritics_tatweel():
    assert normalize("١٢٣") == "۱۲۳"
    assert normalize("بِسمِ") == "بسم"
    assert normalize("ســـلام") == "سلام"


def test_zwnj_and_spaces():
    assert normalize(f"خیلی   خوب {ZWNJ} {ZWNJ} بود {ZWNJ}") == "خیلی خوب بود"
    assert normalize(f"می{ZWNJ}{ZWNJ}خواهم") == f"می{ZWNJ}خواهم"  # real ZWNJ inside a word survives


def test_persian_ratio():
    assert persian_ratio("سلام دنیا") == 1.0
    assert persian_ratio("hello") == 0.0
    assert persian_ratio("۱۲۳ !") == 0.0


def test_read_documents_txt_and_jsonl():
    docs = list(read_documents(SAMPLE))
    assert len(docs) == 34  # 24 paragraphs in articles.txt + 10 lines in notes.jsonl


def test_clean_documents_filters_and_dedups():
    stats = Stats()
    kept = list(clean_documents(read_documents(SAMPLE), stats))
    assert stats.seen == 34
    assert stats.duplicate == 2      # one repeated paragraph in each file
    assert stats.not_persian == 1    # the English paragraph
    assert stats.too_short == 2      # "خیلی خوب بود" and "کوتاه"
    assert stats.kept == len(kept) == 29
    assert all("ي" not in d and "ك" not in d for d in kept)


def test_write_shards_roundtrip(tmp_path):
    docs = [f"سند شمارهٔ {i}" for i in range(25)]
    counts = write_shards(docs, str(tmp_path), shard_docs=10, val_every=10)
    assert counts == {"train": 22, "val": 3, "shards": 3}
    assert list(iter_split(str(tmp_path), "val")) == [docs[0], docs[10], docs[20]]
    assert len(list(iter_split(str(tmp_path), "train"))) == 22
    first = json.loads(open(tmp_path / "train_00000.jsonl", encoding="utf-8").readline())
    assert first["text"] == docs[1]
