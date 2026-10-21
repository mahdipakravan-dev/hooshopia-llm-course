import os

import pytest

from llm.data import Stats, clean_documents, read_documents
from llm.tokenizer import Tokenizer

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "raw")


@pytest.fixture(scope="session")
def docs():
    return list(clean_documents(read_documents(SAMPLE), Stats()))


@pytest.fixture(scope="session")
def tok(docs):
    return Tokenizer.train(docs, vocab_size=512)
