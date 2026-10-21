import math

import torch

from llm.batches import encode_corpus, estimate_loss, get_batch
from llm.bigram import BigramLM


def test_corpus_has_eot_between_docs(tok, docs):
    data = encode_corpus(tok, docs[:3])
    assert (data == tok.eot_id).sum().item() == 3
    assert data[-1].item() == tok.eot_id


def test_batch_targets_are_shifted_inputs():
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, block_size=8, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_untrained_loss_is_close_to_uniform(tok, docs):
    torch.manual_seed(0)
    data = encode_corpus(tok, docs)
    model = BigramLM(tok.vocab_size)
    loss = estimate_loss(model, data, batch_size=16, block_size=32)
    assert abs(loss - math.log(tok.vocab_size)) < 1.0


def test_training_lowers_loss(tok, docs):
    torch.manual_seed(0)
    data = encode_corpus(tok, docs)
    model = BigramLM(tok.vocab_size)
    before = estimate_loss(model, data, 16, 32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
    for _ in range(150):
        _, loss = model(*get_batch(data, 16, 32))
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = estimate_loss(model, data, 16, 32)
    assert after < before - 1.5


def test_generate_shape(tok):
    model = BigramLM(tok.vocab_size)
    out = model.generate(torch.zeros((2, 1), dtype=torch.long), max_new_tokens=5)
    assert out.shape == (2, 6)
