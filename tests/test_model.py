import math

import torch

from llm.batches import encode_corpus, get_batch
from llm.model import GPT, GPTConfig, apply_rope, attention_manual, rope_cache

TINY = dict(block_size=32, n_layer=2, n_head=4, n_embd=64)


def make(vocab=97, **kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab, **{**TINY, **kw}))


def test_shapes_and_initial_loss():
    model = make()
    idx, targets = torch.randint(97, (3, 16)), torch.randint(97, (3, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 16, 97)
    assert abs(loss.item() - math.log(97)) < 0.3   # small init -> nearly uniform predictions


def test_weight_tying():
    model = make()
    assert model.lm_head.weight is model.tok_emb.weight


def test_manual_attention_matches_sdpa():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 10, 16) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(attention_manual(q, k, v), ref, atol=1e-5)

    model = make().eval()
    idx = torch.randint(97, (2, 20))
    fast, _ = model(idx)
    for b in model.blocks:
        b.attn.use_sdpa = False
    slow, _ = model(idx)
    assert torch.allclose(fast, slow, atol=1e-4)


def test_causal_mask_hides_the_future():
    model = make().eval()
    idx = torch.randint(97, (1, 20))
    changed = idx.clone()
    changed[0, 15] = (changed[0, 15] + 1) % 97
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[0, :15], b[0, :15], atol=1e-5)    # positions before 15 can't see it
    assert not torch.allclose(a[0, 15:], b[0, 15:], atol=1e-5)


def test_rope_depends_only_on_distance():
    cos, sin = rope_cache(16, 64)
    torch.manual_seed(0)
    q, k = torch.randn(16), torch.randn(16)
    rot = lambda x, t: apply_rope(x[None], cos[t:t + 1], sin[t:t + 1])[0]
    d1 = rot(q, 10) @ rot(k, 7)
    d2 = rot(q, 40) @ rot(k, 37)
    assert torch.allclose(d1, d2, atol=1e-4)


def test_overfits_one_batch():
    model = make()
    x = torch.randint(97, (4, 32))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1.0


def test_beats_bigram_on_sample(tok, docs):
    from llm.bigram import BigramLM

    data = encode_corpus(tok, docs)
    losses = {}
    for name, model, lr in [("bigram", BigramLM(tok.vocab_size), 3e-2),
                            ("gpt", make(tok.vocab_size), 3e-3)]:
        torch.manual_seed(1)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for _ in range(300):
            _, loss = model(*get_batch(data, 16, 32))
            opt.zero_grad()
            loss.backward()
            opt.step()
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            losses[name] = sum(model(*get_batch(data, 16, 32, generator=g))[1].item() for _ in range(10)) / 10
    assert losses["gpt"] < losses["bigram"]


def test_ignore_index():
    model = make()
    idx = torch.randint(97, (1, 8))
    targets = torch.full((1, 8), -100)
    targets[0, 3] = 5
    _, loss = model(idx, targets)
    logits, _ = model(idx)
    expected = torch.nn.functional.cross_entropy(logits[0, 3:4], torch.tensor([5]))
    assert torch.allclose(loss, expected, atol=1e-5)
