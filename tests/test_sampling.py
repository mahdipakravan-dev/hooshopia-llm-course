import torch

from llm.model import GPT, GPTConfig, KVCache
from llm.sample import filter_logits, pick, stream


def make(block=48):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=101, block_size=block, n_layer=2, n_head=4, n_embd=64)).eval()


def test_cached_logits_match_full_forward():
    model = make()
    idx = torch.randint(101, (1, 30))
    full, _ = model(idx)
    cache = KVCache(model.cfg.n_layer)
    part, _ = model(idx[:, :20], cache=cache)          # prefill 20 tokens
    assert torch.allclose(part, full[:, :20], atol=1e-5)
    steps = [model(idx[:, t:t + 1], cache=cache)[0] for t in range(20, 30)]   # then one at a time
    assert torch.allclose(torch.cat(steps, dim=1), full[:, 20:], atol=1e-4)
    assert cache.seq_len == 30


def test_cache_with_several_new_tokens():
    model = make()
    idx = torch.randint(101, (1, 25))
    full, _ = model(idx)
    cache = KVCache(model.cfg.n_layer)
    model(idx[:, :10], cache=cache)
    chunk, _ = model(idx[:, 10:25], cache=cache)        # 15 new tokens at once after a cached prefix
    assert torch.allclose(chunk, full[:, 10:], atol=1e-4)


def test_greedy_same_with_and_without_cache():
    model = make()
    prompt = torch.randint(101, (12,)).tolist()
    a = list(stream(model, prompt, 30, temperature=0, use_cache=True))
    b = list(stream(model, prompt, 30, temperature=0, use_cache=False))
    assert a == b and len(a) == 30


def test_generation_runs_past_block_size():
    model = make(block=16)
    out = list(stream(model, [1, 2, 3], 50, temperature=1.0, seed=0))
    assert len(out) == 50


def test_stop_token():
    model = make()
    first = next(stream(model, [5, 6, 7], 10, temperature=0))
    out = list(stream(model, [5, 6, 7], 10, temperature=0, stop_ids={first}))
    assert out == [first]


def test_top_k_and_top_p():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    kept = torch.isfinite(filter_logits(logits, top_k=2))
    assert kept.tolist() == [[True, True, False, False, False]]
    kept = torch.isfinite(filter_logits(logits, top_p=0.01))
    assert kept.tolist() == [[True, False, False, False, False]]
    kept = torch.isfinite(filter_logits(logits, top_p=0.9))  # probs ≈ .64 .24 .09 .03 .01
    assert kept.tolist() == [[True, True, True, False, False]]
    g = torch.Generator().manual_seed(0)
    picks = {pick(logits, 1.0, top_k=2, generator=g).item() for _ in range(200)}
    assert picks == {0, 1}
    assert pick(logits, temperature=0).item() == 0
