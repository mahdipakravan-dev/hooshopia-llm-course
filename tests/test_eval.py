import math
import os

import numpy as np
import torch

from llm.checkpoint import save_checkpoint
from llm.evaluate import evaluate, load_prompts, perplexity, to_markdown
from llm.model import GPT, GPTConfig

PROMPTS = os.path.join(os.path.dirname(__file__), "..", "evals", "prompts.jsonl")


def make(vocab, block=32):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=1, n_head=2, n_embd=32)).eval()


def test_perplexity_matches_manual_loss():
    model = make(50)
    data = np.random.default_rng(0).integers(0, 50, size=200).astype(np.uint16)
    loss, ppl = perplexity(model, data, max_tokens=None, batch_size=3)
    x = torch.from_numpy(data[:-1].astype(np.int64))
    y = torch.from_numpy(data[1:].astype(np.int64))
    manual = []
    for s in range(0, len(x), 32):
        manual.append(torch.nn.functional.cross_entropy(model(x[None, s:s + 32])[0][0], y[s:s + 32],
                                                        reduction="sum").item())
    assert math.isclose(loss, sum(manual) / len(y), rel_tol=1e-4)
    assert math.isclose(ppl, math.exp(loss))


def test_untrained_perplexity_is_about_vocab_size():
    model = make(64)
    data = np.random.default_rng(1).integers(0, 64, size=500).astype(np.uint16)
    _, ppl = perplexity(model, data)
    assert 40 < ppl < 100


def test_report(tmp_path, tok):
    model = make(tok.vocab_size)
    path = str(tmp_path / "m" / "ckpt.pt")
    save_checkpoint(path, model)
    tok.save(str(tmp_path / "m" / "tokenizer.json"))
    prompts = load_prompts(PROMPTS)[:2]
    results = evaluate([path], prompts, None, "cpu", max_new_tokens=5)
    assert len(results[path]["answers"]) == 2
    md = to_markdown(results, prompts)
    assert prompts[0] in md and "params" in md
