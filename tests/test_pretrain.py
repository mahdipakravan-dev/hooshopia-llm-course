import json
import os

import numpy as np
import pytest
import torch

from llm.checkpoint import load_model
from llm.data import write_shards
from llm.pretrain import TrainConfig, get_batch, load_config, lr_at, open_tokens, train
from llm.tokenize_data import tokenize_all


@pytest.fixture(scope="module")
def token_dir(tmp_path_factory, tok, docs):
    root = tmp_path_factory.mktemp("pre")
    write_shards(docs, str(root / "clean"), val_every=6)
    tok.save(str(root / "tokenizer.json"))
    meta = tokenize_all(str(root / "tokenizer.json"), str(root / "clean"), str(root / "tokens"))
    return root, meta


def tiny_cfg(root, **kw):
    base = dict(data_dir=str(root / "tokens"), tokenizer=str(root / "tokenizer.json"),
                out_dir=str(root / "ckpt"), block_size=32, n_layer=2, n_head=2, n_embd=32,
                batch_size=8, max_steps=60, lr=3e-3, min_lr=3e-4, warmup_steps=5,
                eval_every=30, eval_iters=4, log_every=20)
    return TrainConfig(**{**base, **kw})


def test_tokenized_files(token_dir, tok):
    root, meta = token_dir
    assert meta["dtype"] == "uint16" and meta["vocab_size"] == tok.vocab_size
    train = open_tokens(str(root / "tokens"), "train")
    assert len(train) == meta["train_tokens"]
    assert int(train[-1]) == tok.eot_id
    first_doc_end = int(np.argmax(np.asarray(train) == tok.eot_id))
    assert tok.decode(train[:first_doc_end].tolist()) in [d for d in open_docs(root)]


def open_docs(root):
    from llm.data import iter_split
    return list(iter_split(str(root / "clean"), "train"))


def test_memmap_batch_is_shifted(token_dir):
    root, _ = token_dir
    data = open_tokens(str(root / "tokens"), "train")
    x, y = get_batch(data, 4, 16, "cpu", torch.Generator().manual_seed(0))
    assert x.dtype == torch.long and x.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_lr_schedule():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(60, cfg) == pytest.approx(0.55)      # halfway through the cosine
    assert lr_at(110, cfg) == pytest.approx(0.1)
    assert all(lr_at(s, cfg) >= lr_at(s + 1, cfg) for s in range(10, 110))


def test_config_overrides(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"_note": "x", "n_layer": 3}))
    cfg = load_config(str(p), ["lr=0.01", "compile=true", "batch_size=4"])
    assert (cfg.n_layer, cfg.lr, cfg.compile, cfg.batch_size) == (3, 0.01, True, 4)
    p.write_text(json.dumps({"n_layerz": 3}))
    with pytest.raises(KeyError):
        load_config(str(p), [])


def test_training_reduces_loss_and_saves(token_dir):
    root, _ = token_dir
    cfg = tiny_cfg(root, out_dir=str(root / "run1"))
    hist = train(cfg, "cpu")
    assert hist["step"] == [0, 30, 60]
    assert hist["train"][-1] < hist["train"][0] - 0.5
    ckpt = os.path.join(cfg.out_dir, "ckpt.pt")
    model, tok = load_model(ckpt)
    assert model.cfg.n_embd == 32 and tok.vocab_size == model.cfg.vocab_size
    assert not model.training


def test_grad_accum_and_resume(token_dir):
    root, _ = token_dir
    cfg = tiny_cfg(root, out_dir=str(root / "run2"), max_steps=20, eval_every=10, grad_accum=2)
    train(cfg, "cpu")
    state = torch.load(os.path.join(cfg.out_dir, "ckpt.pt"), weights_only=False)
    assert state["step"] == 20
    cfg2 = tiny_cfg(root, out_dir=str(root / "run2"), max_steps=40, eval_every=10, grad_accum=2, resume=True)
    hist = train(cfg2, "cpu")
    assert hist["step"][0] == 20 and hist["step"][-1] == 40
