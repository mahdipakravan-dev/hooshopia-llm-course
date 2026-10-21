import os

import torch

from llm.chat import StreamDecoder, build_example, render
from llm.chat_cli import reply
from llm.checkpoint import load_model, save_checkpoint
from llm.model import GPT, GPTConfig
from llm.sft import SFTConfig, collate, load_conversations, masked_loss, train_sft

SFT_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sft.jsonl")
CONV = [{"role": "user", "content": "سلام"}, {"role": "assistant", "content": "سلام! در خدمتم."}]


def test_render_format_and_mask(tok):
    ids, mask = render(tok, CONV)
    assert tok.decode(ids) == "<|user|>سلام<|end|><|assistant|>سلام! در خدمتم.<|end|>"
    n_user = len(tok.encode_ordinary("سلام")) + 2           # <|user|> + text + <|end|>
    assert mask[:n_user + 1] == [0] * (n_user + 1)           # ... and <|assistant|> itself
    assert all(mask[n_user + 1:]) and ids[-1] == tok.special["<|end|>"]
    gen, _ = render(tok, CONV[:1], add_generation_prompt=True)
    assert gen[-1] == tok.special["<|assistant|>"]


def test_targets_only_on_answers(tok):
    x, y = build_example(tok, CONV, block_size=256)
    trained = y[y != -100].tolist()
    assert tok.decode(trained) == "سلام! در خدمتم.<|end|>"
    assert len(x) == len(y)


def test_collate_pads_with_ignore(tok):
    a = build_example(tok, CONV, 256)
    b = build_example(tok, CONV[:1] + [{"role": "assistant", "content": "خوبم"}], 256)
    X, Y = collate([a, b], tok.eot_id)
    assert X.shape == Y.shape == (2, max(len(a[0]), len(b[0])))
    assert (Y[1, len(b[1]):] == -100).all()


def test_stream_decoder_handles_split_characters(tok):
    ids = list("سلام".encode("utf-8"))                  # raw byte tokens: every letter is split in two
    dec = StreamDecoder(tok)
    pieces = [dec.feed(i) for i in ids]
    assert "".join(pieces) + dec.flush() == "سلام"
    assert pieces[0] == ""                                # half a letter is held back


def test_sft_learns_the_answers(tmp_path, tok):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=2, n_embd=64))
    base = tmp_path / "base"
    save_checkpoint(str(base / "ckpt.pt"), model)
    tok.save(str(base / "tokenizer.json"))

    examples = [build_example(tok, c, 128) for c in load_conversations(SFT_DATA)]
    before = masked_loss(model, examples, tok.eot_id, "cpu")
    cfg = SFTConfig(ckpt=str(base / "ckpt.pt"), data=SFT_DATA, out_dir=str(tmp_path / "sft"),
                    steps=120, batch_size=8, lr=3e-3, warmup_steps=10, eval_every=60)
    hist = train_sft(cfg, "cpu")
    assert hist["step"] == [0, 60, 120]

    tuned, tok2 = load_model(str(tmp_path / "sft" / "ckpt.pt"))
    after = masked_loss(tuned, examples, tok.eot_id, "cpu")
    assert after < before - 1.0

    answer = "".join(reply(tuned, tok2, CONV[:1], max_new_tokens=20, temperature=0))
    assert "<|" not in answer                              # stops at <|end|>, never prints special tokens
