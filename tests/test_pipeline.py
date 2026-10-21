"""End to end: every command from the docs, in order, on the sample data with tiny settings."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args, stdin=None):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, input=stdin, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, f"{' '.join(args)}\n{r.stdout}\n{r.stderr}"
    return r.stdout


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return str(tmp_path_factory.mktemp("pipeline"))


def test_full_pipeline(work):
    clean, tok, tokens = f"{work}/clean", f"{work}/tokenizer.json", f"{work}/tokens"
    assert "device" in run("llm.env")
    assert "kept" in run("llm.data", "--input", "data/sample/raw", "--out", clean)
    assert "saved" in run("llm.tokenizer", "train", "--data", clean, "--vocab-size", "600", "--out", tok)
    assert "tokens" in run("llm.tokenizer", "encode", "--tokenizer", tok, "سلام دنیا")
    assert "chars/token" in run("llm.tokenizer", "stats", "--tokenizer", tok, "--data", clean)
    assert "sample" in run("llm.train_bigram", "--data", clean, "--tokenizer", tok, "--steps", "20", "--eval-every", "10")
    assert "gpt" in run("llm.model", "--data", clean, "--tokenizer", tok, "--steps", "20")
    assert "train_tokens" in run("llm.tokenize_data", "--data", clean, "--tokenizer", tok, "--out", tokens)
    run("llm.pretrain", "--config", "configs/tiny.json", "--device", "cpu", "--set", f"data_dir={tokens}",
        f"tokenizer={tok}", f"out_dir={work}/pre", "max_steps=20", "eval_every=10", "log_every=5")
    ckpt = f"{work}/pre/ckpt.pt"
    assert os.path.exists(ckpt)
    assert "temperature 0" in run("llm.sample", "--ckpt", ckpt, "--prompt", "سلام", "--max-new-tokens", "10",
                                  "--temperatures", "0", "1.0")
    assert "tokens/s" in run("llm.sample", "--ckpt", ckpt, "--max-new-tokens", "10", "--bench")
    run("llm.sft", "--ckpt", ckpt, "--data", "data/sample/sft.jsonl", "--out", f"{work}/sft",
        "--steps", "10", "--eval-every", "5", "--batch-size", "4", "--device", "cpu")
    sft = f"{work}/sft/ckpt.pt"
    assert "report" in run("llm.evaluate", "--ckpt", ckpt, sft, "--tokens", tokens,
                           "--out", f"{work}/eval.md", "--max-new-tokens", "5")
    assert open(f"{work}/eval.md", encoding="utf-8").read().startswith("# Evaluation report")
    assert "مدل:" in run("llm.chat_cli", "--ckpt", sft, "--max-new-tokens", "5", stdin="سلام\n")
