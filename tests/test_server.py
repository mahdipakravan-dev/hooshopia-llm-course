import json

import torch
from fastapi.testclient import TestClient

from llm import server
from llm.model import GPT, GPTConfig


def client(tok):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=32)).eval()
    server.set_model(model, tok, "test")
    return TestClient(server.app)


def test_index_and_info(tok):
    c = client(tok)
    assert "هوشوپیا" in c.get("/").text
    info = c.get("/api/info").json()
    assert info["context"] == 64 and info["vocab"] == tok.vocab_size and info["params"] > 0


def test_chat_streams_events(tok):
    c = client(tok)
    body = {"messages": [{"role": "user", "content": "سلام"}], "temperature": 0, "max_new_tokens": 8}
    with c.stream("POST", "/api/chat", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [line[len("data: "):] for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    text = "".join(json.loads(e)["text"] for e in events[:-1])
    assert "<|" not in text


def test_chat_validation(tok):
    c = client(tok)
    assert c.post("/api/chat", json={"messages": []}).status_code == 422
    assert c.post("/api/chat", json={"messages": [{"role": "system", "content": "x"}]}).status_code == 422
    last_not_user = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
    assert c.post("/api/chat", json=last_not_user).status_code == 400
