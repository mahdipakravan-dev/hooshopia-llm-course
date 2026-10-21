"""Serve the model: a local HTTP API with streaming, plus a web chat page. No external API anywhere.

    python -m llm.server --ckpt checkpoints/sft/ckpt.pt          # then open http://localhost:8000

Endpoints
    GET  /            the chat page (llm/static/index.html)
    GET  /api/info    model size and context length
    POST /api/chat    {"messages": [...], "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 256}
                      -> text/event-stream:  data: {"text": "..."}  ...  data: [DONE]
"""
import argparse
import json
import os
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .chat_cli import reply

STATIC = os.path.join(os.path.dirname(__file__), "static")
app = FastAPI(title="Hooshopia LLM")
_state: dict = {}
_lock = threading.Lock()   # one generation at a time; the model is not shared between threads


def set_model(model, tok, name: str = "") -> None:
    _state.update(model=model, tok=tok, name=name)


def _get():
    if "model" not in _state:
        from .checkpoint import load_model
        from .env import get_device

        ckpt = os.environ.get("HOOSHOPIA_CKPT", "checkpoints/sft/ckpt.pt")
        set_model(*load_model(ckpt, get_device()), name=ckpt)
    return _state["model"], _state["tok"]


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.9, gt=0, le=1)
    max_new_tokens: int = Field(256, ge=1, le=2048)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/info")
def info():
    model, tok = _get()
    return {"name": _state.get("name", ""), "params": model.num_params(),
            "context": model.cfg.block_size, "vocab": tok.vocab_size}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.messages[-1].role != "user":
        raise HTTPException(400, "the last message must be from the user")
    model, tok = _get()
    messages = [m.model_dump() for m in req.messages]

    def events():
        with _lock:
            for piece in reply(model, tok, messages, max_new_tokens=req.max_new_tokens,
                               temperature=req.temperature, top_k=None, top_p=req.top_p):
                yield f"data: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft/ckpt.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    os.environ["HOOSHOPIA_CKPT"] = args.ckpt
    _get()   # load before the first request
    print(f"open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
