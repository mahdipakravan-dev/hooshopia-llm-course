"""Chat with the model in the terminal.

    python -m llm.chat_cli --ckpt checkpoints/sft/ckpt.pt
"""
import argparse

from .chat import StreamDecoder, render
from .checkpoint import load_model
from .env import get_device
from .sample import stream


def reply(model, tok, messages, **kw):
    """Yield the assistant's answer piece by piece."""
    ids, _ = render(tok, messages, add_generation_prompt=True)
    dec = StreamDecoder(tok)
    stop = set(tok.special.values())   # <|end|> normally; any special token means the answer is over
    for t in stream(model, ids, stop_ids=stop, **kw):
        if t in stop:
            break
        piece = dec.feed(t)
        if piece:
            yield piece
    tail = dec.flush()
    if tail:
        yield tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    model, tok = load_model(args.ckpt, get_device())
    messages: list[dict] = []
    print("type a message; /reset starts a new chat, Ctrl+C exits")
    while True:
        try:
            text = input("\nشما: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/reset":
            messages = []
            continue
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        print("مدل: ", end="", flush=True)
        answer = ""
        for piece in reply(model, tok, messages, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, top_k=None, top_p=args.top_p):
            answer += piece
            print(piece, end="", flush=True)
        print()
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
