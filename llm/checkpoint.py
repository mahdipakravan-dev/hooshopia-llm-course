"""Save and load a trained model together with its config and tokenizer."""
import os

import torch

from .model import GPT, GPTConfig
from .tokenizer import Tokenizer


def save_checkpoint(path: str, model: GPT, optimizer=None, **extra) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"model": model.state_dict(), "model_cfg": vars(model.cfg).copy(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)   # never leave a half-written checkpoint behind


def load_checkpoint(path: str, device: str = "cpu") -> tuple[GPT, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"])
    return model.to(device), state


def load_model(ckpt_path: str, device: str = "cpu") -> tuple[GPT, Tokenizer]:
    """Model in eval mode + the tokenizer saved next to the checkpoint."""
    model, _ = load_checkpoint(ckpt_path, device)
    tok = Tokenizer.load(os.path.join(os.path.dirname(ckpt_path), "tokenizer.json"))
    return model.eval(), tok
