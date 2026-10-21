"""Chat format: how a conversation becomes one sequence of tokens.

    <|user|>سلام<|end|><|assistant|>سلام! در خدمتم.<|end|><|user|>...

The model never sees "roles"; it sees these special tokens. After SFT it has learned that after
<|assistant|> comes a helpful answer that finishes with <|end|>.
"""
import codecs

import torch

ROLE_TOKENS = {"user": "<|user|>", "assistant": "<|assistant|>"}


def render(tok, messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
    """Token ids of a conversation, and a mask: 1 where the token is part of an assistant answer
    (the ones we train on), 0 elsewhere. With add_generation_prompt the sequence ends with
    <|assistant|>, ready for the model to write the answer."""
    end = tok.special["<|end|>"]
    ids: list[int] = []
    mask: list[int] = []
    for m in messages:
        if m["role"] not in ROLE_TOKENS:
            raise ValueError(f"unknown role: {m['role']}")
        content = tok.encode_ordinary(m["content"])
        train = int(m["role"] == "assistant")
        ids += [tok.special[ROLE_TOKENS[m["role"]]], *content, end]
        mask += [0] + [train] * (len(content) + 1)      # the answer and its <|end|>
    if add_generation_prompt:
        ids.append(tok.special["<|assistant|>"])
        mask.append(0)
    return ids, mask


def build_example(tok, messages: list[dict], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and targets for SFT. Targets that are not part of an answer are -100 (ignored by the loss),
    so the model learns to write answers, not to imitate the user."""
    ids, mask = render(tok, messages)
    ids, mask = ids[:block_size + 1], mask[:block_size + 1]
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor([t if m else -100 for t, m in zip(ids[1:], mask[1:])], dtype=torch.long)
    return x, y


class StreamDecoder:
    """Turn a stream of token ids into text pieces. A Persian letter is 2 bytes in UTF-8 and a token
    can end in the middle of one, so bytes are held back until they form complete characters."""

    def __init__(self, tok):
        self.tok = tok
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, token_id: int) -> str:
        return self._dec.decode(self.tok.decode_bytes([token_id]))

    def flush(self) -> str:
        return self._dec.decode(b"", final=True)
