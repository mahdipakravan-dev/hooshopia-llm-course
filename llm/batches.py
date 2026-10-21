"""Turning a token stream into training batches, and measuring loss."""
from typing import Iterable

import torch


def encode_corpus(tok, docs: Iterable[str]) -> torch.Tensor:
    """All documents as one long stream of token ids, with <|endoftext|> after each document."""
    ids: list[int] = []
    for doc in docs:
        ids.extend(tok.encode_ordinary(doc))
        ids.append(tok.eot_id)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str = "cpu",
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows of `block_size` tokens. y is x shifted by one: at every position
    the model must predict the token that comes next."""
    assert len(data) > block_size + 1, "not enough tokens for one block"
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix.tolist()])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix.tolist()])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, batch_size: int, block_size: int, iters: int = 20,
                  device: str = "cpu", seed: int = 0) -> float:
    """Average loss over a few random batches (same batches every call, so numbers are comparable)."""
    g = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    losses = [model(*get_batch(data, batch_size, block_size, device, g))[1].item() for _ in range(iters)]
    model.train(was_training)
    return sum(losses) / len(losses)
