"""The simplest possible language model: the next token depends only on the current token.

It is a V x V table: row i holds the scores (logits) for every token that can follow token i.
Everything after this (attention, transformers) is about letting the model look further back.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.table(idx)                       # (B, T) -> (B, T, V)
        loss = None
        if targets is not None:
            # cross-entropy = -log(probability the model gave to the correct next token)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1:])              # only the last token matters for a bigram
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx
