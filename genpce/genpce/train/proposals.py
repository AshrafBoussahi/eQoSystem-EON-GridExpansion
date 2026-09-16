"""Learned proposal distributions for the evolutionary engine (estimation-of-distribution ladder).

Both proposals are updated online from the search's elite archive and sample new offspring:

* :class:`PBILProposal` — independent per-position token distributions (population-based
  incremental learning / UMDA). The classic EDA: cheap, no interactions between slots, and the
  smoothing (learning rate + probability floor) prevents collapse.
* :class:`TransformerProposal` — an autoregressive GPT fine-tuned for a few steps per generation on
  the elite archive (dropout, weight decay, small learning rate) and sampled at temperature ``1/β``.
  Captures interactions between slots; its value over PBIL is the empirical question.

Together with plain mutation (no learned proposal) this gives a three-rung ladder that isolates
what, if anything, a generative model contributes to per-instance search.
"""

from __future__ import annotations

import numpy as np
import torch

from genpce.model.gpt import GPT, GPTConfig, sample_sequences, sequence_log_probs

__all__ = ["PBILProposal", "TransformerProposal"]


class PBILProposal:
    """Per-position categorical model ``p[t, v]`` updated toward elite token frequencies."""

    def __init__(self, seq_len: int, vocab_size: int, *, lr: float = 0.1, floor: float = 0.01, seed: int = 0):
        self.seq_len, self.vocab_size, self.lr, self.floor = seq_len, vocab_size, lr, floor
        self.p = np.full((seq_len, vocab_size), 1.0 / vocab_size)
        self.rng = np.random.default_rng(seed)

    def update(self, elites: np.ndarray) -> None:
        elites = np.asarray(elites, dtype=np.int64)
        freq = np.zeros_like(self.p)
        for t in range(self.seq_len):
            freq[t] = np.bincount(elites[:, t], minlength=self.vocab_size) / len(elites)
        self.p = (1.0 - self.lr) * self.p + self.lr * freq
        self.p = np.maximum(self.p, self.floor)
        self.p /= self.p.sum(axis=1, keepdims=True)

    def __call__(self, k: int) -> np.ndarray:
        cum = np.cumsum(self.p, axis=1)
        u = self.rng.random((k, self.seq_len, 1))
        return (u > cum[None]).sum(axis=2).clip(0, self.vocab_size - 1)

    def entropy(self) -> float:
        return float(-(self.p * np.log(self.p)).sum(axis=1).mean())


class TransformerProposal:
    """GPT proposal fine-tuned online on the elite archive (a few MLE steps per generation)."""

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        *,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        steps_per_update: int = 4,
        batch_size: int = 64,
        beta: float = 1.0,
        seed: int = 0,
        model: GPT | None = None,
    ):
        torch.manual_seed(seed)
        self.seq_len, self.vocab_size, self.beta = seq_len, vocab_size, beta
        self.model = model or GPT(GPTConfig(vocab_size=vocab_size, max_len=seq_len, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.steps, self.batch_size = steps_per_update, batch_size
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.last_loss = float("nan")

    def update(self, elites: np.ndarray) -> None:
        elites = torch.as_tensor(np.asarray(elites, dtype=np.int64))
        self.model.train()
        for _ in range(self.steps):
            idx = torch.as_tensor(self.rng.integers(0, len(elites), size=min(self.batch_size, len(elites))))
            logp, _ = sequence_log_probs(self.model, elites[idx], beta=1.0)
            loss = -logp.mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
        self.last_loss = float(loss.detach())
        self.model.eval()

    def __call__(self, k: int) -> np.ndarray:
        toks, _ = sample_sequences(self.model, k, self.seq_len, self.beta, generator=self.gen)
        return toks.cpu().numpy()
