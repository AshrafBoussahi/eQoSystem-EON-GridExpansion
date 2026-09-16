"""A small decoder-only transformer ("GPT") over operator-pool tokens.

Follows the GQE construction (Nakaji et al., Sec. 2.2.1): the network maps a prefix
``{BOS, j_1, ..., j_{k-1}}`` to a vector of *energies* ``w^(k) ∈ R^L`` (one per pool token) and the
next token is sampled with probability ``∝ exp(-β w_j^(k))``. The sequence probability is therefore
``p_N(β, j) ∝ exp(-β w_sum(j))`` with ``w_sum = Σ_k w^(k)_{j_k}``.

Self-contained (no HuggingFace dependency). Autoregressive sampling uses a key/value cache so that
generating a sequence of length ``N`` costs ``O(N)`` block evaluations instead of ``O(N²)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GPTConfig", "GPT", "sample_sequences", "sequence_log_probs"]


@dataclass
class GPTConfig:
    vocab_size: int
    max_len: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0


class _CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # With a cache the new queries attend to every stored key (all of them are in the past),
        # so the causal mask is only needed for the (cache-free) full-sequence pass.
        causal = cache is None or k.shape[2] == T
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, d))


class _Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = _CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Decoder-only transformer returning per-position token energies ``w``."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.bos = cfg.vocab_size  # extra id used only as the start token
        self.tok_emb = nn.Embedding(cfg.vocab_size + 1, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len + 1, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def new_cache(self) -> list[dict]:
        return [dict() for _ in self.blocks]

    def forward(self, idx: torch.Tensor, cache: list[dict] | None = None, pos_offset: int = 0) -> torch.Tensor:
        """``idx``: ``(B, T)`` token ids. Returns energies ``(B, T, L)``.

        With ``cache`` (from :meth:`new_cache`), ``idx`` holds only the *new* tokens and
        ``pos_offset`` their first position; keys/values of earlier tokens are read from the cache.
        """
        T = idx.shape[1]
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for i, block in enumerate(self.blocks):
            x = block(x, None if cache is None else cache[i])
        return self.head(self.ln_f(x))


@torch.no_grad()
def sample_sequences(
    model: GPT, num: int, length: int, beta: float, *, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively sample ``num`` token sequences of ``length`` at inverse temperature ``β``.

    Returns ``(tokens (num, length), log_probs (num, length))`` where ``log_probs`` are the
    per-token log-probabilities under the sampling policy ``softmax(-β w)``.
    """
    model.eval()
    device = next(model.parameters()).device
    cache = model.new_cache()
    current = torch.full((num, 1), model.bos, dtype=torch.long, device=device)
    tokens = torch.empty((num, length), dtype=torch.long, device=device)
    logps = torch.empty((num, length), device=device)
    for k in range(length):
        w = model(current, cache=cache, pos_offset=k)[:, -1, :]
        logp = F.log_softmax(-beta * w, dim=-1)
        j = torch.multinomial(logp.exp(), 1, generator=generator)
        tokens[:, k] = j.squeeze(1)
        logps[:, k] = logp.gather(1, j).squeeze(1)
        current = j
    return tokens, logps


def sequence_log_probs(model: GPT, tokens: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities of ``tokens`` ``(B, N)`` under ``softmax(-β w)`` and the energies.

    Returns ``(log_probs (B, N), w_selected (B, N))``; the latter gives ``w_sum`` by summation,
    which the logit-matching loss compares with the measured cost.
    """
    B, N = tokens.shape
    bos = torch.full((B, 1), model.bos, dtype=torch.long, device=tokens.device)
    inp = torch.cat([bos, tokens[:, :-1]], dim=1)
    w = model(inp)  # (B, N, L)
    logp = F.log_softmax(-beta * w, dim=-1).gather(2, tokens[..., None]).squeeze(2)
    w_sel = w.gather(2, tokens[..., None]).squeeze(2)
    return logp, w_sel
