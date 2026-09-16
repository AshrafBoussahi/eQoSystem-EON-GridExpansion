"""A transformer prior over circuits, learned from elites of many instances (amortisation).

The generator is trained by maximum likelihood on elite token sequences gathered across instances
(never on known solutions — elites are the best circuits *the search found*, ranked by their own
measured reward), with a held-out split to detect memorisation. It is then used for a new instance
as (i) a source of initial populations and/or (ii) a proposal operator inside
:class:`genpce.train.evolution.EvolutionarySearch`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from genpce.model.gpt import GPT, GPTConfig, sample_sequences, sequence_log_probs

__all__ = ["PriorConfig", "train_prior", "ModelProposal", "mean_token_entropy"]


@dataclass
class PriorConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 5e-4
    weight_decay: float = 0.05
    batch_size: int = 64
    max_steps: int = 2000
    patience: int = 200  # early stopping on validation NLL
    val_fraction: float = 0.1
    seed: int = 0


def train_prior(
    elites: np.ndarray, vocab_size: int, cfg: PriorConfig, *, model: GPT | None = None, val_elites: np.ndarray | None = None
) -> tuple[GPT, dict]:
    """Fit (or fine-tune) a GPT on elite sequences ``(K, N)`` with early stopping on held-out NLL.

    Pass ``val_elites`` from an instance *not* represented in ``elites`` to measure generalisation
    across instances (a random within-run split cannot: an ES elite set is a cluster of mutants).
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    elites = np.unique(np.asarray(elites, dtype=np.int64), axis=0)
    K, N = elites.shape
    if val_elites is not None:
        val = torch.as_tensor(np.unique(np.asarray(val_elites, dtype=np.int64), axis=0))
        train = torch.as_tensor(elites)
        n_val = int(len(val))
    else:
        perm = rng.permutation(K)
        n_val = max(1, int(cfg.val_fraction * K))
        val = torch.as_tensor(elites[perm[:n_val]])
        train = torch.as_tensor(elites[perm[n_val:]])
    if model is None:
        model = GPT(GPTConfig(vocab_size=vocab_size, max_len=N, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, dropout=cfg.dropout))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, best_state, since_best, step = float("inf"), None, 0, 0
    history = []
    for step in range(cfg.max_steps):
        model.train()
        idx = torch.randint(0, len(train), (min(cfg.batch_size, len(train)),))
        logp, _ = sequence_log_probs(model, train[idx], beta=1.0)
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                vlogp, _ = sequence_log_probs(model, val, beta=1.0)
                vloss = float(-vlogp.mean())
            history.append({"step": step, "train_nll": float(loss.detach()), "val_nll": vloss})
            if vloss < best_val - 1e-4:
                best_val, since_best = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 20
                if since_best >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    info = {"steps": step + 1, "best_val_nll_per_token": best_val,  # losses are already per-token means "n_train": int(len(train)), "n_val": int(n_val),
            "uniform_nll_per_token": float(np.log(vocab_size)), "history": history}
    return model, info


class ModelProposal:
    """Callable ``k -> (k, N)`` token array sampled from a GPT at inverse temperature ``beta``."""

    def __init__(self, model: GPT, seq_len: int, beta: float = 1.0, seed: int = 0):
        self.model, self.seq_len, self.beta = model, seq_len, beta
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, k: int) -> np.ndarray:
        toks, _ = sample_sequences(self.model, k, self.seq_len, self.beta, generator=self.gen)
        return toks.cpu().numpy()


@torch.no_grad()
def mean_token_entropy(model: GPT, tokens: np.ndarray, beta: float = 1.0) -> float:
    """Average per-position entropy (nats) of the model's next-token distribution along ``tokens``."""
    t = torch.as_tensor(np.asarray(tokens, dtype=np.int64))
    bos = torch.full((t.shape[0], 1), model.bos, dtype=torch.long)
    w = model(torch.cat([bos, t[:, :-1]], dim=1))
    logp = torch.log_softmax(-beta * w, dim=-1)
    return float(-(logp.exp() * logp).sum(-1).mean())
