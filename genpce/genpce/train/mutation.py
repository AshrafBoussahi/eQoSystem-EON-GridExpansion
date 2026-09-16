"""Local circuit landscape: single-token mutations, their effects, and learned mutation scorers.

The unit of learning is an *edit* ``(i, v)`` — "replace the token at position ``i`` by ``v``" — applied
to a parent circuit ``U``. For a parent we can enumerate all ``N·(V−1)`` single-token children,
evaluate them, and record the reward change ``ΔR`` and the correlator change ``Δc``. This gives a
complete, exact picture of the local landscape around ``U`` (a *sensitivity map*) and dense
supervision for models that score edits:

* :class:`MutationScorer` — a bidirectional transformer encoder over the parent tokens whose
  per-position states are combined with token embeddings through a bilinear head, producing scores
  for **all** ``N × V`` edits in one forward pass (a mutation policy ``p(i, v | U)``);
* :class:`MLPScorer` — a flat one-hot MLP baseline with the same output;
* optional correlator head predicting ``Δc`` (dense, structured target).

Trained with a regression loss on ``ΔR`` (dense) or a listwise ranking loss (Boltzmann targets), and
evaluated by rank statistics on held-out parents. :class:`LearnedMutationProposal` wraps a trained
scorer as an edit sampler for the evolutionary engine, with online updates from evaluated children.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

__all__ = [
    "enumerate_single_mutations",
    "MutationDataset",
    "build_mutation_dataset",
    "MutationScorer",
    "MLPScorer",
    "ScorerConfig",
    "train_scorer",
    "rank_metrics",
    "LearnedMutationProposal",
]


# ---------------------------------------------------------------------- enumeration
def enumerate_single_mutations(parent: np.ndarray, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All children differing from ``parent`` in exactly one token.

    Returns ``(children (K, N), positions (K,), tokens (K,))`` with ``K = N·(V−1)``.
    """
    parent = np.asarray(parent, dtype=np.int64)
    N = len(parent)
    pos, tok = [], []
    for i in range(N):
        for v in range(vocab_size):
            if v != parent[i]:
                pos.append(i)
                tok.append(v)
    pos, tok = np.array(pos), np.array(tok)
    children = np.repeat(parent[None], len(pos), axis=0)
    children[np.arange(len(pos)), pos] = tok
    return children, pos, tok


# ---------------------------------------------------------------------- dataset
@dataclass
class MutationDataset:
    """Complete single-mutation neighbourhoods of ``P`` parents.

    ``dR[p, i, v]`` = reward(child) − reward(parent) (``NaN`` for the no-op ``v == parent[p, i]``);
    ``dcut`` likewise for the raw cut; ``dc[p, i, v, :]`` the correlator change (optional, float16).
    """

    parents: np.ndarray  # (P, N)
    parent_score: np.ndarray  # (P,)
    parent_cut: np.ndarray  # (P,)
    dR: np.ndarray  # (P, N, V)
    dcut: np.ndarray  # (P, N, V)
    dc: np.ndarray | None = None  # (P, N, V, m)
    meta: dict = field(default_factory=dict)

    @property
    def P(self) -> int:
        return int(self.parents.shape[0])

    @property
    def N(self) -> int:
        return int(self.parents.shape[1])

    @property
    def V(self) -> int:
        return int(self.dR.shape[2])

    def save(self, path: str) -> None:
        np.savez_compressed(path, parents=self.parents, parent_score=self.parent_score, parent_cut=self.parent_cut, dR=self.dR, dcut=self.dcut,
                            dc=self.dc if self.dc is not None else np.zeros(0, dtype=np.float16), meta=np.array([str(self.meta)]))

    @classmethod
    def load(cls, path: str) -> "MutationDataset":
        z = np.load(path, allow_pickle=False)
        dc = z["dc"] if z["dc"].size else None
        return cls(z["parents"], z["parent_score"], z["parent_cut"], z["dR"], z["dcut"], dc)

    def sensitivity(self) -> np.ndarray:
        """``S[p, i]`` = mean ΔR over replacement tokens at position ``i`` (Experiment 5)."""
        return np.nanmean(self.dR, axis=2)

    def improving_fraction(self) -> np.ndarray:
        return np.nanmean(self.dR > 0, axis=(1, 2))


def build_mutation_dataset(parents: np.ndarray, pool, evaluator, reward, *, keep_dc: bool = False, batch: int = 512) -> MutationDataset:
    """Evaluate every single-token child of every parent (``P·N·(V−1)`` circuit executions)."""
    parents = np.asarray(parents, dtype=np.int64)
    P, N = parents.shape
    V = pool.size
    m = reward.inst.m
    dR = np.full((P, N, V), np.nan, dtype=np.float32)
    dcut = np.full((P, N, V), np.nan, dtype=np.float32)
    dc = np.zeros((P, N, V, m), dtype=np.float16) if keep_dc else None
    pscore, pcut = np.zeros(P), np.zeros(P)
    for p in range(P):
        corr_p = evaluator.evaluate([pool.to_circuit(parents[p])])[0]
        s_p, c_p = reward(corr_p[None])
        pscore[p], pcut[p] = float(s_p[0]), float(c_p[0])
        children, pos, tok = enumerate_single_mutations(parents[p], V)
        for start in range(0, len(children), batch):
            sl = slice(start, start + batch)
            corr = evaluator.evaluate(pool.to_circuits(children[sl]))
            s, c = reward(corr)
            dR[p, pos[sl], tok[sl]] = s - pscore[p]
            dcut[p, pos[sl], tok[sl]] = c - pcut[p]
            if keep_dc:
                dc[p, pos[sl], tok[sl]] = (corr[:, :m] - corr_p[None, :m]).astype(np.float16)
    return MutationDataset(parents, pscore, pcut, dR, dcut, dc, meta={"pool": pool.name})


# ---------------------------------------------------------------------- models
@dataclass
class ScorerConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 5e-4
    weight_decay: float = 0.01
    batch_parents: int = 16
    max_steps: int = 3000
    patience: int = 300
    loss: str = "mse"  # "mse" | "listwise"
    listwise_tau: float = 0.5  # temperature of the Boltzmann target over ΔR (in reward units × 100)
    predict_dc: bool = False
    dc_weight: float = 1.0
    seed: int = 0


class MutationScorer(nn.Module):
    """Bidirectional transformer over the parent; bilinear head scores every edit ``(i, v)``."""

    def __init__(self, vocab_size: int, seq_len: int, cfg: ScorerConfig, m: int | None = None):
        super().__init__()
        self.cfg, self.V, self.N = cfg, vocab_size, seq_len
        d = cfg.d_model
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.ln = nn.LayerNorm(d)
        self.edit_emb = nn.Embedding(vocab_size, d)  # embedding of the *replacement* token
        self.pos_bias = nn.Parameter(torch.zeros(seq_len))
        self.tok_bias = nn.Parameter(torch.zeros(vocab_size))
        self.proj = nn.Linear(d, d)
        self.dc_head = nn.Linear(2 * d, m) if (cfg.predict_dc and m) else None

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(N, device=tokens.device))[None]
        return self.ln(self.encoder(x))  # (B, N, d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Scores for all edits: ``(B, N, V)``."""
        h = self.proj(self.encode(tokens))  # (B, N, d)
        return torch.einsum("bnd,vd->bnv", h, self.edit_emb.weight) / np.sqrt(h.shape[-1]) + self.pos_bias[None, :, None] + self.tok_bias[None, None, :]

    def forward_dc(self, tokens: torch.Tensor, pos: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        """Predicted correlator change for edits ``(pos, tok)`` of each sequence: ``(B, m)``."""
        h = self.encode(tokens)
        hp = h[torch.arange(len(pos)), pos]
        return self.dc_head(torch.cat([hp, self.edit_emb(tok)], dim=-1))


class MLPScorer(nn.Module):
    """Flat one-hot MLP baseline producing the same ``(B, N, V)`` edit scores."""

    def __init__(self, vocab_size: int, seq_len: int, hidden: int = 512):
        super().__init__()
        self.V, self.N = vocab_size, seq_len
        self.net = nn.Sequential(nn.Linear(seq_len * vocab_size, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, seq_len * vocab_size))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(tokens, self.V).float().reshape(tokens.shape[0], -1)
        return self.net(x).reshape(tokens.shape[0], self.N, self.V)


# ---------------------------------------------------------------------- training / evaluation
def _loss(pred: torch.Tensor, target: torch.Tensor, cfg: ScorerConfig) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if cfg.loss == "mse":
        return F.mse_loss(pred[mask], target[mask])
    # listwise: Boltzmann target over all valid edits of each parent (ΔR scaled to ~[−1, 1] × 100)
    B = pred.shape[0]
    logits = pred.reshape(B, -1).masked_fill(~mask.reshape(B, -1), -1e9)
    t = (target.reshape(B, -1) * 100.0 / cfg.listwise_tau).masked_fill(~mask.reshape(B, -1), -1e9)
    return -(F.softmax(t, dim=1) * F.log_softmax(logits, dim=1)).sum(1).mean()


def rank_metrics(pred: np.ndarray, true: np.ndarray, ks=(1, 5, 10, 50)) -> dict:
    """Rank statistics of predicted edit scores against exact ΔR for one parent (flattened)."""
    valid = ~np.isnan(true)
    p, t = pred[valid], true[valid]
    K = len(t)
    order_true = np.argsort(t)[::-1]
    rank_of = np.empty(K, dtype=int)
    rank_of[order_true] = np.arange(K)
    best_pred = int(np.argmax(p))
    out = {"spearman": float(spearmanr(p, t).correlation), "rank_of_pred_best": int(rank_of[best_pred]), "K": K,
           "regret": float(t.max() - t[best_pred]), "oracle_gain": float(t.max()), "pred_best_gain": float(t[best_pred])}
    for k in ks:
        top_pred = np.argsort(p)[::-1][:k]
        out[f"hit@{k}"] = float(rank_of[best_pred] < k)  # predicted best is within the true top-k
        out[f"mean_true_top{k}"] = float(t[top_pred].mean())
        out[f"improving_top{k}"] = float((t[top_pred] > 0).mean())
    out["improving_random"] = float((t > 0).mean())
    out["mean_true_random"] = float(t.mean())
    return out


def train_scorer(model: nn.Module, data: MutationDataset, cfg: ScorerConfig, *, train_idx: np.ndarray, val_idx: np.ndarray, verbose: bool = False) -> dict:
    """Train on parents ``train_idx``; early-stop on validation loss over parents ``val_idx``."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    X = torch.as_tensor(data.parents)
    Y = torch.as_tensor(data.dR)
    DC = torch.as_tensor(data.dc.astype(np.float32)) if (cfg.predict_dc and data.dc is not None) else None
    best, best_state, since, hist = float("inf"), None, 0, []
    for step in range(cfg.max_steps):
        model.train()
        idx = rng.choice(train_idx, size=min(cfg.batch_parents, len(train_idx)), replace=False)
        pred = model(X[idx])
        loss = _loss(pred, Y[idx], cfg)
        if DC is not None:
            # sample one edit per parent for the correlator head
            pos = torch.as_tensor(rng.integers(0, data.N, size=len(idx)))
            tok = torch.as_tensor(rng.integers(0, data.V, size=len(idx)))
            ok = tok != X[idx, pos]
            if ok.any():
                dc_pred = model.forward_dc(X[idx][ok], pos[ok], tok[ok])
                loss = loss + cfg.dc_weight * F.mse_loss(dc_pred, DC[idx][ok, pos[ok], tok[ok]])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                vl = float(_loss(model(X[val_idx]), Y[val_idx], cfg))
            hist.append({"step": step, "train": float(loss.detach()), "val": vl})
            if verbose:
                print(f"    step {step} train {float(loss.detach()):.4f} val {vl:.4f}", flush=True)
            if vl < best - 1e-6:
                best, since, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 25
                if since >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_val": best, "steps": step + 1, "history": hist}


@torch.no_grad()
def evaluate_scorer(model: nn.Module, data: MutationDataset, idx: np.ndarray) -> dict:
    """Average rank metrics over parents ``idx``."""
    model.eval()
    pred = model(torch.as_tensor(data.parents[idx])).cpu().numpy()
    rows = [rank_metrics(pred[j].reshape(-1), data.dR[p].reshape(-1)) for j, p in enumerate(idx)]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# ---------------------------------------------------------------------- proposal
class LearnedMutationProposal:
    """Edit sampler for the evolutionary engine: scores all edits of a parent, samples with temperature.

    ``epsilon`` of the proposed edits are uniformly random (exploration floor). ``update`` performs a
    few gradient steps on newly evaluated ``(parent, i, v, ΔR)`` records (online learning).
    """

    def __init__(self, model: nn.Module, *, temperature: float = 1.0, epsilon: float = 0.2, lr: float = 3e-4, steps_per_update: int = 4, seed: int = 0):
        self.model, self.T, self.eps = model, temperature, epsilon
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.steps = steps_per_update
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.buffer: list[tuple[np.ndarray, int, int, float]] = []
        self.buffer_cap = 20_000

    @torch.no_grad()
    def propose(self, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One edit per parent row: returns ``(children, pos, tok)``."""
        self.model.eval()
        parents = np.asarray(parents, dtype=np.int64)
        B, N = parents.shape
        scores = self.model(torch.as_tensor(parents))  # (B, N, V)
        V = scores.shape[-1]
        scores[torch.arange(B)[:, None], torch.arange(N)[None, :], torch.as_tensor(parents)] = -1e9  # forbid no-ops
        flat = torch.softmax(scores.reshape(B, -1) / self.T, dim=1)
        choice = torch.multinomial(flat, 1, generator=self.gen).squeeze(1).numpy()
        pos, tok = choice // V, choice % V
        rand = self.rng.random(B) < self.eps
        if rand.any():
            rp = self.rng.integers(0, N, size=int(rand.sum()))
            rt = (parents[rand, rp] + self.rng.integers(1, V, size=int(rand.sum()))) % V
            pos[rand], tok[rand] = rp, rt
        children = parents.copy()
        children[np.arange(B), pos] = tok
        return children, pos, tok

    def record(self, parents: np.ndarray, pos: np.ndarray, tok: np.ndarray, dR: np.ndarray) -> None:
        for p, i, v, r in zip(parents, pos, tok, dR):
            self.buffer.append((np.asarray(p, dtype=np.int64), int(i), int(v), float(r)))
        if len(self.buffer) > self.buffer_cap:
            del self.buffer[: len(self.buffer) - self.buffer_cap]

    def update(self) -> float:
        if len(self.buffer) < 32:
            return float("nan")
        self.model.train()
        loss_val = float("nan")
        for _ in range(self.steps):
            idx = self.rng.choice(len(self.buffer), size=min(128, len(self.buffer)), replace=False)
            P = torch.as_tensor(np.stack([self.buffer[i][0] for i in idx]))
            pos = torch.as_tensor([self.buffer[i][1] for i in idx])
            tok = torch.as_tensor([self.buffer[i][2] for i in idx])
            y = torch.as_tensor([self.buffer[i][3] for i in idx], dtype=torch.float32)
            pred = self.model(P)[torch.arange(len(idx)), pos, tok]
            loss = F.mse_loss(pred, y)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            loss_val = float(loss.detach())
        self.model.eval()
        return loss_val
