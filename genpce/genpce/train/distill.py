"""Target-correlation-conditioned circuit synthesis (distillation from a variational teacher).

The learning problem is inverted relative to reward-driven generation. A continuous variational
solver (the *teacher*) finds a high-quality point ``c*`` in correlation space; the task is then to
produce a discrete, hardware-native circuit ``U`` whose correlator vector ``c(U)`` reproduces
``c*`` — an inverse problem with a dense ``m``-dimensional target, solvable by ordinary supervised
learning. Four pieces:

* :func:`build_vqa_corpus` — run the variational solver from many initialisations and depths and
  record the correlator vector at every checkpoint (poor, intermediate and near-optimal states);
* :class:`CorrelationTargetReward` — the three-part matching objective (correlation reconstruction,
  sign agreement, margin) that the evolutionary engine can maximise directly; used to establish the
  **representability ceiling** of a discrete circuit family for a given target;
* :class:`CorrelatorPredictor` — forward model ``U → c`` (learnability of correlation space);
* :class:`ConditionalGenerator` — inverse model ``c* → U``: a decoder-only transformer whose
  every position is conditioned on a projection of the target vector (prefix + additive
  conditioning), trained by cross-entropy on ``(c(U), U)`` pairs, sampled at inference and
  *verified by executing the circuit*.

Metrics (all computed on executed circuits, never on model predictions alone):
``D_c`` mean squared correlation error, ``A_x`` sign agreement, ``r`` decoded cut ratio,
``r_1000`` cut ratio from 1000-shot readout, ``median |c|`` margin, and gate counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genpce.baselines.pce_vqa import BrickworkMSAnsatz, TorchSimulator
from genpce.model.gpt import GPTConfig, _Block
from genpce.pce.correlators import CorrelatorSet, decode_signs
from genpce.pce.loss import RelaxedLossParams, relaxed_loss_torch
from genpce.problems.maxcut import MaxCutInstance
from genpce.train.evolution import Reward

__all__ = [
    "VQACorpus",
    "build_vqa_corpus",
    "CorrelationTargetReward",
    "ContinuousBrickworkCZ",
    "continuous_ceiling",
    "SingleAxisBrickworkCZ",
    "refine_and_snap",
    "graph_importance",
    "CorrelatorPredictor",
    "ConditionalGenerator",
    "train_forward",
    "train_inverse",
    "reconstruction_metrics",
]


# ---------------------------------------------------------------------- teacher corpus
@dataclass
class VQACorpus:
    """Checkpoints of variational optimisation trajectories: ``(θ, c, cut, loss, step, run, layers)``."""

    corr: np.ndarray  # (T, m)
    cut: np.ndarray  # (T,)
    loss: np.ndarray  # (T,)
    step: np.ndarray  # (T,)
    run: np.ndarray  # (T,)  trajectory id
    layers: np.ndarray  # (T,)
    params: list = field(default_factory=list)  # list of (N_p,) arrays (ragged across depths)
    meta: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        np.savez_compressed(path, corr=self.corr, cut=self.cut, loss=self.loss, step=self.step, run=self.run, layers=self.layers,
                            params=np.array(self.params, dtype=object), meta=np.array([str(self.meta)]))

    @classmethod
    def load(cls, path: str) -> "VQACorpus":
        z = np.load(path, allow_pickle=True)
        return cls(z["corr"], z["cut"], z["loss"], z["step"], z["run"], z["layers"], list(z["params"]))

    def finals(self) -> np.ndarray:
        """Indices of the last checkpoint of each trajectory."""
        return np.array([np.flatnonzero(self.run == r)[-1] for r in np.unique(self.run)])


def build_vqa_corpus(inst: MaxCutInstance, cset: CorrelatorSet, *, layers_list=(4, 8), seeds=range(10), lr=0.01, max_steps=1500,
                     every=5, stop_window=50, stop_tol=0.01, verbose=False) -> VQACorpus:
    """Run the Sciorilli variational solver from many initialisations; record every ``every`` steps."""
    lp = RelaxedLossParams.sciorilli(inst, cset.n, cset.k)
    corr, cut, loss, step, run, layers, params = [], [], [], [], [], [], []
    rid = 0
    for L in layers_list:
        ansatz = BrickworkMSAnsatz(cset.n, L)
        sim = TorchSimulator(ansatz, cset)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            theta = torch.tensor(ansatz.random_params(rng), requires_grad=True)
            opt = torch.optim.Adam([theta], lr=lr)
            hist = []
            t0 = perf_counter()
            for s in range(1, max_steps + 1):
                opt.zero_grad()
                c_t = sim.correlators(theta)
                l = relaxed_loss_torch(c_t, inst, lp)
                l.backward()
                l_val = float(l.detach())
                stop = len(hist) > stop_window and hist[-1 - stop_window] - l_val < stop_tol
                if s % every == 0 or s == 1 or stop or s == max_steps:
                    c_np = c_t.detach().numpy()
                    corr.append(c_np); cut.append(inst.cut_value(decode_signs(c_np[: inst.m]))); loss.append(l_val)
                    step.append(s); run.append(rid); layers.append(L); params.append(theta.detach().numpy().copy())
                hist.append(l_val)
                if stop:
                    break
                opt.step()
            if verbose:
                print(f"  layers={L} seed={seed}: {s} steps, final cut {cut[-1]:.0f} ({perf_counter()-t0:.0f}s)", flush=True)
            rid += 1
    return VQACorpus(np.stack(corr), np.array(cut), np.array(loss), np.array(step), np.array(run), np.array(layers), params,
                     meta={"instance": inst.name, "layers": list(layers_list), "seeds": list(seeds)})


# ---------------------------------------------------------------------- target-matching reward
def graph_importance(inst: MaxCutInstance) -> np.ndarray:
    """``w_i = Σ_{j:(i,j)∈E} |W_ij|`` normalised to mean 1 — how much each variable matters to the cut."""
    w = np.zeros(inst.m)
    np.add.at(w, inst.edges[:, 0], np.abs(inst.weights))
    np.add.at(w, inst.edges[:, 1], np.abs(inst.weights))
    return w / w.mean()


@dataclass(frozen=True)
class CorrelationTargetReward(Reward):
    """Score = −[ λ_c·mean w_i(c_i−c*_i)² + λ_s·mean softplus(−γ x*_i c_i) + λ_m·(−mean|c_i|) ].

    Duck-compatible with :class:`Reward` for the evolutionary engine; ``cut`` is still the real cut of
    the decoded signs, so search logs remain comparable with reward-driven runs.
    """

    target: np.ndarray = None
    weights: np.ndarray = None  # per-variable importance (mean 1); None = uniform
    lam_corr: float = 1.0
    lam_sign: float = 0.0
    lam_margin: float = 0.0
    gamma: float = 20.0

    @classmethod
    def for_target(cls, inst: MaxCutInstance, cset: CorrelatorSet, target: np.ndarray, *, weights=None, lam_corr=1.0, lam_sign=0.0, lam_margin=0.0, gamma=20.0):
        return cls(inst, RelaxedLossParams.sciorilli(inst, cset.n, cset.k), 0.0, 0.0, 0.2,
                   target=np.asarray(target, dtype=np.float64)[: inst.m], weights=weights, lam_corr=lam_corr, lam_sign=lam_sign, lam_margin=lam_margin, gamma=gamma)

    def __call__(self, corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(corr)[..., : self.inst.m]
        cut = self.cuts(corr)
        w = np.ones(self.inst.m) if self.weights is None else self.weights
        loss = self.lam_corr * np.mean(w * (c - self.target) ** 2, axis=-1)
        if self.lam_sign:
            xs = np.sign(self.target)
            loss = loss + self.lam_sign * np.mean(np.logaddexp(0.0, -self.gamma * xs * c), axis=-1)
        if self.lam_margin:
            loss = loss - self.lam_margin * np.mean(np.abs(c), axis=-1)
        return -loss, cut


# ---------------------------------------------------------------------- structural (continuous) ceiling
class ContinuousBrickworkCZ(TorchSimulator):
    """Continuous relaxation of the discrete token family: the same CZ-brickwork template of ``layers``
    layers, but an arbitrary single-qubit gate ``Rz(a)·Ry(b)·Rz(c)`` in every slot. Every discrete
    circuit of the pool is a point of this family, so its best correlation distance to a target is a
    lower bound on what any discrete circuit of that depth can reach (a structural ceiling)."""

    def __init__(self, n: int, layers: int, cset: CorrelatorSet):
        super().__init__(BrickworkMSAnsatz(n, layers), cset)
        self.layers = layers

    @property
    def num_params(self) -> int:
        return 3 * self.n * self.layers

    def statevector(self, params: torch.Tensor) -> torch.Tensor:
        psi = torch.zeros((2,) * self.n, dtype=self.dtype)
        psi[(0,) * self.n] = 1.0
        p = params.reshape(self.layers, self.n, 3)
        for layer in range(self.layers):
            for q in range(self.n):
                for axis, th in (("z", p[layer, q, 0]), ("y", p[layer, q, 1]), ("z", p[layer, q, 2])):
                    psi = self._apply_1q(psi, self._rot(axis, th, self.dtype), q)
            for q0 in range(layer % 2, self.n - 1, 2):  # CZ brickwork, same pairing as Pool._entangling_layer
                idx = [slice(None)] * self.n
                idx[self._axis(q0)] = 1
                idx[self._axis(q0 + 1)] = 1
                mask = torch.ones((2,) * self.n, dtype=self.dtype)
                mask[tuple(idx)] = -1.0
                psi = psi * mask
        return psi.reshape(-1)


def continuous_ceiling(cset: CorrelatorSet, layers: int, target: np.ndarray, *, restarts: int = 4, steps: int = 400, lr: float = 0.05, seed: int = 0) -> dict:
    """Adam on ``mean (c(θ) − c*)²`` over the continuous relaxation; best of ``restarts`` starts."""
    sim = ContinuousBrickworkCZ(cset.n, layers, cset)
    t = torch.as_tensor(np.asarray(target, dtype=np.float64)[: cset.m])
    rng = np.random.default_rng(seed)
    best = {"D_c": float("inf")}
    for r in range(restarts):
        theta = torch.tensor(rng.uniform(-np.pi, np.pi, size=sim.num_params), requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            c = sim.correlators(theta)
            loss = torch.mean((c - t) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            c = sim.correlators(theta).numpy()
        d = float(np.mean((c[: len(t)] - t.numpy()) ** 2))
        if d < best["D_c"]:
            best = {"D_c": d, "corr": c, "params": theta.detach().numpy().copy(), "restart": r}
    return best


class SingleAxisBrickworkCZ(ContinuousBrickworkCZ):
    """The discrete family with its angles made continuous: one rotation per slot about the axis fixed
    by a token sequence (identity slots become ``Rz`` starting at angle 0). Snapping the optimised
    angles back to the pool's grid returns a valid token sequence."""

    def __init__(self, n: int, layers: int, cset: CorrelatorSet, axes: list[str]):
        super().__init__(n, layers, cset)
        if len(axes) != n * layers:
            raise ValueError("one axis per slot")
        self.axes = axes

    @property
    def num_params(self) -> int:
        return self.n * self.layers

    def statevector(self, params: torch.Tensor) -> torch.Tensor:
        psi = torch.zeros((2,) * self.n, dtype=self.dtype)
        psi[(0,) * self.n] = 1.0
        for layer in range(self.layers):
            for q in range(self.n):
                slot = layer * self.n + q
                psi = self._apply_1q(psi, self._rot(self.axes[slot], params[slot], self.dtype), q)
            for q0 in range(layer % 2, self.n - 1, 2):
                idx = [slice(None)] * self.n
                idx[self._axis(q0)] = 1
                idx[self._axis(q0 + 1)] = 1
                mask = torch.ones((2,) * self.n, dtype=self.dtype)
                mask[tuple(idx)] = -1.0
                psi = psi * mask
        return psi.reshape(-1)


def refine_and_snap(pool, cset: CorrelatorSet, tokens: np.ndarray, target: np.ndarray, *, steps: int = 300, lr: float = 0.02) -> dict:
    """Continuous angle refinement of a discrete circuit (axes fixed), then snapping to the grid.

    Returns the correlation distance of the discrete start, of the refined continuous angles, and of
    the snapped (again discrete) circuit, plus the snapped token ids — separating "the angle grid is
    too coarse" from "one rotation per slot is too restrictive" and from "the search did not find it".
    """
    n, layers = cset.n, len(tokens) // cset.n
    axes, angles = [], []
    for j in tokens:
        tok = pool.tokens[int(j)]
        axes.append("z" if tok.kind == "id" else tok.kind[1])
        angles.append(0.0 if tok.kind == "id" else float(tok.theta))
    sim = SingleAxisBrickworkCZ(n, layers, cset, axes)
    t = torch.as_tensor(np.asarray(target, dtype=np.float64)[: cset.m])
    theta = torch.tensor(angles, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        d_start = float(torch.mean((sim.correlators(theta) - t) ** 2))
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((sim.correlators(theta) - t) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        c_ref = sim.correlators(theta).numpy()
    d_ref = float(np.mean((c_ref[: len(t)] - t.numpy()) ** 2))
    # snap every angle to the nearest token of the same axis (or identity for ~0), wrapping to (-π, π]
    grid = {}
    for j, tok in enumerate(pool.tokens):
        if tok.kind == "id":
            for ax in "xyz":
                grid.setdefault(ax, []).append((0.0, j))
        else:
            grid.setdefault(tok.kind[1], []).append((float(tok.theta), j))
    snapped = []
    th = np.angle(np.exp(1j * theta.detach().numpy()))
    for slot, ax in enumerate(axes):
        cands = grid[ax]
        k = int(np.argmin([abs(np.angle(np.exp(1j * (th[slot] - a)))) for a, _ in cands]))
        snapped.append(cands[k][1])
    snapped = np.array(snapped, dtype=np.int64)
    return {"D_c_start": d_start, "D_c_refined": d_ref, "corr_refined": c_ref, "snapped_tokens": snapped, "axes": axes, "angles_refined": th, "mean_abs_angle_move": float(np.mean(np.abs(np.angle(np.exp(1j * (th - np.array(angles)))))))}


# ---------------------------------------------------------------------- metrics
def reconstruction_metrics(c_gen: np.ndarray, c_target: np.ndarray, inst: MaxCutInstance) -> dict:
    """``D_c``, ``A_x`` (raw and up to a global flip), decoded cut ratio, margin, for one circuit."""
    m = inst.m
    cg, ct = np.asarray(c_gen)[:m], np.asarray(c_target)[:m]
    bk = inst.best_known() or 1.0
    xg, xt = decode_signs(cg), decode_signs(ct)
    agree = float(np.mean(xg == xt))
    return {"D_c": float(np.mean((cg - ct) ** 2)), "rel_err": float(np.linalg.norm(cg - ct) / (np.linalg.norm(ct) + 1e-12)),
            "A_x": agree, "A_x_flip": max(agree, 1 - agree), "ratio": inst.cut_value(xg) / bk, "target_ratio": inst.cut_value(xt) / bk,
            "median_abs_corr": float(np.median(np.abs(cg))), "target_median_abs_corr": float(np.median(np.abs(ct)))}


# ---------------------------------------------------------------------- forward model U → c
class CorrelatorPredictor(nn.Module):
    """Bidirectional encoder over tokens, mean-pooled, regressing the ``m`` correlators."""

    def __init__(self, vocab_size: int, seq_len: int, m: int, d_model: int = 128, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, m))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens) + self.pos(torch.arange(tokens.shape[1], device=tokens.device))[None]
        return self.head(self.enc(x).mean(1))


def train_forward(model: CorrelatorPredictor, tokens: np.ndarray, corr: np.ndarray, *, train_idx, val_idx, lr=5e-4, wd=0.01, batch=128, max_steps=4000, patience=400, seed=0) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, Y = torch.as_tensor(tokens, dtype=torch.long), torch.as_tensor(corr, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, state, since, hist = float("inf"), None, 0, []
    for s in range(max_steps):
        model.train()
        idx = rng.choice(train_idx, size=min(batch, len(train_idx)), replace=False)
        loss = F.mse_loss(model(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if s % 50 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(float(F.mse_loss(model(X[val_idx[i : i + 1024]]), Y[val_idx[i : i + 1024]], reduction="sum")) for i in range(0, len(val_idx), 1024)) / (len(val_idx) * Y.shape[1])
            hist.append({"step": s, "train": float(loss.detach()), "val": vl})
            if vl < best - 1e-7:
                best, since, state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 50
                if since >= patience:
                    break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return {"best_val_mse": best, "steps": s + 1, "history": hist}


# ---------------------------------------------------------------------- inverse model c* → U
class ConditionalGenerator(nn.Module):
    """Decoder-only transformer generating tokens conditioned on a target correlation vector.

    The target ``c*`` is projected to ``d_model`` and (i) used as the start-of-sequence embedding and
    (ii) added to every position (additive conditioning), so each generation step sees the target.
    """

    def __init__(self, vocab_size: int, seq_len: int, m: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.V, self.N, self.m = vocab_size, seq_len, m
        self.cond = nn.Sequential(nn.Linear(m, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len + 1, d_model)
        cfg = GPTConfig(vocab_size=vocab_size, max_len=seq_len + 1, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _forward(self, target: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        """``target (B, m)``, ``prefix (B, T)`` tokens already generated → logits ``(B, T+1, V)``."""
        h = self.cond(target)  # (B, d)
        x = torch.cat([h[:, None, :], self.tok(prefix)], dim=1) + h[:, None, :]
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for blk in self.blocks:
            x = blk(x, None)
        return self.head(self.ln(x))

    def log_probs(self, target: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Per-token log-probabilities of ``tokens (B, N)`` given ``target``."""
        logits = self._forward(target, tokens[:, :-1])  # positions 0..N-1 predict tokens 0..N-1
        return F.log_softmax(logits, dim=-1).gather(2, tokens[..., None]).squeeze(2)

    @torch.no_grad()
    def sample(self, target: torch.Tensor, num: int, temperature: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
        """``num`` circuits for one target ``(m,)`` (or a batch of targets ``(num, m)``)."""
        self.eval()
        t = target if target.dim() == 2 else target[None].expand(num, -1)
        prefix = torch.zeros((t.shape[0], 0), dtype=torch.long)
        for k in range(self.N):
            logits = self._forward(t, prefix)[:, -1, :] / temperature
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            prefix = torch.cat([prefix, nxt], dim=1)
        return prefix


def train_inverse(model: ConditionalGenerator, tokens: np.ndarray, corr: np.ndarray, *, train_idx, val_idx, lr=5e-4, wd=0.01, batch=64, max_steps=6000, patience=600, seed=0, noise=0.0) -> dict:
    """Cross-entropy on ``(c(U), U)`` pairs; optional Gaussian noise on the conditioning vector."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, C = torch.as_tensor(tokens, dtype=torch.long), torch.as_tensor(corr, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, state, since, hist = float("inf"), None, 0, []
    for s in range(max_steps):
        model.train()
        idx = rng.choice(train_idx, size=min(batch, len(train_idx)), replace=False)
        c_in = C[idx] + noise * torch.randn_like(C[idx]) if noise else C[idx]
        loss = -model.log_probs(c_in, X[idx]).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if s % 50 == 0:
            model.eval()
            with torch.no_grad():
                vl = float(sum(float(-model.log_probs(C[val_idx[i : i + 512]], X[val_idx[i : i + 512]]).sum()) for i in range(0, len(val_idx), 512)) / (len(val_idx) * X.shape[1]))
            hist.append({"step": s, "train": float(loss.detach()), "val": vl})
            if vl < best - 1e-5:
                best, since, state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 50
                if since >= patience:
                    break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return {"best_val_nll_per_token": best, "uniform_nll_per_token": float(np.log(model.V)), "steps": s + 1, "history": hist}
