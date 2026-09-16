"""GenPCE training loop: GQE-style generative circuit synthesis with PCE readout.

Per epoch (cf. Nakaji et al., Sec. 2.2.2):

1. sample ``n_sample`` token sequences from the transformer at inverse temperature ``β``;
2. build the circuits, evaluate them (any :class:`genpce.sim.evaluator.Evaluator`) → correlators;
3. decode signs → spins → reward (exact cut by default — no relaxation needed);
4. push ``(tokens, log-probs, reward)`` into a FIFO replay buffer and ``(tokens, correlators)``
   into the instance-agnostic :class:`CircuitDatabase`;
5. run ``n_iter`` minibatch updates with the GRPO (PPO-clip, group-normalised advantages) and/or
   logit-matching loss;
6. adapt ``β`` with the dispersion-triggered schedule (Nakaji et al., Appendix A.3).

Everything the paper reports (best cut, evaluations, gate counts, β, entropy) is logged per epoch
to a pandas ``DataFrame``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
import torch

from genpce.model.gpt import GPT, GPTConfig, sample_sequences, sequence_log_probs
from genpce.pce.correlators import CorrelatorSet, decode_signs
from genpce.pce.loss import RelaxedLossParams, relaxed_loss_np
from genpce.pool.vocab import Pool
from genpce.problems.maxcut import MaxCutInstance, one_pass_bit_swap
from genpce.sim.evaluator import Evaluator

__all__ = ["GenPCEConfig", "ReplayBuffer", "CircuitDatabase", "GenPCETrainer", "random_search"]


# ---------------------------------------------------------------------- config
@dataclass
class GenPCEConfig:
    seq_len: int
    epochs: int = 300
    n_sample: int = 50
    n_batch: int = 50
    n_iter: int = 5
    buffer_size: int = 1000
    beta0: float = 1.0
    beta_step: float = 0.02
    beta_dispersion_tol: float = 1e-3
    beta_min: float = 0.05
    lr: float = 1e-4
    weight_decay: float = 0.01
    clip_eps: float = 0.2
    loss: str = "grpo"  # "grpo" | "logit_matching" | "cem" | "grpo+lm" | "grpo+cem"
    lm_weight: float = 1.0
    elite_fraction: float = 0.1  # top fraction of the buffer imitated by the "cem" loss
    cem_weight: float = 1.0
    entropy_weight: float = 0.0  # bonus on the mean per-position policy entropy (prevents collapse)
    mutation_rate: float = 0.0  # fraction of sampled sequences that receive one random token mutation
    reward: str = "cut"  # "cut" | "relaxed" | "shaped"
    shaped_lambda: float = 0.5
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    seed: int = 0
    log_every: int = 1
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- storage
@dataclass
class ReplayBuffer:
    """FIFO buffer of ``(tokens, old log-probs, reward)`` triples."""

    capacity: int
    tokens: list[np.ndarray] = field(default_factory=list)
    logps: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def push(self, tokens: np.ndarray, logps: np.ndarray, rewards: np.ndarray) -> None:
        for t, lp, r in zip(tokens, logps, rewards):
            self.tokens.append(np.asarray(t, dtype=np.int64))
            self.logps.append(np.asarray(lp, dtype=np.float32))
            self.rewards.append(float(r))
        overflow = len(self.tokens) - self.capacity
        if overflow > 0:
            del self.tokens[:overflow]
            del self.logps[:overflow]
            del self.rewards[:overflow]

    def __len__(self) -> int:
        return len(self.tokens)

    def sample(self, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.choice(len(self.tokens), size=min(size, len(self.tokens)), replace=False)
        return self._gather(idx)

    def sample_elite(self, size: int, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A random subset of the top ``fraction`` of the buffer by reward."""
        k = max(1, int(round(fraction * len(self.tokens))))
        elite = np.argsort(np.asarray(self.rewards))[::-1][:k]
        idx = rng.choice(elite, size=min(size, k), replace=False)
        return self._gather(idx)

    def _gather(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.stack([self.tokens[i] for i in idx]),
            np.stack([self.logps[i] for i in idx]),
            np.array([self.rewards[i] for i in idx], dtype=np.float64),
        )


@dataclass
class CircuitDatabase:
    """Instance-agnostic store of ``(tokens, correlator vector)`` pairs.

    Because the PCE readout depends only on the circuit (and on ``(n, k)``), these records can be
    re-scored *exactly* under any objective over the same variables — the basis of zero-quantum-cost
    transfer between instances.
    """

    tokens: list[np.ndarray] = field(default_factory=list)
    corr: list[np.ndarray] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)

    def push(self, tokens: np.ndarray, corr: np.ndarray, epoch: int) -> None:
        for t, c in zip(tokens, corr):
            self.tokens.append(np.asarray(t, dtype=np.int64))
            self.corr.append(np.asarray(c, dtype=np.float32))
            self.epoch.append(int(epoch))

    def __len__(self) -> int:
        return len(self.tokens)

    def rescore(self, inst: MaxCutInstance) -> np.ndarray:
        """Cut values of every stored circuit under a (possibly new) instance — no quantum cost."""
        if not self.corr:
            return np.zeros(0)
        x = decode_signs(np.stack(self.corr).astype(np.float64))
        return inst.cut_values(x)

    def save(self, path: str) -> None:
        np.savez_compressed(path, tokens=np.stack(self.tokens), corr=np.stack(self.corr), epoch=np.array(self.epoch))

    @classmethod
    def load(cls, path: str) -> "CircuitDatabase":
        data = np.load(path)
        db = cls()
        db.tokens = list(data["tokens"])
        db.corr = list(data["corr"])
        db.epoch = list(int(e) for e in data["epoch"])
        return db


# ---------------------------------------------------------------------- trainer
class GenPCETrainer:
    """Train a transformer to write circuits whose PCE-decoded spins maximise a MaxCut objective."""

    def __init__(
        self,
        inst: MaxCutInstance,
        cset: CorrelatorSet,
        pool: Pool,
        evaluator: Evaluator,
        cfg: GenPCEConfig,
        *,
        model: GPT | None = None,
        reference_cut: float | None = None,
    ):
        if cset.m < inst.m:
            raise ValueError(f"correlator set encodes {cset.m} variables but instance has {inst.m}")
        self.inst = inst
        self.cset = cset
        self.pool = pool
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.model = model or GPT(
            GPTConfig(
                vocab_size=pool.size,
                max_len=cfg.seq_len,
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
            )
        )
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.database = CircuitDatabase()
        self.beta = cfg.beta0
        self.loss_params = RelaxedLossParams.sciorilli(inst, cset.n, cset.k)
        # Reward normalisation uses only instance-intrinsic quantities (the Edwards–Erdős /
        # Poljak–Turzík cut lower bound ν, computed from the graph). The optimal or best-known cut is
        # NEVER used in training — only for reporting approximation ratios in the logs.
        self.reference_cut = float(reference_cut or self.loss_params.nu)
        self.evaluations = 0  # circuits executed (each needs 3 measurement settings)
        self.best = {"cut": -math.inf, "tokens": None, "corr": None, "epoch": -1, "cut_ls": -math.inf}
        self.records: list[dict] = []
        self.lm_offset: float | None = None

    # -- rewards ----------------------------------------------------------------
    def rewards(self, cuts: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """Higher is better. ``"cut"`` = exact normalised cut (no relaxation, no α/β)."""
        r_cut = cuts / self.reference_cut
        if self.cfg.reward == "cut":
            return r_cut
        r_relaxed = -relaxed_loss_np(corr[:, : self.inst.m], self.inst, self.loss_params) / self.loss_params.nu
        if self.cfg.reward == "relaxed":
            return r_relaxed
        if self.cfg.reward == "shaped":
            return r_cut + self.cfg.shaped_lambda * r_relaxed
        raise ValueError(f"unknown reward {self.cfg.reward!r}")

    # -- losses -----------------------------------------------------------------
    def _grpo_loss(self, logp_new: torch.Tensor, logp_old: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        ratio = torch.exp(logp_new - logp_old)  # (B, N) per-token importance ratio
        adv = adv[:, None]
        clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
        return -torch.mean(torch.minimum(ratio * adv, clipped * adv))

    def _logit_matching_loss(self, w_sel: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """``mean (exp(-β w_sum) - exp(-β E'))²`` with ``E' = -r - E_min ≥ 0`` (Nakaji et al., Eq. 6)."""
        energy = -rewards
        if self.lm_offset is None:
            self.lm_offset = float(energy.min())
        shifted = energy - self.lm_offset
        w_sum = w_sel.sum(dim=1)
        return torch.mean((torch.exp(-self.beta * w_sum) - torch.exp(-self.beta * shifted)) ** 2)

    def _entropy(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean per-position entropy (nats) of the policy along the given sequences."""
        B, N = tokens.shape
        bos = torch.full((B, 1), self.model.bos, dtype=torch.long, device=tokens.device)
        w = self.model(torch.cat([bos, tokens[:, :-1]], dim=1))
        logp = torch.log_softmax(-self.beta * w, dim=-1)
        return -(logp.exp() * logp).sum(-1).mean()

    def _cem_loss(self) -> torch.Tensor:
        """Elite imitation (cross-entropy method): maximise the likelihood of the best buffer entries.

        The transformer becomes the distribution model of an estimation-of-distribution algorithm:
        it learns to propose sequences that resemble the elites, then samples new candidates around
        them. Low-variance and robust to rugged, plateau-like rewards.
        """
        tokens, _, _ = self.buffer.sample_elite(self.cfg.n_batch, self.cfg.elite_fraction, self.rng)
        logp, _ = sequence_log_probs(self.model, torch.as_tensor(tokens, device=self.device), self.beta)
        return -logp.mean()

    def _update(self) -> float:
        self.model.train()
        loss = torch.zeros((), device=self.device)
        if self.cfg.loss in ("grpo", "grpo+lm", "grpo+cem", "logit_matching"):
            tokens, logp_old, rewards = self.buffer.sample(self.cfg.n_batch, self.rng)
            tokens_t = torch.as_tensor(tokens, device=self.device)
            logp_old_t = torch.as_tensor(logp_old, dtype=torch.float32, device=self.device)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            logp_new, w_sel = sequence_log_probs(self.model, tokens_t, self.beta)
            if self.cfg.loss in ("grpo", "grpo+lm", "grpo+cem"):
                loss = loss + self._grpo_loss(logp_new, logp_old_t, rewards_t)
            if self.cfg.loss in ("logit_matching", "grpo+lm"):
                loss = loss + self.cfg.lm_weight * self._logit_matching_loss(w_sel, rewards_t)
        if self.cfg.loss in ("cem", "grpo+cem"):
            loss = loss + self.cfg.cem_weight * self._cem_loss()
        if self.cfg.entropy_weight > 0:
            tokens_h, _, _ = self.buffer.sample(self.cfg.n_batch, self.rng)
            loss = loss - self.cfg.entropy_weight * self._entropy(torch.as_tensor(tokens_h, device=self.device))
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return float(loss)

    def _mutate(self, tokens: np.ndarray, logps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replace one random token in a ``mutation_rate`` fraction of the sequences (local exploration).

        Mutated sequences are re-scored under the current policy so that stored log-probabilities
        stay consistent with the tokens actually evaluated.
        """
        tokens = tokens.copy()
        hit = np.flatnonzero(self.rng.random(len(tokens)) < self.cfg.mutation_rate)
        if len(hit) == 0:
            return tokens, logps
        pos = self.rng.integers(0, tokens.shape[1], size=len(hit))
        tokens[hit, pos] = self.rng.integers(0, self.pool.size, size=len(hit))
        with torch.no_grad():
            lp, _ = sequence_log_probs(self.model, torch.as_tensor(tokens[hit], device=self.device), self.beta)
        logps = logps.copy()
        logps[hit] = lp.cpu().numpy()
        return tokens, logps

    # -- main loop --------------------------------------------------------------
    def run(self, epochs: int | None = None, callback: Callable[[dict], None] | None = None) -> pd.DataFrame:
        epochs = self.cfg.epochs if epochs is None else epochs
        t_start = perf_counter()
        for epoch in range(epochs):
            tokens_t, logps_t = sample_sequences(self.model, self.cfg.n_sample, self.cfg.seq_len, self.beta, generator=self.gen)
            tokens = tokens_t.cpu().numpy()
            logps = logps_t.cpu().numpy()
            if self.cfg.mutation_rate > 0:
                tokens, logps = self._mutate(tokens, logps)
            corr = self.evaluator.evaluate(self.pool.to_circuits(tokens))
            self.evaluations += len(tokens)
            x = decode_signs(corr[:, : self.inst.m])
            cuts = self.inst.cut_values(x)
            rewards = self.rewards(cuts, corr)
            self.buffer.push(tokens, logps, rewards)
            self.database.push(tokens, corr, epoch)

            i_best = int(np.argmax(cuts))
            if cuts[i_best] > self.best["cut"]:
                x_ls = one_pass_bit_swap(self.inst, x[i_best])
                self.best.update(
                    cut=float(cuts[i_best]), tokens=tokens[i_best].copy(), corr=corr[i_best].copy(),
                    epoch=epoch, cut_ls=float(self.inst.cut_value(x_ls)),
                )

            losses = [self._update() for _ in range(self.cfg.n_iter)]

            # dispersion-triggered β schedule (Nakaji et al., App. A.3)
            if float(np.std(rewards)) < self.cfg.beta_dispersion_tol:
                self.beta = max(self.cfg.beta_min, self.beta - self.cfg.beta_step)
            else:
                self.beta += self.cfg.beta_step

            if epoch % self.cfg.log_every == 0 or epoch == epochs - 1:
                counts = self.pool.gate_counts(tokens[i_best])
                rec = {
                    "epoch": epoch,
                    "evaluations": self.evaluations,
                    "beta": self.beta,
                    "loss": float(np.mean(losses)),
                    "cut_mean": float(cuts.mean()),
                    "cut_max": float(cuts.max()),
                    "cut_std": float(cuts.std()),
                    "reward_mean": float(rewards.mean()),
                    "best_cut": self.best["cut"],
                    "best_cut_ls": self.best["cut_ls"],
                    "best_epoch": self.best["epoch"],
                    "unique_frac": len({t.tobytes() for t in tokens}) / len(tokens),
                    "min_abs_corr_best": float(np.min(np.abs(corr[i_best, : self.inst.m]))),
                    "two_qubit_best": counts["two_qubit"],
                    "non_clifford_best": counts["non_clifford"],
                    "gates_best": counts["gates"],
                    "seconds": perf_counter() - t_start,
                }
                if self.inst.best_known():
                    rec["ratio_best"] = self.best["cut"] / self.inst.best_known()
                    rec["ratio_best_ls"] = self.best["cut_ls"] / self.inst.best_known()
                    rec["ratio_epoch_mean"] = rec["cut_mean"] / self.inst.best_known()
                self.records.append(rec)
                if callback:
                    callback(rec)
        return pd.DataFrame(self.records)

    # -- transfer helpers -------------------------------------------------------
    def warm_start(self, database: CircuitDatabase, top_fraction: float = 0.1, *, reward_from_cut: bool = True) -> int:
        """Seed the replay buffer with the top ``top_fraction`` of ``database`` re-scored on ``self.inst``.

        Old log-probabilities are recomputed under the *current* policy, so the first GRPO updates
        see importance ratios of one (pure re-weighting by advantage). Returns the number of
        records inserted. No quantum evaluations are spent.
        """
        if len(database) == 0:
            return 0
        cuts = database.rescore(self.inst)
        keep = max(1, int(round(top_fraction * len(database))))
        idx = np.argsort(cuts)[::-1][:keep]
        tokens = np.stack([database.tokens[i] for i in idx])
        corr = np.stack([database.corr[i] for i in idx]).astype(np.float64)
        rewards = self.rewards(cuts[idx], corr)
        with torch.no_grad():
            logp, _ = sequence_log_probs(self.model, torch.as_tensor(tokens, device=self.device), self.beta)
        self.buffer.push(tokens, logp.cpu().numpy(), rewards)
        return int(keep)


# ---------------------------------------------------------------------- random baseline
def random_search(
    inst: MaxCutInstance,
    cset: CorrelatorSet,
    pool: Pool,
    evaluator: Evaluator,
    *,
    seq_len: int,
    total: int,
    batch: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Uniformly random token sequences — the no-learning control with identical quantum budget."""
    rng = np.random.default_rng(seed)
    best = -math.inf
    best_ls = -math.inf
    records = []
    done = 0
    while done < total:
        b = min(batch, total - done)
        tokens = rng.integers(0, pool.size, size=(b, seq_len))
        corr = evaluator.evaluate(pool.to_circuits(tokens))
        x = decode_signs(corr[:, : inst.m])
        cuts = inst.cut_values(x)
        i = int(np.argmax(cuts))
        if cuts[i] > best:
            best = float(cuts[i])
            best_ls = float(inst.cut_value(one_pass_bit_swap(inst, x[i])))
        done += b
        rec = {"evaluations": done, "best_cut": best, "best_cut_ls": best_ls, "cut_mean": float(cuts.mean())}
        if inst.best_known():
            rec["ratio_best"] = best / inst.best_known()
            rec["ratio_best_ls"] = best_ls / inst.best_known()
        records.append(rec)
    return pd.DataFrame(records)
