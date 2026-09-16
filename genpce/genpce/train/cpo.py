"""Preference-based training of the circuit generator (DPO / CPO, best-vs-others).

Follows the training scheme of conditional-GQE / GQCO (Minami et al., 2025):

* at every step sample ``M`` circuits *on-policy* from the generator at temperature ``T_train``;
* evaluate them; the circuit with the lowest energy ``E`` (highest reward) is the winner ``w``;
* best-vs-others DPO loss with a **Boltzmann reference** ``π_ref(U) ∝ exp(−E(U))``:

      L(w, ℓ) = −log σ( β [ (log p_θ(w) + E(w)) − (log p_θ(ℓ) + E(ℓ)) ] ),

  averaged over all losers ``ℓ ≠ w`` — the energy gap ``E(ℓ) − E(w)`` is a margin the model's
  log-probability gap must exceed;
* CPO term: ``−log p_θ(w)`` (Xu et al., 2024), so the winner's absolute probability increases even
  when all samples are near-identical;
* evaluation samples at a higher temperature ``T_eval`` and keeps the best of the batch.

Extensions kept behind flags: ``n_winners > 1`` (top-k vs the rest), an entropy bonus, and
per-token normalisation of sequence log-probabilities (``normalize_by_length``) which keeps the
loss scale independent of ``N``.

Quantum cost is counted in circuit executions; the optimum is never used (energies are derived
from :class:`genpce.train.evolution.Reward`, normalised by the graph-intrinsic bound ν).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from genpce.model.gpt import GPT, GPTConfig, sample_sequences, sequence_log_probs
from genpce.pce.correlators import CorrelatorSet, decode_signs
from genpce.pool.vocab import Pool
from genpce.problems.maxcut import MaxCutInstance, one_pass_bit_swap
from genpce.sim.evaluator import Evaluator
from genpce.train.evolution import Reward
from genpce.train.gqe import CircuitDatabase

__all__ = ["CPOConfig", "CPOTrainer"]


@dataclass
class CPOConfig:
    seq_len: int
    budget: int = 20_000  # circuit executions
    samples_per_step: int = 256  # M
    updates_per_batch: int = 1  # gradient steps on each sampled batch (off-policy reuse; Boltzmann reference)
    n_winners: int = 1  # best-vs-others (1) or top-k-vs-rest
    loss_type: str = "cpo"  # "cpo" (best-vs-others DPO + NLL) | "boltzmann" (fit log p = −β_B·E + c on the batch)
    beta_boltzmann: float = 0.1  # inverse temperature of the target Boltzmann distribution (per unit energy)
    beta_boltzmann_final: float | None = None  # optional linear ramp of β_B over the budget
    beta_dpo: float = 0.1
    cpo_weight: float = 1.0
    energy_scale: float = 1.0  # E = −energy_scale · ν · score  (edge units when energy_scale = 1)
    temperature_train: float = 1.0
    temperature_eval: float = 2.0
    normalize_by_length: bool = False
    entropy_weight: float = 0.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.0
    seed: int = 0
    log_every: int = 1
    record_database: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class CPOTrainer:
    """Per-instance generator training with best-vs-others DPO/CPO (GQCO-style)."""

    def __init__(
        self,
        inst: MaxCutInstance,
        cset: CorrelatorSet,
        pool: Pool,
        evaluator: Evaluator,
        reward: Reward,
        cfg: CPOConfig,
        *,
        model: GPT | None = None,
    ):
        if cset.m < inst.m:
            raise ValueError(f"correlator set encodes {cset.m} variables but instance has {inst.m}")
        self.inst, self.cset, self.pool, self.evaluator, self.reward, self.cfg = inst, cset, pool, evaluator, reward, cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.model = model or GPT(GPTConfig(vocab_size=pool.size, max_len=cfg.seq_len, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, dropout=cfg.dropout))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
        self.database = CircuitDatabase()
        self.evaluations = 0
        self.step = 0
        self.records: list[dict] = []
        self.best = {"score": -np.inf, "cut": -np.inf, "tokens": None, "corr": None, "evaluations": 0}

    # -- energies --------------------------------------------------------------
    def _energies(self, corr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        score, cut = self.reward(corr)
        energy = -self.cfg.energy_scale * self.reward.params.nu * score
        return energy, score, cut

    def _evaluate(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        corr = self.evaluator.evaluate(self.pool.to_circuits(tokens))
        if self.cfg.record_database:
            self.database.push(tokens, corr, self.step)
        self.evaluations += len(tokens)
        energy, score, cut = self._energies(corr)
        i = int(np.argmax(score))
        if score[i] > self.best["score"]:
            self.best.update(score=float(score[i]), cut=float(cut[i]), tokens=tokens[i].copy(), corr=corr[i].copy(), evaluations=self.evaluations)
        return corr, energy, score, cut

    # -- loss ------------------------------------------------------------------
    def _beta_boltzmann(self) -> float:
        cfg = self.cfg
        if cfg.beta_boltzmann_final is None:
            return cfg.beta_boltzmann
        frac = min(1.0, self.evaluations / cfg.budget)
        return cfg.beta_boltzmann + frac * (cfg.beta_boltzmann_final - cfg.beta_boltzmann)

    def _loss(self, tokens: torch.Tensor, energy: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        beta_sample = 1.0 / cfg.temperature_train
        logp_tok, _ = sequence_log_probs(self.model, tokens, beta_sample)
        logp = logp_tok.sum(dim=1)
        if cfg.loss_type == "boltzmann":
            # Boltzmann fitting (logit matching in log-space): log p_θ(U) ≈ −β_B E(U) + const on the batch.
            # Unlike best-vs-others preferences this has a non-degenerate optimum (a distribution, not
            # a point mass) whose sharpness is set by β_B — the GQE logit-matching idea without exp().
            beta_b = self._beta_boltzmann()
            target = -beta_b * energy
            resid = (logp - target) - (logp - target).mean()
            loss = (resid ** 2).mean()
            with torch.no_grad():
                corr_fit = torch.corrcoef(torch.stack([logp, target]))[0, 1] if len(logp) > 2 else torch.tensor(0.0)
            info = {"dpo": float("nan"), "nll_winner": float(-logp[torch.argmin(energy)].detach()), "pref_acc": float(corr_fit), "beta_b": beta_b}
            if cfg.entropy_weight > 0:
                B, N = tokens.shape
                bos = torch.full((B, 1), self.model.bos, dtype=torch.long)
                w = self.model(torch.cat([bos, tokens[:, :-1]], dim=1))
                lp = torch.log_softmax(-beta_sample * w, dim=-1)
                ent = -(lp.exp() * lp).sum(-1).mean()
                loss = loss - cfg.entropy_weight * ent
                info["entropy"] = float(ent.detach())
            return loss, info
        if cfg.normalize_by_length:
            logp = logp / tokens.shape[1]
        # Boltzmann reference: log π_ref(U) = −E(U) + const  ⇒  log(p/π_ref) = log p + E
        implicit = logp + energy
        order = torch.argsort(energy)  # ascending energy: winners first
        winners, losers = order[: cfg.n_winners], order[cfg.n_winners :]
        diff = implicit[winners][:, None] - implicit[losers][None, :]  # (k, M−k)
        dpo = -F.logsigmoid(cfg.beta_dpo * diff).mean()
        nll = -logp[winners].mean()
        loss = dpo + cfg.cpo_weight * nll
        info = {"dpo": float(dpo.detach()), "nll_winner": float(nll.detach()), "pref_acc": float((diff.detach() > 0).float().mean())}
        if cfg.entropy_weight > 0:
            B, N = tokens.shape
            bos = torch.full((B, 1), self.model.bos, dtype=torch.long)
            w = self.model(torch.cat([bos, tokens[:, :-1]], dim=1))
            lp = torch.log_softmax(-beta_sample * w, dim=-1)
            ent = -(lp.exp() * lp).sum(-1).mean()
            loss = loss - cfg.entropy_weight * ent
            info["entropy"] = float(ent.detach())
        return loss, info

    # -- main loop -------------------------------------------------------------
    def run(self, callback: Callable[[dict], None] | None = None) -> pd.DataFrame:
        cfg = self.cfg
        t0 = perf_counter()
        bk = self.inst.best_known()
        while self.evaluations < cfg.budget:
            self.step += 1
            m = min(cfg.samples_per_step, cfg.budget - self.evaluations)
            toks_t, _ = sample_sequences(self.model, m, cfg.seq_len, 1.0 / cfg.temperature_train, generator=self.gen)
            tokens = toks_t.cpu().numpy()
            corr, energy, score, cut = self._evaluate(tokens)
            self.model.train()
            energy_t = torch.as_tensor(energy, dtype=torch.float32)
            for _ in range(cfg.updates_per_batch):
                loss, info = self._loss(toks_t, energy_t)
                self.opt.zero_grad()
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
            self.model.eval()
            if self.step % cfg.log_every == 0 or self.evaluations >= cfg.budget:
                counts = self.pool.gate_counts(self.best["tokens"])
                rec = {
                    "step": self.step, "evaluations": self.evaluations, "loss": float(loss.detach()), **info,
                    "batch_mean_cut": float(cut.mean()), "batch_max_cut": float(cut.max()),
                    "best_cut": self.best["cut"], "unique_frac": len({t.tobytes() for t in tokens}) / len(tokens),
                    "median_abs_corr_best": float(np.median(np.abs(self.best["corr"][: self.inst.m]))),
                    "two_qubit_best": counts["two_qubit"], "non_clifford_best": counts["non_clifford"],
                    "seconds": perf_counter() - t0,
                }
                if bk:
                    rec["ratio_best"] = self.best["cut"] / bk
                    rec["ratio_batch_mean"] = rec["batch_mean_cut"] / bk
                    rec["ratio_batch_max"] = rec["batch_max_cut"] / bk
                self.records.append(rec)
                if callback:
                    callback(rec)
        return pd.DataFrame(self.records)

    # -- evaluation ------------------------------------------------------------
    def evaluate(self, num: int = 100, temperature: float | None = None) -> dict:
        """Sample ``num`` circuits at the evaluation temperature and keep the best (GQCO protocol).

        Counts as ``num`` further circuit executions.
        """
        T = cfg_T = self.cfg.temperature_eval if temperature is None else temperature
        toks_t, _ = sample_sequences(self.model, num, self.cfg.seq_len, 1.0 / T, generator=self.gen)
        tokens = toks_t.cpu().numpy()
        corr, energy, score, cut = self._evaluate(tokens)
        i = int(np.argmax(score))
        x = decode_signs(corr[i, : self.inst.m])
        return {"cut": float(cut[i]), "cut_ls": float(self.inst.cut_value(one_pass_bit_swap(self.inst, x))), "tokens": tokens[i],
                "corr": corr[i], "mean_cut": float(cut.mean()), "temperature": cfg_T}

    def best_with_local_search(self) -> tuple[float, float]:
        x = decode_signs(self.best["corr"][: self.inst.m])
        return self.best["cut"], self.inst.cut_value(one_pass_bit_swap(self.inst, x))
