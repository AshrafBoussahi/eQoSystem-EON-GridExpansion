"""Elitist evolutionary search in token space — the per-instance engine of GenPCE.

Phase-0 findings (docs/02_E0_report.md) showed that, for correlation-encoded objectives, an elitist
(μ+λ) search over discrete, parameter-free circuits reaches VQA-level cut quality at roughly a tenth
of the parameter-shift VQA budget, whereas per-instance policy-gradient training of a generator does
not learn. This module is that engine, written so that a generative model can plug in as an
*additional proposal operator* (and as a cross-instance prior):

* **Selection** — keep the best ``mu`` sequences (by reward) of parents ∪ offspring.
* **Proposals** — offspring are mutants of random parents (1–``max_mutations`` random token
  replacements); optionally a fraction ``model_fraction`` is sampled from a transformer at inverse
  temperature ``model_beta``.
* **Reward** — :class:`Reward`: exact cut, relaxed PCE loss, their mixture ("shaped"), or a margin
  term; all normalised by the instance-intrinsic bound ``ν``. The optimum is never used.
* **Archive** — every evaluated ``(tokens, correlators)`` goes to a :class:`CircuitDatabase`
  (instance-agnostic quantum data) and the running elite set is available for model training.

Quantum cost is counted as circuit executions (each needing three measurement settings).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from genpce.pce.correlators import CorrelatorSet, decode_signs
from genpce.pce.loss import RelaxedLossParams, relaxed_loss_np
from genpce.pool.vocab import Pool
from genpce.problems.maxcut import MaxCutInstance, one_pass_bit_swap
from genpce.sim.evaluator import Evaluator
from genpce.train.gqe import CircuitDatabase

__all__ = ["Reward", "EvolutionConfig", "EvolutionarySearch"]


# ---------------------------------------------------------------------- reward
@dataclass(frozen=True)
class Reward:
    """Scalar score of a correlator vector for one instance (higher is better).

    ``score = cut/ν + relaxed_weight · (−L_relaxed/ν) + margin_weight · mean(min(|c|, τ))/τ``

    ``ν`` is the Edwards–Erdős / Poljak–Turzík cut bound of the instance (graph-intrinsic).
    """

    inst: MaxCutInstance
    params: RelaxedLossParams
    relaxed_weight: float = 0.5
    margin_weight: float = 0.0
    margin_tau: float = 0.2

    @classmethod
    def shaped(cls, inst: MaxCutInstance, cset: CorrelatorSet, relaxed_weight: float = 0.5, margin_weight: float = 0.0) -> "Reward":
        return cls(inst, RelaxedLossParams.sciorilli(inst, cset.n, cset.k), relaxed_weight, margin_weight)

    @classmethod
    def exact(cls, inst: MaxCutInstance, cset: CorrelatorSet) -> "Reward":
        return cls.shaped(inst, cset, relaxed_weight=0.0)

    def cuts(self, corr: np.ndarray) -> np.ndarray:
        return self.inst.cut_values(decode_signs(corr[..., : self.inst.m]))

    def __call__(self, corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(score, cut)`` for a batch of correlator vectors ``(B, m')``."""
        nu = self.params.nu
        cut = self.cuts(corr)
        score = cut / nu
        c = corr[..., : self.inst.m]
        if self.relaxed_weight:
            score = score - self.relaxed_weight * relaxed_loss_np(c, self.inst, self.params) / nu
        if self.margin_weight:
            score = score + self.margin_weight * np.mean(np.minimum(np.abs(c), self.margin_tau), axis=-1) / self.margin_tau
        return score, cut


# ---------------------------------------------------------------------- config
@dataclass
class EvolutionConfig:
    seq_len: int
    budget: int = 20_000  # circuit executions
    mu: int = 10
    lam: int = 50
    max_mutations: int = 2
    model_fraction: float = 0.0  # share of offspring proposed by the model (if one is attached)
    model_beta: float = 1.0
    archive_size: int = 200  # running elite archive (by reward) used to update a learned proposal
    record_database: bool = True  # store every evaluated (tokens, correlators); disable for very large runs
    learned_fraction: float = 0.0  # share of offspring proposed by a learned *mutation* scorer (single-token edits)
    single_token_mutations: bool = False  # random offspring are single-token edits (set automatically with a mutation proposal)
    learned_update_every: int = 1  # generations between online updates of the mutation scorer
    update_every: int = 1  # generations between proposal updates
    seed: int = 0
    log_every: int = 1  # generations

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- search
class EvolutionarySearch:
    """(μ+λ) evolutionary search over token sequences with optional model proposals."""

    def __init__(
        self,
        inst: MaxCutInstance,
        cset: CorrelatorSet,
        pool: Pool,
        evaluator: Evaluator,
        reward: Reward,
        cfg: EvolutionConfig,
        *,
        proposal: Callable[[int], np.ndarray] | None = None,
        mutation_proposal=None,
        init_population: np.ndarray | None = None,
        database: CircuitDatabase | None = None,
    ):
        if cset.m < inst.m:
            raise ValueError(f"correlator set encodes {cset.m} variables but instance has {inst.m}")
        self.inst, self.cset, self.pool, self.evaluator, self.reward, self.cfg = inst, cset, pool, evaluator, reward, cfg
        self.proposal = proposal  # callable: number of sequences -> (k, seq_len) token array
        self.mutation_proposal = mutation_proposal  # LearnedMutationProposal (propose / record / update)
        if mutation_proposal is not None:
            cfg.single_token_mutations = True
        self.rng = np.random.default_rng(cfg.seed)
        self.database = database if database is not None else CircuitDatabase()
        self.evaluations = 0
        self.generation = 0
        self.records: list[dict] = []
        self.best = {"score": -math.inf, "cut": -math.inf, "tokens": None, "corr": None, "evaluations": 0}
        self._init = init_population
        self.archive_tokens = np.zeros((0, cfg.seq_len), dtype=np.int64)
        self.archive_scores = np.zeros(0)
        self._last_scorer_loss = float("nan")

    # -- evaluation ----------------------------------------------------------
    def _evaluate(self, seqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        corr = self.evaluator.evaluate(self.pool.to_circuits(seqs))
        if self.cfg.record_database:
            self.database.push(seqs, corr, self.generation)
        self.evaluations += len(seqs)
        score, cut = self.reward(corr)
        i = int(np.argmax(score))
        if score[i] > self.best["score"]:
            self.best.update(score=float(score[i]), cut=float(cut[i]), tokens=seqs[i].copy(), corr=corr[i].copy(), evaluations=self.evaluations)
        return score, cut, corr

    def _update_archive(self, seqs: np.ndarray, scores: np.ndarray) -> None:
        """Merge new sequences into the running top-``archive_size`` elite archive (deduplicated)."""
        toks = np.concatenate([self.archive_tokens, seqs])
        sc = np.concatenate([self.archive_scores, scores])
        # keep the best score per unique sequence
        best = {}
        for i, row in enumerate(toks):
            key = row.tobytes()
            if key not in best or sc[i] > sc[best[key]]:
                best[key] = i
        idx = np.array(list(best.values()))
        order = idx[np.argsort(sc[idx])[::-1][: self.cfg.archive_size]]
        self.archive_tokens, self.archive_scores = toks[order], sc[order]

    def _mutants(self, parents: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Random offspring; returns ``(kids, parent_index, pos, tok)`` (pos/tok valid for single edits)."""
        pidx = self.rng.integers(0, len(parents), size=k)
        kids = parents[pidx].copy()
        pos = np.zeros(k, dtype=np.int64)
        tok = np.zeros(k, dtype=np.int64)
        if self.cfg.single_token_mutations:
            pos = self.rng.integers(0, self.cfg.seq_len, size=k)
            tok = (kids[np.arange(k), pos] + self.rng.integers(1, self.pool.size, size=k)) % self.pool.size  # never a no-op
            kids[np.arange(k), pos] = tok
            return kids, pidx, pos, tok
        n_mut = self.rng.integers(1, self.cfg.max_mutations + 1, size=k)
        for i in range(k):
            p = self.rng.choice(self.cfg.seq_len, size=int(n_mut[i]), replace=False)
            kids[i, p] = self.rng.integers(0, self.pool.size, size=int(n_mut[i]))
        return kids, pidx, pos, tok

    # -- main loop -----------------------------------------------------------
    def run(self, callback: Callable[[dict], None] | None = None) -> pd.DataFrame:
        cfg = self.cfg
        t0 = perf_counter()
        pop = (
            self.rng.integers(0, self.pool.size, size=(cfg.mu, cfg.seq_len))
            if self._init is None
            else np.asarray(self._init, dtype=np.int64)[: cfg.mu]
        )
        if len(pop) < cfg.mu:  # top up a short warm-start population with random sequences
            pop = np.concatenate([pop, self.rng.integers(0, self.pool.size, size=(cfg.mu - len(pop), cfg.seq_len))])
        score, cut, corr = self._evaluate(pop)
        self._update_archive(pop, score)
        self._log(pop, score, cut, corr, t0, callback)
        while self.evaluations < cfg.budget:
            self.generation += 1
            k_model = int(round(cfg.model_fraction * cfg.lam)) if self.proposal is not None else 0
            k_learned = int(round(cfg.learned_fraction * cfg.lam)) if self.mutation_proposal is not None else 0
            kids_r, pidx, pos, tok = self._mutants(pop, cfg.lam - k_model - k_learned)
            kids = [kids_r]
            if k_learned:
                lp = self.rng.integers(0, len(pop), size=k_learned)
                kids_l, pos_l, tok_l = self.mutation_proposal.propose(pop[lp])
                kids.append(np.asarray(kids_l, dtype=np.int64))
                pidx, pos, tok = np.concatenate([pidx, lp]), np.concatenate([pos, pos_l]), np.concatenate([tok, tok_l])
            if k_model:
                kids.append(np.asarray(self.proposal(k_model), dtype=np.int64))
            kids = np.concatenate(kids)
            ks, kc, kcorr = self._evaluate(kids)
            if self.mutation_proposal is not None:
                n_edit = len(pidx)
                self.mutation_proposal.record(pop[pidx], pos, tok, ks[:n_edit] - score[pidx])
                if self.generation % cfg.learned_update_every == 0:
                    self._last_scorer_loss = self.mutation_proposal.update()
            self._update_archive(kids, ks)
            if self.proposal is not None and hasattr(self.proposal, "update") and self.generation % cfg.update_every == 0:
                self.proposal.update(self.archive_tokens)
            allp = np.concatenate([pop, kids])
            alls, allc, allcorr = np.concatenate([score, ks]), np.concatenate([cut, kc]), np.concatenate([corr, kcorr])
            order = np.argsort(alls)[::-1][: cfg.mu]
            pop, score, cut, corr = allp[order], alls[order], allc[order], allcorr[order]
            if self.generation % cfg.log_every == 0 or self.evaluations >= cfg.budget:
                self._log(pop, score, cut, corr, t0, callback)
        self.population, self.population_scores = pop, score
        return pd.DataFrame(self.records)

    def _log(self, pop, score, cut, corr, t0, callback) -> None:
        bk = self.inst.best_known()
        counts = self.pool.gate_counts(pop[0])
        rec = {
            "generation": self.generation,
            "evaluations": self.evaluations,
            "best_score": float(score[0]),
            "best_cut": float(cut[0]),
            "pop_mean_cut": float(cut.mean()),
            "median_abs_corr": float(np.median(np.abs(corr[0, : self.inst.m]))),
            "min_abs_corr": float(np.min(np.abs(corr[0, : self.inst.m]))),
            "two_qubit": counts["two_qubit"],
            "non_clifford": counts["non_clifford"],
            "scorer_loss": self._last_scorer_loss,
            "seconds": perf_counter() - t0,
        }
        if bk:
            rec["ratio_best"] = rec["best_cut"] / bk
            rec["ratio_pop_mean"] = rec["pop_mean_cut"] / bk
        self.records.append(rec)
        if callback:
            callback(rec)

    # -- results -------------------------------------------------------------
    def elites(self, top: int) -> np.ndarray:
        """The ``top`` best sequences ever evaluated (by reward), from the archive."""
        scores, _ = self.reward(np.stack(self.database.corr).astype(np.float64))
        idx = np.argsort(scores)[::-1][:top]
        return np.stack([self.database.tokens[i] for i in idx])

    def best_with_local_search(self) -> tuple[float, float]:
        """(best cut, best cut after one bit-swap pass)."""
        x = decode_signs(self.best["corr"][: self.inst.m])
        return self.best["cut"], self.inst.cut_value(one_pass_bit_swap(self.inst, x))
