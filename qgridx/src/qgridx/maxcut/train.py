"""Project 2, Arm B: GQE+PCE training loop for MaxCut, trimmed from
train.py's train_regime_a. Reuses the generic (problem-agnostic) pieces
unchanged -- grpo_loss/dpo_ranking_loss/logit_matching_loss, ReplayBuffer,
DispersionTemperatureSchedule, GQEDecoderOnly -- and drops everything tied
to QUBOInstance/cuts/feasibility (power-grid concepts that don't exist for
MaxCut: every bitstring is feasible, there is no budget/domain-wall
constraint, and there is no cheap-vs-full decode split since
local_search_1bit_2bit is already fast enough to run on every sample,
matching protocol 1.4's single decode pipeline used both for the training
reward and the final evaluation).
"""
from dataclasses import dataclass

import numpy as np
import torch

from qgridx.generator.model import GQEDecoderOnly
from qgridx.maxcut.reward import reward_batch_maxcut
from qgridx.generator.train import DispersionTemperatureSchedule, ReplayBuffer, ReplayEntry, dpo_ranking_loss, grpo_loss, logit_matching_loss


@dataclass
class RegimeMaxCutConfig:
    max_evals: int = 10000
    group_size: int = 16
    n_iter: int = 4
    replay_batch_size: int = 64
    buffer_capacity: int = 2000
    lr: float = 3e-4
    checkpoints: tuple = (500, 2000, 5000, 10000)
    objective: str = "dpo"
    seed: int = 0


@dataclass
class RegimeMaxCutResult:
    best_cost_per_eval: dict
    best_tokens_per_eval: dict
    telemetry: list
    best_tokens: np.ndarray
    best_cost: float          # = -E_full (lower is better)
    best_E_full: float
    best_E_raw: float          # pre-local-search cut of the SAME best circuit
                                 # (protocol 1.4 step 6: report r_circuit too)


def train_gqe_maxcut(vocab, n: int, assignment, W: np.ndarray, edges: list, cfg: RegimeMaxCutConfig):
    torch.manual_seed(cfg.seed)
    model = GQEDecoderOnly(vocab.vocab_size, vocab.max_len)
    init_model = GQEDecoderOnly(vocab.vocab_size, vocab.max_len)
    init_model.load_state_dict(model.state_dict())
    for p in init_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    temp_schedule = DispersionTemperatureSchedule()
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    n_evals = 0
    iteration = 0
    best_cost = np.inf
    best_tokens = None
    best_E_raw = None
    best_cost_per_eval = {}
    best_tokens_per_eval = {}
    telemetry = []
    ckpt_sorted = sorted(cfg.checkpoints)
    ckpt_ptr = 0

    while n_evals < max(cfg.checkpoints):
        tokens, logp_old = model.sample(cfg.group_size, vocab.BOS_ID, vocab.EOS_ID, vocab.min_gates,
                                         temperature=temp_schedule.temperature, generator=gen)
        costs, rewards, raw_cuts = reward_batch_maxcut(tokens, vocab, n, assignment, W, edges)
        n_evals += cfg.group_size
        if costs.min() < best_cost:
            best_cost = float(costs.min())
            best_tokens = tokens[costs.argmin()].numpy()
            best_E_raw = float(raw_cuts[costs.argmin()])

        finite = costs[np.isfinite(costs)]
        raw_dispersion = float(finite.std()) if len(finite) > 1 else 0.0
        scale = max(abs(finite.mean()), 1e-6) if len(finite) else 1.0
        norm_dispersion = raw_dispersion / scale
        temp, action = temp_schedule.update(norm_dispersion, iteration)

        logp_old_seq = logp_old.sum(dim=1)
        buffer.add([ReplayEntry(None, tokens[i].numpy(), rewards[i], logp_old_seq[i].item(), iteration)
                    for i in range(cfg.group_size)])

        losses, mean_advs, clip_fracs = [], [], []
        for _ in range(cfg.n_iter):
            entries = buffer.sample(cfg.replay_batch_size, rng)
            tok_b = torch.tensor(np.stack([e.tokens for e in entries]), dtype=torch.long)
            logp_old_b = torch.tensor([e.logp_old for e in entries], dtype=torch.float64)
            reward_b = torch.tensor([e.reward for e in entries], dtype=torch.float64)
            group_b = torch.tensor([e.iteration for e in entries], dtype=torch.long)

            logp_new_b = model.sequence_logprob(tok_b, vocab.EOS_ID)
            if cfg.objective == "grpo":
                with torch.no_grad():
                    logp_init_b = init_model.sequence_logprob(tok_b, vocab.EOS_ID)
                loss, mean_adv, clip_frac = grpo_loss(logp_new_b, logp_old_b, reward_b, group_b, logp_init_b)
            elif cfg.objective == "dpo":
                loss = dpo_ranking_loss(logp_new_b, reward_b, group_b)
                mean_adv, clip_frac = 0.0, 0.0
            else:
                loss = logit_matching_loss(logp_new_b, reward_b, group_b)
                mean_adv, clip_frac = 0.0, 0.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            mean_advs.append(mean_adv)
            clip_fracs.append(clip_frac)

        telemetry.append(dict(iteration=iteration, n_evals=n_evals, best_cost=best_cost,
                               batch_min_cost=float(costs.min()), dispersion=norm_dispersion,
                               temperature=temp, temp_action=action, loss=float(np.mean(losses))))

        while ckpt_ptr < len(ckpt_sorted) and n_evals >= ckpt_sorted[ckpt_ptr]:
            best_cost_per_eval[ckpt_sorted[ckpt_ptr]] = best_cost
            best_tokens_per_eval[ckpt_sorted[ckpt_ptr]] = best_tokens.copy()
            ckpt_ptr += 1
        iteration += 1

    return RegimeMaxCutResult(best_cost_per_eval=best_cost_per_eval, best_tokens_per_eval=best_tokens_per_eval,
                               telemetry=telemetry, best_tokens=best_tokens, best_cost=best_cost,
                               best_E_full=-best_cost, best_E_raw=best_E_raw)
