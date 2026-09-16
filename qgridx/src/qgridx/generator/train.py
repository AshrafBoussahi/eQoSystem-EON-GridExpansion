"""Sprint 9, A2: the Nakaji-style training machinery -- GRPO (group-normalized
advantages, clipped importance ratios), a logit-matching ablation arm, a FIFO
replay buffer, and a dispersion-triggered temperature schedule.

Documented approximation (stated once, applies to this whole module): these
are constructed directly from the plan's own natural-language description of
Nakaji et al.'s mechanisms, not verified against the paper's exact equations
(not available in this session) -- the same transparent-approximation
practice this project has used since Sprint 5's DPO noise-gate proxy. GRPO
here is standard group-relative PPO: per-context (or per-instance) group of
G samples, advantages normalized within the group, a PPO-style clipped
surrogate objective against the *sampling* (behavior) policy's log-probs
(enabling multiple gradient steps per batch of environment interactions --
the replay buffer's whole point), plus a small KL-to-frozen-initial-policy
term standing in for the paper's "frozen-init reference," since there is no
pretrained/cloned checkpoint to preserve here (training is from scratch).
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

CLIP_EPS = 0.2
KL_INIT_WEIGHT = 0.01


def grpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, rewards: torch.Tensor,
              group_ids: torch.Tensor, logp_init: torch.Tensor = None):
    """logp_new/logp_old/logp_init: (N,) sequence log-probs under current/
    behavior(sampling-time)/frozen-initial policy. rewards: (N,). group_ids:
    (N,) int, which group (context or restart) each sample belongs to --
    advantages are normalized within-group (GRPO's defining move, replacing
    a learned value/critic baseline with the group's own empirical stats).
    Returns (loss, mean_advantage_abs, mean_clip_fraction)."""
    advantages = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        mask = group_ids == g
        r = rewards[mask]
        std = r.std() if mask.sum() > 1 else torch.tensor(1.0)
        advantages[mask] = (r - r.mean()) / (std + 1e-6)

    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
    clip_fraction = (unclipped != clipped).float().mean()
    surrogate = -torch.min(unclipped, clipped).mean()

    kl_term = torch.tensor(0.0)
    if logp_init is not None:
        kl_term = (logp_new - logp_init).mean() * KL_INIT_WEIGHT

    loss = surrogate + kl_term
    return loss, advantages.abs().mean().item(), clip_fraction.item()


def dpo_ranking_loss(logp_new: torch.Tensor, rewards: torch.Tensor, group_ids: torch.Tensor,
                      min_gap: float = 0.02):
    """B3 ablation arm: Sprint 7's listwise PL + CPO-anchor recipe (arm3_dpo),
    ported to a per-instance GQE group instead of per-context M=16 samples --
    same noise-gated listwise ranking + best-sample NLL anchor, applied within
    each group (one group = one fresh sampling round, matching GRPO's own
    per-iteration grouping so the two objectives are compared on identical
    sampling batches, per the plan's "identical everything else" ask)."""
    total_loss = torch.tensor(0.0)
    n_groups = 0
    for g in torch.unique(group_ids):
        mask = g == group_ids
        r = rewards[mask]
        lp = logp_new[mask]
        if len(r) < 2:
            continue
        ranking = torch.argsort(r, descending=True)  # best (highest reward) first
        r_sorted = r[ranking]
        lp_sorted = lp[ranking]
        scale = max(r_sorted.abs().max().item(), 1e-6)
        admitted = [(abs((r_sorted[k] - r_sorted[k + 1]).item()) / scale) > min_gap
                    for k in range(len(r_sorted) - 1)]
        k_max = 0
        for ok in admitted:
            if not ok:
                break
            k_max += 1
        k_max = max(k_max, 1)

        pl_loss = torch.tensor(0.0)
        for k in range(k_max):
            remaining = lp_sorted[k:]
            pl_loss = pl_loss - (lp_sorted[k] - torch.logsumexp(remaining, dim=0))
        cpo_loss = -lp_sorted[0]
        total_loss = total_loss + pl_loss + cpo_loss
        n_groups += 1
    return total_loss / max(n_groups, 1)


def logit_matching_loss(logp_new: torch.Tensor, rewards: torch.Tensor, group_ids: torch.Tensor,
                         cost_offset: float = None):
    """Ablation arm (B3): direct reward-weighted log-likelihood ("logit
    matching" per the plan), with a cost offset for stability (their
    Appendix B.1) -- subtract a running baseline (here: the group mean, same
    data GRPO already computes) before weighting, avoiding the
    all-positive-reward degenerate case where every sample's log-prob is
    pushed up regardless of relative quality."""
    weights = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        mask = group_ids == g
        r = rewards[mask]
        offset = cost_offset if cost_offset is not None else r.mean()
        weights[mask] = r - offset
    return -(weights.detach() * logp_new).mean()


@dataclass
class ReplayEntry:
    context_key: object   # None for Regime A; instance index/id for Regime B
    tokens: np.ndarray
    reward: float
    logp_old: float
    iteration: int


class ReplayBuffer:
    """FIFO replay buffer (Sprint 9 A2): evaluations reused across multiple
    gradient updates instead of discarded, directly addressing the
    evaluation-budget economics the plan calls out -- a fresh batch of
    N_sample circuit evaluations buys N_iter gradient steps, not one."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[ReplayEntry] = deque(maxlen=capacity)

    def add(self, entries):
        self.buffer.extend(entries)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.buffer), size=min(batch_size, len(self.buffer)))
        return [self.buffer[i] for i in idx]


class DispersionTemperatureSchedule:
    """Sprint 9 A2: dispersion-triggered beta (here, sampling temperature)
    schedule (plan's Appendix A.3 analog). Tracks a smoothed measure of
    batch reward dispersion (std of decoded costs); if dispersion collapses
    below a low-water mark (samples have converged/homogenized -- GRPO's
    within-group advantage signal is starving), the schedule *re-broadens*
    (raises temperature) to regain exploration; if dispersion is comfortably
    high (samples are still spread, meaningful advantage signal exists), it
    *sharpens* (lowers temperature) toward exploitation. Multiplicative
    step alpha=0.02 per the plan; low/high water marks are the "tau" the
    plan asks be tuned briefly on the smoke test."""

    def __init__(self, init_temp: float = 1.0, alpha: float = 0.02,
                 low_water: float = 0.15, high_water: float = 0.6,
                 min_temp: float = 0.3, max_temp: float = 2.5):
        self.temperature = init_temp
        self.alpha = alpha
        self.low_water = low_water
        self.high_water = high_water
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.history = []  # (step, dispersion, temperature, action)

    def update(self, dispersion: float, step: int):
        action = "hold"
        if dispersion < self.low_water:
            self.temperature = min(self.max_temp, self.temperature * (1 + self.alpha))
            action = "broaden"
        elif dispersion > self.high_water:
            self.temperature = max(self.min_temp, self.temperature * (1 - self.alpha))
            action = "sharpen"
        self.history.append((step, dispersion, self.temperature, action))
        return self.temperature, action

    def fired_broaden_fraction(self):
        if not self.history:
            return 0.0
        return sum(1 for h in self.history if h[3] == "broaden") / len(self.history)

    def fired_sharpen_fraction(self):
        if not self.history:
            return 0.0
        return sum(1 for h in self.history if h[3] == "sharpen") / len(self.history)


@dataclass
class RegimeAConfig:
    max_evals: int = 10000
    group_size: int = 16          # circuits sampled per fresh training iteration (N_sample)
    n_iter: int = 4               # gradient steps per fresh batch, reusing the buffer (N_iter)
    replay_batch_size: int = 64
    buffer_capacity: int = 2000
    lr: float = 3e-4
    checkpoints: tuple = (200, 1000, 2400, 10000)
    objective: str = "dpo"        # "dpo" (D33 default) | "grpo" | "logit_matching"
    seed: int = 0
    use_shot_noise: bool = False  # Sprint 11 A1 (PR-41, D32): hardware-honest training reward
    n_shots: int = 1024
    margin_aware_lambda: float | None = None  # Sprint 2 WS-1, arm N4: if set, trains on EXACT
    # correlators (use_shot_noise must be False) but with reward_batch_single_instance_margin_aware's
    # flip-probability-penalized reward instead of the plain cheap-decode-cost reward. Mutually
    # exclusive with use_shot_noise (N4 is an exact-training arm by design, see reward.py docstring).
    radius: int = 1  # Sprint X.12: joint local-search neighborhood radius (both training
    # reward and any full-portfolio verification use this). Default 1 matches every prior
    # result in this project. Diagnosed directly: on some near-degenerate-optimum instances
    # (true optimum flat/near-zero, many close local competitors), radius=1 traps local
    # search at a fixed suboptimal point regardless of how good the circuit's raw signal is
    # -- verified by running radius=1 vs radius=2 local search from 100 random starting
    # points on affected instances (radius=1: 4-12/100 reach the true optimum; radius=2:
    # 100/100). Widening the radius is a decoder-side fix, orthogonal to circuit training.
    # NOTE: has NO effect when inst.B>10 -- joint_neighborhood_search ignores radius
    # entirely there and falls back to plain coordinate descent (the exhaustive
    # (2*radius+1)^B combo search this parameter controls is combinatorially infeasible
    # past B~10). Use large_b_boost for that regime instead.
    large_b_boost: bool = False  # Sprint X.15: the B>10 analogue of `radius` (see
    # qgridx/decoder/repair.py's _pairwise_boost_search docstring). Diagnosed directly on the
    # m=252 (B=28) PEGASE-1354 scale-up: plain coordinate descent reaches the true optimum
    # 19/30 times from random restarts; the sampled pairwise-move boost reaches it 30/30
    # across every tested instance. Default False matches every prior result in this
    # project (including the original, lower m=252 run this flag was added to improve).
    verify_best_with_full_decode: bool = False  # Sprint 14 A3 (PR-51 fix): re-decode a new
    # cheap-decode personal-best via the full (naive + MAP/ILP) decode_portfolio path before
    # accepting it as best_tokens/best_cost -- fixes the discovered IEEE-57-specific mechanism
    # where a circuit whose full decode already equals the oracle optimum got replaced by a later
    # circuit with a lower *cheap*-decode cost but a worse full-decode cost (non-monotonic
    # exact-match across checkpoints). Only re-decodes on the rare "new personal best" event, not
    # every sample, so the extra cost is one full decode per improvement, not per rollout.


@dataclass
class RegimeAResult:
    best_cost_per_eval: dict           # {n_evals_checkpoint: best_cost_so_far}
    best_tokens_per_eval: dict         # {n_evals_checkpoint: best_tokens_so_far} -- the actual
                                        # circuit found by that checkpoint, NOT just its cost.
                                        # Added after discovering that scripts decoding a single
                                        # final `best_tokens` against every checkpoint label
                                        # produced an artifactually flat curve (Sprint 9 B1 /
                                        # Sprint 11 A1 both had this bug -- see the follow-up
                                        # investigation after Sprint 11's report).
    telemetry: list                    # list of dicts per training iteration
    best_tokens: np.ndarray
    best_cost: float
    temp_schedule: DispersionTemperatureSchedule
    buffer: ReplayBuffer


def train_regime_a(inst, vocab, unitary_table, n, assignment, cuts, cfg: RegimeAConfig):
    """Sprint 9, A3/B1: per-instance GQE from scratch. No circuit-parameter
    gradients anywhere (banned by definition) -- every gradient step updates
    only the token-emission policy's weights; circuit angles are the fixed
    grid values baked into `unitary_table`."""
    from qgridx.generator.model import GQEDecoderOnly
    from qgridx.generator.reward import reward_batch_single_instance, reward_batch_single_instance_margin_aware, reward_batch_single_instance_noisy

    assert not (cfg.use_shot_noise and cfg.margin_aware_lambda is not None), \
        "margin_aware_lambda (arm N4) trains on exact correlators; mutually exclusive with use_shot_noise"

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
    best_cost_full = np.inf      # Sprint 14 A3 (PR-51 fix): see cfg.verify_best_with_full_decode
    best_tokens_full = None
    best_cost_per_eval = {}
    best_tokens_per_eval = {}
    telemetry = []
    ckpt_sorted = sorted(cfg.checkpoints)
    ckpt_ptr = 0

    dispersions = []
    while n_evals < max(cfg.checkpoints):
        tokens, logp_old = model.sample(cfg.group_size, vocab.BOS_ID, vocab.EOS_ID, vocab.min_gates,
                                         temperature=temp_schedule.temperature, generator=gen)
        if cfg.use_shot_noise:
            costs, rewards = reward_batch_single_instance_noisy(
                tokens, unitary_table, n, assignment, inst, cuts, N_shots=cfg.n_shots,
                seed=cfg.seed * 100003 + iteration)
        elif cfg.margin_aware_lambda is not None:
            costs, rewards, _ = reward_batch_single_instance_margin_aware(
                tokens, unitary_table, n, assignment, inst, cuts, lam=cfg.margin_aware_lambda)
        else:
            costs, rewards = reward_batch_single_instance(tokens, unitary_table, n, assignment, inst, cuts,
                                                            radius=cfg.radius, large_b_boost=cfg.large_b_boost)
        n_evals += cfg.group_size
        if costs.min() < best_cost:
            best_cost = costs.min()
            best_tokens = tokens[costs.argmin()].numpy()
            if cfg.verify_best_with_full_decode:
                # re-decode this new cheap-best candidate under the full (naive + MAP/ILP) path and
                # track it separately -- fixes the discovered mismatch where a later circuit with a
                # lower *cheap*-decode cost silently displaced an earlier one that was better under
                # full decode (see cfg field docstring). best_tokens_full/best_cost_full are reported
                # at checkpoint time instead of the cheap-tracked best_tokens/best_cost. Sprint 2 WS-1:
                # extended to the noise-aware arms too -- their deployed decoder is ALSO noisy, so the
                # re-verification must use the same noisy correlators, not the exact ones.
                if cfg.use_shot_noise:
                    full_costs, _ = reward_batch_single_instance_noisy(
                        tokens[costs.argmin():costs.argmin() + 1], unitary_table, n, assignment, inst, cuts,
                        N_shots=cfg.n_shots, seed=cfg.seed * 100003 + iteration, use_full_portfolio=True)
                else:
                    full_costs, _ = reward_batch_single_instance(
                        tokens[costs.argmin():costs.argmin() + 1], unitary_table, n, assignment, inst, cuts,
                        use_full_portfolio=True, radius=cfg.radius, large_b_boost=cfg.large_b_boost)
                if full_costs[0] < best_cost_full:
                    best_cost_full = float(full_costs[0])
                    best_tokens_full = best_tokens.copy()

        finite = costs[np.isfinite(costs)]
        raw_dispersion = float(finite.std()) if len(finite) > 1 else 0.0
        scale = max(abs(finite.mean()), 1e-6) if len(finite) else 1.0
        norm_dispersion = raw_dispersion / scale
        dispersions.append(norm_dispersion)
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
                               temperature=temp, temp_action=action, loss=float(np.mean(losses)),
                               mean_advantage=float(np.mean(mean_advs)), clip_fraction=float(np.mean(clip_fracs))))

        while ckpt_ptr < len(ckpt_sorted) and n_evals >= ckpt_sorted[ckpt_ptr]:
            if cfg.verify_best_with_full_decode and best_tokens_full is not None:
                best_cost_per_eval[ckpt_sorted[ckpt_ptr]] = best_cost_full
                best_tokens_per_eval[ckpt_sorted[ckpt_ptr]] = best_tokens_full.copy()
            else:
                best_cost_per_eval[ckpt_sorted[ckpt_ptr]] = best_cost
                best_tokens_per_eval[ckpt_sorted[ckpt_ptr]] = best_tokens.copy()
            ckpt_ptr += 1
        iteration += 1

    report_tokens = best_tokens_full if (cfg.verify_best_with_full_decode and best_tokens_full is not None) else best_tokens
    report_cost = best_cost_full if (cfg.verify_best_with_full_decode and best_tokens_full is not None) else best_cost
    return RegimeAResult(best_cost_per_eval=best_cost_per_eval, best_tokens_per_eval=best_tokens_per_eval,
                          telemetry=telemetry, best_tokens=report_tokens, best_cost=report_cost,
                          temp_schedule=temp_schedule, buffer=buffer)
