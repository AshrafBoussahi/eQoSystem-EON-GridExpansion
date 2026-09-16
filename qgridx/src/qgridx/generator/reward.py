"""Sprint 9: reward computation shared by Regime A and B. A sampled token
sequence -> executed state -> 3-family correlators -> cheap D16 decode (cost
= training reward) or full D2 portfolio decode (evaluation only, per the
project's standing cheap/expensive split).
"""
import numpy as np
import torch
from scipy.stats import norm

from qgridx.decoder.repair import joint_neighborhood_search, repair_budget, repair_domain_wall, sign_readout
from qgridx.encoding.loss import correlators
from qgridx.decoder.portfolio import _satisfies_cuts, decode_portfolio, project_via_ilp
from qgridx.encoding.shot_noise import all_string_shot_estimates, gather_assignment_estimates
from qgridx.generator.executor import execute_batch


def cheap_decode_cost(pi: np.ndarray, inst, cuts=None, radius: int = 1, large_b_boost: bool = False):
    cuts = cuts or []
    x = repair_domain_wall(sign_readout(pi.copy()), inst)
    x = repair_budget(x, inst)
    if cuts and not _satisfies_cuts(x, cuts):
        x = project_via_ilp(x, inst, cuts=cuts)
    x = joint_neighborhood_search(x, inst, radius=radius, cuts=cuts, large_b_boost=large_b_boost)
    return inst.cost(x) if inst.is_feasible(x) and _satisfies_cuts(x, cuts) else np.inf, x


def reward_batch_single_instance(tokens: torch.Tensor, unitary_table: torch.Tensor, n: int,
                                  assignment, inst, cuts=None, use_full_portfolio=False, radius: int = 1,
                                  large_b_boost: bool = False):
    """tokens: (batch, L), same instance for every row (Regime A: many
    restarts/samples of one instance). Returns (costs: np.ndarray (batch,),
    rewards: np.ndarray (batch,) = -cost with inf mapped to a large finite
    penalty so GRPO's group statistics stay finite). `radius`: joint local-
    search neighborhood radius (default 1, matching every prior result in
    this project); widening it lets the decoder escape local optima that a
    single simultaneous +-1-level move per site can't reach -- see
    RegimeAConfig.radius docstring for when this matters. `large_b_boost`:
    the B>10 analogue of `radius` (see _pairwise_boost_search's docstring) --
    `radius` has no effect at all when inst.B>10 (joint_neighborhood_search
    ignores it and falls back to plain coordinate descent), so this is the
    lever that actually matters for large-B configs like the m=252
    PEGASE-1354 scale-up."""
    with torch.no_grad():
        state = execute_batch(tokens, unitary_table, n)
        pi = correlators(state, assignment, n).numpy()
    costs = np.zeros(len(tokens))
    for i in range(len(tokens)):
        if use_full_portfolio:
            pr = decode_portfolio(pi[i], inst, cuts=cuts, radius=radius, large_b_boost=large_b_boost)
            costs[i] = pr.cost
        else:
            costs[i], _ = cheap_decode_cost(pi[i], inst, cuts, radius=radius, large_b_boost=large_b_boost)
    finite_costs = costs[np.isfinite(costs)]
    penalty = (finite_costs.max() + 10.0) if len(finite_costs) else 100.0
    safe_costs = np.where(np.isfinite(costs), costs, penalty)
    rewards = -safe_costs
    return costs, rewards


def reward_batch_single_instance_margin_aware(tokens: torch.Tensor, unitary_table: torch.Tensor, n: int,
                                               assignment, inst, cuts=None, N_ref: int = 1024,
                                               lam: float = 1.0, use_full_portfolio: bool = False):
    """Sprint 2, WS-1, arm N4: analytic margin-aware reward. Trains on EXACT
    statevector correlators (no shot noise anywhere in this function -- N4's
    whole point is that a differentiable, noiseless training signal can still
    be pushed toward noise-robust solutions if the reward itself accounts for
    deployment fragility). Reward = -(decoded cost + lam * sum_i stake_i *
    P_flip_i), where P_flip_i = Phi(-|mu_i| * sqrt(N_ref)) is the probability
    a real N_ref-shot readout would flip decision i's sign (normal
    approximation to the shot-noise estimator, the same flip-probability
    model the plan's D2/S1.C-family shot-budget reasoning already uses), and
    stake_i = |c_i| is the linear cost coefficient magnitude -- a small
    margin only matters if the decision it's attached to is expensive to get
    wrong. This is the "decoded cost plus a stake-weighted small-margin
    penalty" formulation (the plan's own stated equivalent to full
    flip-pattern propagation through the decoder, which is intractable at
    m~42-45 -- 2^m decode calls). `lam=1.0` is an untuned, first-pass default
    (documented, not dev-tuned -- time-budget decision, matching this
    project's established practice for other reused-but-unretuned
    hyperparameters, e.g. SPSA's a/c_pert)."""
    with torch.no_grad():
        state = execute_batch(tokens, unitary_table, n)
        pi = correlators(state, assignment, n).numpy()
    stake = np.abs(inst.c)
    costs = np.zeros(len(tokens))
    penalties = np.zeros(len(tokens))
    for i in range(len(tokens)):
        if use_full_portfolio:
            pr = decode_portfolio(pi[i], inst, cuts=cuts)
            costs[i] = pr.cost
        else:
            costs[i], _ = cheap_decode_cost(pi[i], inst, cuts)
        p_flip = norm.cdf(-np.abs(pi[i]) * np.sqrt(N_ref))
        penalties[i] = float(np.sum(stake * p_flip))
    finite_costs = costs[np.isfinite(costs)]
    penalty_ceiling = (finite_costs.max() + 10.0) if len(finite_costs) else 100.0
    safe_costs = np.where(np.isfinite(costs), costs, penalty_ceiling)
    rewards = -(safe_costs + lam * penalties)
    return costs, rewards, penalties


def reward_batch_single_instance_noisy(tokens: torch.Tensor, unitary_table: torch.Tensor, n: int,
                                        assignment, inst, cuts=None, N_shots: int = 1024, seed: int = 0,
                                        use_full_portfolio: bool = False):
    """Sprint 11, A1 (PR-41, D32): the hardware-honest reward. Same executor
    and cheap D16 decode path as `reward_batch_single_instance`, but the
    correlators feeding the decode are N=1024-shot noisy estimates (Sprint 3
    B1's shot-noise simulator), not the exact statevector expectation --
    removing the asymmetry where Sprint 9's training used exact correlators
    while the SPSA baseline it was compared against was trained under shot
    noise the whole time.

    `use_full_portfolio` (Sprint 2, WS-1): mirrors the exact version's flag,
    extending Sprint 14's `verify_best_with_full_decode` fix to the noise-
    aware training arms (N2/N3) -- the deployed decoder for a noise-aware
    arm is ALSO noisy (matching deployment), so the full-decode verification
    step for these arms must re-decode through `decode_portfolio` under the
    SAME noisy correlators, not the exact ones the original (exact-training)
    fix used."""
    k = len(assignment[0][0])
    with torch.no_grad():
        state = execute_batch(tokens, unitary_table, n)
        mu_all, sig_all, strings = all_string_shot_estimates(state, n, k, N_shots, seed=seed)
        mu, sigma = gather_assignment_estimates(mu_all, sig_all, strings, assignment)
        mu_np = mu.numpy()
        sigma_np = sigma.numpy()
    costs = np.zeros(len(tokens))
    for i in range(len(tokens)):
        if use_full_portfolio:
            pr = decode_portfolio(mu_np[i], inst, sigma=sigma_np[i], cuts=cuts)
            costs[i] = pr.cost
        else:
            costs[i], _ = cheap_decode_cost(mu_np[i], inst, cuts)
    finite_costs = costs[np.isfinite(costs)]
    penalty = (finite_costs.max() + 10.0) if len(finite_costs) else 100.0
    safe_costs = np.where(np.isfinite(costs), costs, penalty)
    rewards = -safe_costs
    return costs, rewards
