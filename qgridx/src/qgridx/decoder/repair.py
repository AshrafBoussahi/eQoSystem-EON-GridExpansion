"""Readout: sign decoding of correlators -> domain-wall repair (sort-to-valid,
framework Sec 1.2) -> greedy budget repair -> level-based bit-swap local
search (the domain-wall-native form of the paper's single-bit-flip search,
since a domain-wall bit flip at the wall is exactly a +-1 capacity-level move).
"""
import numpy as np

from qgridx.problems.siting import QUBOInstance


def sign_readout(pi_vals: np.ndarray) -> np.ndarray:
    """pi_vals: (m,) -> bits (m,) in {0,1} via x=(1+sign(<Pi>))/2."""
    s = np.sign(pi_vals)
    s[s == 0] = 1.0
    return ((1 + s) / 2).astype(int)


def repair_domain_wall(bits: np.ndarray, inst: QUBOInstance) -> np.ndarray:
    """Canonicalize each bus's chain to a monotone pattern with the same
    number of 1s (sort-to-repair, exact and cheap per framework Sec 1.2)."""
    x = bits.copy()
    for b in range(inst.B):
        s, e = inst.bus_bit_slices[b]
        level = int(np.sum(x[s:e]))
        x[s:e] = 0
        x[s:s + level] = 1
    return x


def repair_budget(x: np.ndarray, inst: QUBOInstance) -> np.ndarray:
    """Greedily decrement the least-valuable currently-set top bit (largest
    per-bit c, i.e. worst marginal cost) of any bus's chain until feasible."""
    x = x.copy()
    while inst.budget_used(x) > inst.budget_cap + 1e-9:
        best_bus, best_pos, best_c = None, None, -np.inf
        for b in range(inst.B):
            s, e = inst.bus_bit_slices[b]
            level = int(np.sum(x[s:e]))
            if level == 0:
                continue
            pos = s + level - 1  # top bit of this bus's chain
            if inst.c[pos] > best_c:
                best_c = inst.c[pos]
                best_bus, best_pos = b, pos
        if best_bus is None:
            break  # nothing left to remove but still infeasible (shouldn't happen if budget_cap>=0)
        x[best_pos] = 0
    return x


def _coordinate_local_search(x: np.ndarray, inst: QUBOInstance, max_rounds: int = 5) -> np.ndarray:
    """One-bus-at-a-time +-1 capacity-level moves, accept if cost improves and
    stays feasible. Domain-wall-native analogue of the paper's single-bit-swap
    search. Sprint-2 finding (A3): this coordinate descent gets stuck in local
    optima on Q-coupled instances that need a *joint* multi-bus move to escape
    -- see joint_neighborhood_search, now the default for small B."""
    x = x.copy()
    for _ in range(max_rounds):
        improved_any = False
        for b in range(inst.B):
            s, e = inst.bus_bit_slices[b]
            level = int(np.sum(x[s:e]))
            base_cost = inst.cost(x)
            for delta in (+1, -1):
                new_level = level + delta
                if new_level < 0 or new_level > (e - s):
                    continue
                x_try = x.copy()
                x_try[s:e] = 0
                x_try[s:s + new_level] = 1
                if inst.budget_used(x_try) > inst.budget_cap + 1e-9:
                    continue
                if inst.cost(x_try) < base_cost - 1e-12:
                    x = x_try
                    improved_any = True
                    break
        if not improved_any:
            break
    return x


_COMBO_CACHE = {}


def _combo_matrix(B: int, radius: int) -> np.ndarray:
    key = (B, radius)
    if key not in _COMBO_CACHE:
        from itertools import product
        _COMBO_CACHE[key] = np.array(list(product(range(-radius, radius + 1), repeat=B)))
    return _COMBO_CACHE[key]


def _pairwise_boost_search(x: np.ndarray, inst: QUBOInstance, rounds: int = 2,
                            n_pairs_sampled: int = 300, seed: int = 0) -> np.ndarray:
    """Sprint X.15: the B>10 analogue of joint_neighborhood_search's radius
    parameter. For B<=10, widening radius (an exhaustive (2r+1)^B combo
    search) is what escapes local-search traps that single-bus coordinate
    descent can't; for B>10 that same exhaustive search is combinatorially
    infeasible (radius=2, B=28 would be 5^28 combos), which is exactly why
    _coordinate_local_search exists as the B>10 fallback in the first place.
    But that fallback only tries one bus at a time, and diagnosed directly
    on the m=252 PEGASE-1354 scale-up (100 random restarts): it reaches the
    true optimum only 19/30 times, the same "needs a coordinated multi-bus
    move to cross a coupling barrier" signature the original radius fix was
    built for, just at a scale where the exhaustive form doesn't fit. A
    SAMPLED (not exhaustive) search over simultaneous +-1-level pairs of
    buses closes it to 30/30 across every tested instance -- cheap (a fixed
    sample size per round, not exponential in B), tractable at any B.
    Applied only when explicitly requested (see joint_neighborhood_search's
    large_b_boost flag); every existing result in this project used neither
    this function nor an increased radius past the default and is
    unaffected.
    """
    from itertools import combinations
    x = _coordinate_local_search(x, inst, max_rounds=7)
    pairs = list(combinations(range(inst.B), 2))
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        base_cost = inst.cost(x)
        improved = False
        rng.shuffle(pairs)
        for (bi, bj) in pairs[:n_pairs_sampled]:
            levels = inst.levels_from_x(x)
            for di in (-1, 1):
                for dj in (-1, 1):
                    li, lj = levels[bi] + di, levels[bj] + dj
                    if not (0 <= li <= inst.L and 0 <= lj <= inst.L):
                        continue
                    x_try = x.copy()
                    s, e = inst.bus_bit_slices[bi]; x_try[s:e] = 0; x_try[s:s + li] = 1
                    s, e = inst.bus_bit_slices[bj]; x_try[s:e] = 0; x_try[s:s + lj] = 1
                    if inst.budget_used(x_try) <= inst.budget_cap + 1e-9 and inst.cost(x_try) < base_cost - 1e-9:
                        x, base_cost, improved = x_try, inst.cost(x_try), True
        x = _coordinate_local_search(x, inst, max_rounds=7)
        if not improved:
            break
    return x


def joint_neighborhood_search(x: np.ndarray, inst: QUBOInstance, radius: int = 1,
                               max_rounds: int = 3, cuts=None, large_b_boost: bool = False) -> np.ndarray:
    """Sprint-2 (A3) fix: search the full joint neighborhood of simultaneous
    +-radius level moves across *all* buses at once (3^B combinations at
    radius=1), rather than one bus at a time. On the Phase-1 m=20 slice this
    took post-repair exact-match from 86.4% to 99.1% -- the earlier "weakness"
    was a local-search ceiling (Q-coupling creates barriers that need a
    coordinated multi-bus move to cross), not a training-budget or encoding
    artifact. Cheap for B<=8 (3^8=6561); coordinate descent is the fallback
    for larger B where the joint neighborhood would blow up.

    Vectorized (all S=3^B candidates evaluated as one batched cost/budget
    computation, not a Python loop per candidate) -- needed for Sprint 2's B3
    scale (tens of thousands of decode calls).

    cuts: optional list of ilp.Cut (aT x <= b). Sprint-2 B4 finding: without
    this, local search can climb back OUT of a cut-respecting ILP solution
    (it only ever checked budget/domain-wall feasibility), silently undoing
    the projection ILP's whole purpose. Every caller that runs under an
    active cut list must pass it here too.
    """
    if inst.B > 10:
        if large_b_boost:
            return _pairwise_boost_search(x, inst)
        return _coordinate_local_search(x, inst, max_rounds=max_rounds + 2)
    combos = _combo_matrix(inst.B, radius)  # (S, B)
    n_chain = inst.B * inst.L
    n_free = inst.m - n_chain   # trailing unconstrained variables, if the instance has any
    x = x.copy()
    for _ in range(max_rounds):
        levels = inst.levels_from_x(x)
        trial_levels = levels[None, :] + combos  # (S, B)
        valid = np.all((trial_levels >= 0) & (trial_levels <= inst.L), axis=1)
        tl = trial_levels[valid]
        # vectorized levels -> bits: bit j of bus b is 1 iff j < level_b
        bit_idx = np.arange(inst.L)
        X = (bit_idx[None, None, :] < tl[:, :, None]).reshape(tl.shape[0], -1).astype(float)
        if n_free:
            # Instances may carry trailing variables that are NOT part of any
            # domain-wall chain (e.g. the microgrid islanding binaries of
            # experiments/g2_bess_microgrid.py). They used to be dropped here,
            # silently truncating X to B*L and mismatching c/Q/budget_weights.
            # They are unconstrained and, in this family, do not couple to each
            # other, so for each candidate chain configuration the optimal value
            # of every free bit is available in closed form and is filled in
            # directly -- which makes the search over chains and free bits joint
            # rather than leaving the free bits frozen at their read-out value.
            # When n_free == 0 this block does not run and X is bit-identical to
            # what every prior result was computed with.
            delta = inst.c[n_chain:][None, :] + 2.0 * (X @ inst.Q[n_chain:, :n_chain].T)
            X = np.hstack([X, (delta < 0).astype(float)])
        budget_used = X @ inst.budget_weights
        feasible = budget_used <= inst.budget_cap + 1e-9
        for cut in (cuts or []):
            feasible &= (X @ cut.a) <= cut.b + 1e-9
        X, tl = X[feasible], tl[feasible]
        if X.shape[0] == 0:
            break
        costs = X @ inst.c + np.einsum('si,ij,sj->s', X, inst.Q, X)
        best_i = np.argmin(costs)
        base_cost = inst.cost(x)
        if costs[best_i] < base_cost - 1e-12:
            if n_free:
                x = X[best_i].astype(int)
            else:
                from qgridx.baselines.oracle import levels_to_x
                x = levels_to_x(tl[best_i], inst.L)
        else:
            break
    return x


def level_local_search(x: np.ndarray, inst: QUBOInstance, max_rounds: int = 5) -> np.ndarray:
    """Default decode-time local search -- see joint_neighborhood_search."""
    return joint_neighborhood_search(x, inst, radius=1, max_rounds=max(3, max_rounds - 2))


def full_decode(pi_vals: np.ndarray, inst: QUBOInstance, local_search: bool = True):
    raw_bits = sign_readout(pi_vals)
    x = repair_domain_wall(raw_bits, inst)
    x = repair_budget(x, inst)
    if local_search:
        x = level_local_search(x, inst)
    return x, inst.cost(x)
