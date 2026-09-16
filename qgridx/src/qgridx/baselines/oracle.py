"""Brute-force oracle for QUBOInstance. Exploits the domain-wall structure:
instead of enumerating all 2^m raw bit strings, enumerate over the (L+1)^B
per-bus capacity levels directly (every level maps to exactly one valid,
canonical monotone bit pattern), which is exact and cheap for B <= ~10.
"""
from itertools import product

import numpy as np

from qgridx.problems.siting import QUBOInstance


def level_to_bits(level: int, L: int) -> np.ndarray:
    bits = np.zeros(L, dtype=int)
    bits[:level] = 1
    return bits


def levels_to_x(levels, L: int) -> np.ndarray:
    return np.concatenate([level_to_bits(lv, L) for lv in levels])


def brute_force_optimum(inst: QUBOInstance):
    """Returns (best_x, best_cost, n_feasible, n_enumerated)."""
    best_x, best_cost = None, np.inf
    n_feasible = 0
    n_enum = 0
    level_range = range(inst.L + 1)
    for levels in product(level_range, repeat=inst.B):
        n_enum += 1
        x = levels_to_x(levels, inst.L)
        budget_used = x @ inst.budget_weights
        if budget_used > inst.budget_cap + 1e-9:
            continue
        n_feasible += 1
        cost = inst.cost(x)
        if cost < best_cost:
            best_cost = cost
            best_x = x
    return best_x, best_cost, n_feasible, n_enum


def _all_level_combos(B: int, L: int) -> np.ndarray:
    """(enum, B) int array of every level combination, vectorized (no Python
    loop) -- needed for Sprint 4's v2 scale (enum up to ~2e6)."""
    grids = np.meshgrid(*([np.arange(L + 1)] * B), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def vectorized_oracle(inst: QUBOInstance, cuts=None, top_k: int = 1):
    """Fully vectorized brute-force oracle: builds every (L+1)^B level
    combination as one matrix, decodes to bits, filters by budget and any
    extra linear cuts (list of objects with .a, .b, aT x <= b), and returns
    the top_k lowest-cost feasible plans. Exact -- same guarantee as
    brute_force_optimum, just fast enough for B up to ~9-10.
    Returns: (top_x list, top_cost array, n_feasible, n_enum).
    """
    combos = _all_level_combos(inst.B, inst.L)  # (enum, B)
    n_enum = combos.shape[0]
    bit_idx = np.arange(inst.L)
    X = (bit_idx[None, None, :] < combos[:, :, None]).reshape(n_enum, -1).astype(np.float64)  # (enum, m)

    budget_used = X @ inst.budget_weights
    feasible = budget_used <= inst.budget_cap + 1e-9
    for cut in (cuts or []):
        feasible &= (X @ cut.a) <= cut.b + 1e-9
    n_feasible = int(feasible.sum())
    Xf = X[feasible]
    if Xf.shape[0] == 0:
        return [], np.array([]), 0, n_enum

    # Q is very sparse (at most C(B,2) nonzero congestion-coupling entries,
    # out of an m x m matrix) -- the dense einsum x^T Q x costs O(enum * m^2)
    # and dominated runtime at v2 scale (~5s/instance at m=36); exploit
    # sparsity instead: O(enum * nnz), typically 30-100x fewer operations.
    iu, ju = np.triu_indices_from(inst.Q, k=1)
    nz = inst.Q[iu, ju] != 0
    iu, ju, qvals = iu[nz], ju[nz], inst.Q[iu, ju][nz]
    quad = 2.0 * (Xf[:, iu] * Xf[:, ju] * qvals).sum(axis=1) if len(qvals) else 0.0
    costs = Xf @ inst.c + quad
    k = min(top_k, len(costs))
    order = np.argpartition(costs, k - 1)[:k]
    order = order[np.argsort(costs[order])]
    top_x = [Xf[i].astype(int) for i in order]
    top_cost = costs[order]
    return top_x, top_cost, n_feasible, n_enum
