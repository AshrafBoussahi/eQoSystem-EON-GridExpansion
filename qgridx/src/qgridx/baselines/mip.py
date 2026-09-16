"""DOE Phase 3 hardening (Tier 1, PR-4x): a genuine MIP baseline via scipy's
bundled HiGHS solver (`scipy.optimize.milp`), which only solves LINEAR MIPs --
so the QUBO's quadratic term is linearized with the standard McCormick
envelope (one auxiliary binary y_ij = x_i*x_j per nonzero, off-diagonal Q
entry, which `instance_factory.build_instance`'s Q is sparse in -- at most
C(B,2) nonzeros, so this stays small even at B~20). Domain-wall monotonicity
(the per-bus leading-ones constraint) and the budget cap are both already
linear and are added directly, exactly matching QUBOInstance.is_feasible.
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from qgridx.problems.siting import QUBOInstance


def solve_mip(inst: QUBOInstance, time_limit_s: float = 30.0, cuts=None):
    """cuts: optional list of objects with .a (m,)/.b, each an extra linear
    feasibility cut (a^T x <= b) -- must match whatever cuts the comparison
    oracle/GQE run used (e.g. library_v2's mutex-slice configs), or the MIP
    and the reference solve different feasible sets and are not comparable.
    Returns (best_x or None, best_cost or None, wall_s, status_str)."""
    m = inst.m
    iu, ju = np.triu_indices(m, k=1)
    nz = inst.Q[iu, ju] != 0
    iu, ju, qvals = iu[nz], ju[nz], inst.Q[iu, ju][nz]
    n_pairs = len(qvals)

    n_vars = m + n_pairs
    c = np.zeros(n_vars)
    c[:m] = inst.c
    c[m:] = 2.0 * qvals  # x^T Q x = 2 * sum_{i<j} Q_ij x_i x_j (zero diagonal)

    constraints = []

    # budget: budget_weights^T x <= budget_cap
    a_budget = np.zeros(n_vars)
    a_budget[:m] = inst.budget_weights
    constraints.append(LinearConstraint(a_budget, -np.inf, inst.budget_cap))

    # domain-wall monotonicity per bus: x[s+k] <= x[s+k-1] for k=1..L-1
    mono_rows = []
    for (s, e) in inst.bus_bit_slices:
        for k in range(s + 1, e):
            row = np.zeros(n_vars)
            row[k] = 1.0
            row[k - 1] = -1.0
            mono_rows.append(row)
    if mono_rows:
        A_mono = np.array(mono_rows)
        constraints.append(LinearConstraint(A_mono, -np.inf, 0.0))

    # extra feasibility cuts (e.g. mutex slices), a^T x <= b, x-only
    for cut in (cuts or []):
        a_cut = np.zeros(n_vars)
        a_cut[:m] = cut.a
        constraints.append(LinearConstraint(a_cut, -np.inf, cut.b))

    # McCormick envelope per y_ij = x_i * x_j
    if n_pairs:
        A_mc1 = np.zeros((n_pairs, n_vars)); A_mc2 = np.zeros((n_pairs, n_vars)); A_mc3 = np.zeros((n_pairs, n_vars))
        for p, (i, j) in enumerate(zip(iu, ju)):
            A_mc1[p, m + p] = 1.0; A_mc1[p, i] = -1.0       # y_ij - x_i <= 0
            A_mc2[p, m + p] = 1.0; A_mc2[p, j] = -1.0       # y_ij - x_j <= 0
            A_mc3[p, m + p] = -1.0; A_mc3[p, i] = 1.0; A_mc3[p, j] = 1.0  # xi+xj-y <= 1  (y>=xi+xj-1)
        constraints.append(LinearConstraint(A_mc1, -np.inf, 0.0))
        constraints.append(LinearConstraint(A_mc2, -np.inf, 0.0))
        constraints.append(LinearConstraint(A_mc3, -np.inf, 1.0))

    bounds = Bounds(0.0, 1.0)
    integrality = np.ones(n_vars)

    t0 = time.time()
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
               options=dict(time_limit=time_limit_s))
    dt = time.time() - t0

    if res.x is None:
        return None, None, dt, ("timeout_no_incumbent" if not res.success else "infeasible")
    x = np.round(res.x[:m]).astype(int)
    status = "optimal" if res.status == 0 else "time_limit_incumbent"
    return x, float(inst.cost(x)), dt, status
