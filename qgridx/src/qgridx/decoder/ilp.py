"""B2: MAP decoder as a projection ILP (framework Sec 5 / P9, P3). Given
log-odds weights w_i ~ mu_i/sigma_i^2, solve

    x* = argmax_{x in F} sum_i w_i x_i

where F is defined by: budget, domain-wall chain validity (linear, x_{b,l+1}
<= x_{b,l}), mutex groups (linear, sum of the group <= 1), and an optional
list of extra linear feasibility cuts (aT x <= b) -- the B4 cut-list
interface, empty by default. Solved via scipy.optimize.milp (HiGHS), which
needs no new dependency and is exact for these small (m<=25) binary programs.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from qgridx.problems.siting import QUBOInstance


@dataclass
class Cut:
    a: np.ndarray   # (m,)
    b: float        # aT x <= b


def _domain_wall_constraints(inst: QUBOInstance):
    """x_{b,l+1} - x_{b,l} <= 0 for each bus's chain, l=1..L-1 (0-indexed
    consecutive positions within the bus's slice)."""
    rows = []
    for (s, e) in inst.bus_bit_slices:
        for pos in range(s, e - 1):
            row = np.zeros(inst.m)
            row[pos + 1] = 1
            row[pos] = -1
            rows.append(row)
    return rows


def map_decode_ilp(w: np.ndarray, inst: QUBOInstance, mutex_groups=None, cuts=None):
    """w: (m,) log-odds weights. mutex_groups: list of index-lists, each with
    sum(x[group]) <= 1. cuts: list of Cut. Returns (x*, solve_time_s, status)."""
    import time
    m = inst.m
    A_rows, b_vals = [], []

    A_rows.append(inst.budget_weights.copy())
    b_vals.append(inst.budget_cap)

    for row in _domain_wall_constraints(inst):
        A_rows.append(row)
        b_vals.append(0.0)

    for group in (mutex_groups or []):
        row = np.zeros(m)
        row[group] = 1.0
        A_rows.append(row)
        b_vals.append(1.0)

    for cut in (cuts or []):
        A_rows.append(cut.a)
        b_vals.append(cut.b)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    bounds = Bounds(0, 1)
    integrality = np.ones(m)

    t0 = time.time()
    res = milp(c=-w, constraints=constraints, bounds=bounds, integrality=integrality)
    dt = time.time() - t0

    if not res.success:
        return None, dt, res.status
    x = np.round(res.x).astype(int)
    return x, dt, res.status
