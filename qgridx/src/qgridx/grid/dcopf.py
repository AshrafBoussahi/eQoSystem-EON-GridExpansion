"""Minimal DC-OPF solved as an LP, used only to derive a physics-informed
surrogate Hamiltonian H0 = (c, Q) per framework Sec 1.5 -- not a
power-flow-accuracy tool. Returns dispatch, system lambda, and per-bus LMPs
(lambda + congestion component from line-limit duals via PTDF), which is
exactly the standard DC-OPF LMP decomposition.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from qgridx.grid.cases import case14 as case14


@dataclass
class DCOPFResult:
    Pg: np.ndarray            # generator dispatch (MW), aligned with case.GEN rows
    total_cost: float
    lmp: np.ndarray           # (N_BUS,) nodal marginal price ($/MWh)
    lambda_sys: float         # system energy price (balance-constraint dual)
    line_flow: np.ndarray     # (n_branch,) MW
    mu_line: np.ndarray       # (n_branch,) dual of the (relaxed) line-limit constraints
    feasible: bool


def solve_dcopf(load_mw: np.ndarray, ptdf: np.ndarray | None = None, case=case14) -> DCOPFResult:
    """Solve a lossless DC-OPF LP for a given per-bus load vector (MW, length
    N_BUS). Minimizes linear generation cost subject to system balance,
    generator limits, and line-flow limits (both directions). `case`: the grid
    topology module (default IEEE-14; Sprint 8 Track B passes ieee_case30 for
    the cross-grid factory extension) -- same DC-OPF LP, different topology."""
    if ptdf is None:
        ptdf = case.compute_ptdf()
    n_gen = case.GEN.shape[0]
    gen_bus_idx = np.array([case.bus_index(b) for b in case.gen_buses()])
    Pmax = case.GEN[:, 1]
    Pmin = case.GEN[:, 2]
    cost = case.GEN[:, 3]
    rate = case.rate_a()
    n_branch = rate.shape[0]

    total_load = load_mw.sum()

    # Decision variables: Pg (n_gen,)
    # Net injection at bus i = sum_g Pg[g] * [gen_bus[g]==i] - load[i]
    # Map generator vector to bus-injection vector via selection matrix S (N_BUS x n_gen)
    S = np.zeros((case.N_BUS, n_gen))
    for g, bi in enumerate(gen_bus_idx):
        S[bi, g] = 1.0

    # line flow = PTDF @ (S @ Pg - load)   (slack absorbs the reference angle)
    # => line flow = (PTDF @ S) @ Pg - PTDF @ load
    PS = ptdf @ S                      # (n_branch, n_gen)
    const_flow = -ptdf @ load_mw       # (n_branch,) flow contribution from fixed load

    # Inequality constraints: PS @ Pg + const_flow <=  rate
    #                        -PS @ Pg - const_flow <=  rate
    A_ub = np.vstack([PS, -PS])
    b_ub = np.concatenate([rate - const_flow, rate + const_flow])

    # Equality: sum(Pg) = total_load
    A_eq = np.ones((1, n_gen))
    b_eq = np.array([total_load])

    bounds = list(zip(Pmin, Pmax))

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")

    if not res.success:
        return DCOPFResult(Pg=np.zeros(n_gen), total_cost=np.nan,
                            lmp=np.zeros(case.N_BUS), lambda_sys=np.nan,
                            line_flow=np.zeros(n_branch), mu_line=np.zeros(n_branch),
                            feasible=False)

    Pg = res.x
    lambda_sys = res.eqlin.marginals[0] if res.eqlin is not None else 0.0
    # marginals of A_ub rows: first n_branch = forward-direction (flow <= rate)
    # duals, next n_branch = reverse-direction (flow >= -rate) duals. scipy
    # convention: marginals <= 0 on active inequality constraints.
    marg = res.ineqlin.marginals if res.ineqlin is not None else np.zeros(2 * n_branch)
    mu_upper = marg[:n_branch]
    mu_lower = marg[n_branch:]
    mu_line = mu_upper - mu_lower  # net signed congestion price per line (see derivation below)

    # LMP_i = dCost/dload_i, derived from LP sensitivity rather than assumed.
    # The load vector enters BOTH right-hand sides, which the previous version of
    # this function missed:
    #     b_eq             = total_load           -> d/dload_i = 1
    #     b_ub(fwd rows)   = rate + PTDF@load     -> d/dload_i = +PTDF[l,i]
    #     b_ub(rev rows)   = rate - PTDF@load     -> d/dload_i = -PTDF[l,i]
    # With dObj/db = dual (scipy's own signs), this gives
    #     LMP = lambda + PTDF^T (mu_upper - mu_lower)
    #
    # BUG FIX (2026-07-17, DOE Phase 3 prep, experiments/e2d_*.py + e2e_*.py):
    # this line previously read `lambda_sys - ptdf.T @ (mu_upper + mu_lower)` --
    # wrong sign on the congestion term AND adding the two directions instead of
    # differencing them. Validated by central finite difference of the LP's own
    # optimal cost (LMP is dCost/dload by definition) on IEEE-14/30/57/118 at two
    # load levels: the expression below reproduces finite-difference LMPs to
    # 0.00000 in 6 of 8 cases (0.0045 mean in the 7th, LP degeneracy), whereas the
    # previous expression was off by a mean of 21.7 $/MWh -- larger than lambda
    # itself -- and reported nearly every bus BELOW lambda when finite difference
    # shows nearly every bus above it. Every grid instance generated before this
    # date was built from the incorrect LMPs; see results/doe_phase3/
    # e2e_lmp_formula_variants.csv for the full variant comparison.
    lmp = lambda_sys + ptdf.T @ mu_line

    line_flow = PS @ Pg + const_flow

    return DCOPFResult(Pg=Pg, total_cost=res.fun, lmp=lmp, lambda_sys=lambda_sys,
                        line_flow=line_flow, mu_line=mu_line, feasible=True)
