"""Sprint 14, C2 (DOE Phase 3 alignment): N-1 line-outage contingency
evaluation. Extends the DC-OPF LP with per-bus load-shedding variables (a
standard security-constrained DC-OPF formulation) so that "unserved energy"
under a line outage is a real, computed LP quantity, not a proxy -- directly
answering the challenge's own suggested resilience metric ("expected
unserved energy under N-1 ... scenarios").

Not built on top of an earlier synthetic cut-discovery testbed: those implement
a synthetic cut-discovery testbed rather than real grid physics, so reusing
it for an actual N-1 study would be reusing the wrong abstraction. This module
extends the real DC-OPF LP in :mod:`qgridx.grid.dcopf` directly.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # pragma: no cover
    # Type-checking only, deliberately. The grid layer must not depend on the
    # problem layer at runtime: `qgridx.problems.siting` imports
    # `qgridx.grid.dcopf`, so importing it here at module scope makes
    # `qgridx.grid` a partially initialized module and the package fails to
    # import. `QUBOInstance` is used in one annotation and nowhere else.
    from qgridx.problems.siting import QUBOInstance

SHED_COST = 1000.0  # $/MWh, standard VOLL-style penalty >> any generator's marginal cost
MW_PER_CAPACITY_TIER = 25.0  # documented modeling choice: each domain-wall tier = 25 MW of
# deployed storage/discharge capacity at that bus -- illustrative (no published per-tier MW
# rating exists anywhere in this project's encoding, which treats capacity abstractly via
# cost/benefit terms only), chosen to be plausible battery-ESS scale for a single candidate bus.


def compute_ptdf_outage(case_module, outage_branch_idx: int) -> np.ndarray:
    """Same as case_module.compute_ptdf(), but with one branch's reactance
    removed from the susceptance matrix (an open circuit -- the real N-1
    topology change, not just a zeroed rating on the still-connected branch)."""
    from qgridx.grid.cases import case14 as default_case
    cm = case_module or default_case
    A, b, _, _ = cm.incidence_and_susceptance()
    keep = np.ones(len(b), dtype=bool)
    keep[outage_branch_idx] = False
    A_out, b_out = A[keep], b[keep]

    Bf = b_out[:, None] * A_out
    Bbus = A_out.T @ Bf
    slack = cm.bus_index(cm.SLACK_BUS)
    keep_bus = [i for i in range(cm.N_BUS) if i != slack]
    Bbus_red = Bbus[np.ix_(keep_bus, keep_bus)]
    Xbus_red = np.linalg.inv(Bbus_red)
    Xbus = np.zeros((cm.N_BUS, cm.N_BUS))
    Xbus[np.ix_(keep_bus, keep_bus)] = Xbus_red

    # full-size PTDF (n_branch, N_BUS), with the outaged row zeroed (it carries no flow)
    ptdf_full = np.zeros((len(b), cm.N_BUS))
    ptdf_full[keep] = Bf @ Xbus
    return ptdf_full


@dataclass
class SheddingResult:
    unserved_mw: float
    total_cost: float
    feasible: bool
    gen_cost: float = float("nan")  # dispatch cost only, with the VOLL shedding penalty removed.
    # Kept separate because `total_cost` mixes the two and SHED_COST dominates it by ~2 orders of
    # magnitude whenever anything sheds, which makes total_cost useless as a congestion measure.
    # Defaulted so the pre-existing callers that construct SheddingResult positionally still work.
    shed_by_bus: np.ndarray = None   # (N_BUS,) per-bus unserved MW. Needed to attribute outage
    # exposure to individual candidate buses, which is what makes a microgrid islanding decision's
    # value a computed quantity rather than an assumed one.


def solve_dcopf_with_shedding(load_mw: np.ndarray, ptdf: np.ndarray, case_module,
                               extra_gen_bus_mw: dict = None) -> SheddingResult:
    """Security-constrained DC-OPF: minimize generation cost + SHED_COST *
    sum(shedding), subject to balance (gen + shed = load), generator limits,
    and (surviving, post-outage) line limits. extra_gen_bus_mw: optional
    {bus_num: extra_Pmax_mw} -- models deployed storage discharge capacity at
    sited buses as extra dispatchable, zero-marginal-cost generation during
    the contingency (a standard simplification: storage already charged,
    available to inject during the event)."""
    cm = case_module
    n_gen = cm.GEN.shape[0]
    gen_bus_idx = np.array([cm.bus_index(b) for b in cm.gen_buses()])
    Pmax = cm.GEN[:, 1].copy()
    Pmin = cm.GEN[:, 2].copy()
    cost = cm.GEN[:, 3].copy()
    rate = cm.rate_a()
    n_branch = rate.shape[0]
    n_bus = cm.N_BUS

    extra_gen_bus_mw = extra_gen_bus_mw or {}
    extra_buses = list(extra_gen_bus_mw.keys())
    n_extra = len(extra_buses)
    extra_bus_idx = [cm.bus_index(b) for b in extra_buses]

    # decision vector: [Pg (n_gen), Pextra (n_extra), shed (n_bus)]
    n_vars = n_gen + n_extra + n_bus
    S = np.zeros((n_bus, n_gen))
    for g, bi in enumerate(gen_bus_idx):
        S[bi, g] = 1.0
    S_extra = np.zeros((n_bus, n_extra))
    for e, bi in enumerate(extra_bus_idx):
        S_extra[bi, e] = 1.0
    S_shed = np.eye(n_bus)

    # net injection at bus i = gen_i + extra_i + shed_i - load_i (shedding un-serves load,
    # which reduces net consumption exactly like adding generation would -- same sign as Pg/Pextra)
    PS = ptdf @ np.hstack([S, S_extra, S_shed])  # flow contribution per decision var
    const_flow = -ptdf @ load_mw
    A_ub = np.vstack([PS, -PS])
    b_ub = np.concatenate([rate - const_flow, rate + const_flow])

    # balance: sum(Pg) + sum(Pextra) + sum(shed) = total_load
    A_eq = np.hstack([np.ones((1, n_gen)), np.ones((1, n_extra)), np.ones((1, n_bus))])
    b_eq = np.array([load_mw.sum()])

    c_obj = np.concatenate([cost, np.zeros(n_extra), np.full(n_bus, SHED_COST)])
    bounds = (list(zip(Pmin, Pmax)) + [(0.0, extra_gen_bus_mw[b]) for b in extra_buses]
              + [(0.0, max(l, 0.0)) for l in load_mw])

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return SheddingResult(unserved_mw=np.nan, total_cost=np.nan, feasible=False)
    shed = res.x[n_gen + n_extra:]
    return SheddingResult(unserved_mw=float(shed.sum()), total_cost=float(res.fun), feasible=True,
                          gen_cost=float(cost @ res.x[:n_gen]), shed_by_bus=shed.copy())


def plan_to_bus_mw(inst: "QUBOInstance", x: np.ndarray) -> dict:
    """Convert a decoded siting plan's per-bus domain-wall level into
    {bus_num: MW} using MW_PER_CAPACITY_TIER."""
    levels = inst.levels_from_x(x)
    return {bus: int(level) * MW_PER_CAPACITY_TIER for bus, level in zip(inst.candidate_buses, levels)
            if level > 0}
