"""Power-system layer: network models, DC optimal power flow, contingencies.

This is the only part of qGridX that knows what a bus is. Everything the
quantum layer sees has already been reduced to coefficients here.

Modules
-------
cases        IEEE 14/30/57/118-bus test systems, one module each.
dcopf        DC optimal power flow; returns locational marginal prices and
             line-limit duals, which become the objective coefficients.
contingency  N-1 screening: post-outage PTDF rebuild plus a
             security-constrained DC-OPF with per-bus load shedding.
"""
from qgridx.grid.dcopf import solve_dcopf, DCOPFResult
from qgridx.grid.contingency import (
    compute_ptdf_outage,
    solve_dcopf_with_shedding,
    plan_to_bus_mw,
    SheddingResult,
    MW_PER_CAPACITY_TIER,
    SHED_COST,
)

__all__ = [
    "solve_dcopf", "DCOPFResult",
    "compute_ptdf_outage", "solve_dcopf_with_shedding", "plan_to_bus_mw",
    "SheddingResult", "MW_PER_CAPACITY_TIER", "SHED_COST",
]
