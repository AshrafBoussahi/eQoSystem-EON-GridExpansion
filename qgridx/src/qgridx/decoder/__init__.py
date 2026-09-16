"""From correlation signs to an implementable plan.

Sign read-out gives a bit string, not a plan. This layer repairs it into one:
domain-wall monotonicity, then budget feasibility, then local search. Infeasible
output is structurally impossible, whatever the measurement noise does, so the
planner always receives something buildable.

Modules
-------
repair     Sign read-out, monotonicity and budget repair, and the joint
           neighbourhood search over simultaneous multi-bus tier moves.
ilp        Maximum-a-posteriori decode as a small integer program.
portfolio  Runs the repair path and the ILP path and keeps the better plan.
"""
from qgridx.decoder.repair import (
    sign_readout, repair_domain_wall, repair_budget, joint_neighborhood_search,
    full_decode,
)
from qgridx.decoder.portfolio import decode_portfolio
from qgridx.decoder.ilp import map_decode_ilp, Cut

__all__ = [
    "sign_readout", "repair_domain_wall", "repair_budget",
    "joint_neighborhood_search", "full_decode",
    "decode_portfolio", "map_decode_ilp", "Cut",
]
