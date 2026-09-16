"""Sprint 3, A1 (D2): the portfolio decoder. Runs both the naive
(sort-repair + joint-LS) and MAP+ILP (+ joint-LS) paths, projects the naive
plan through the cut-respecting ILP whenever cuts are active (naive repair
has no notion of arbitrary linear cuts), evaluates both candidates on the
surrogate cost, and returns the cheaper *feasible* one. This makes MAP+ILP's
real job -- feasibility and cut machinery -- explicit, and structurally
prevents Sprint-2 debt #3 (a caller forgetting to pass the same cut list to
local search): both paths always route through this one function.
"""
from dataclasses import dataclass

import numpy as np

from qgridx.problems.siting import QUBOInstance
from qgridx.decoder.repair import joint_neighborhood_search, repair_budget, repair_domain_wall, sign_readout
from qgridx.decoder.ilp import map_decode_ilp


def _satisfies_cuts(x: np.ndarray, cuts) -> bool:
    return all(cut.a @ x <= cut.b + 1e-9 for cut in (cuts or []))


def project_via_ilp(x: np.ndarray, inst: QUBOInstance, cuts=None) -> np.ndarray:
    """Project an arbitrary plan onto the feasible (budget+domain-wall+cuts)
    set by maximizing agreement with x's own bit values -- the projection ILP
    used exactly as the framework's MAP decoder, just with "confidence"
    weights standing in for x itself rather than shot-estimated correlators."""
    w = 2.0 * x - 1.0
    x_proj, _, status = map_decode_ilp(w, inst, cuts=cuts)
    if x_proj is None:
        raise RuntimeError(f"projection ILP infeasible, status={status} "
                            "(budget_cap should always admit the all-zero plan)")
    return x_proj


@dataclass
class PortfolioResult:
    x: np.ndarray
    cost: float
    winner: str          # "naive" or "map_ilp"
    x_naive: np.ndarray
    x_map: np.ndarray


def decode_portfolio(mu: np.ndarray, inst: QUBOInstance, sigma: np.ndarray = None,
                      cuts=None, radius: int = 1, large_b_boost: bool = False) -> PortfolioResult:
    cuts = cuts or []

    # naive path
    x_naive = repair_domain_wall(sign_readout(mu.copy()), inst)
    x_naive = repair_budget(x_naive, inst)
    if cuts and not _satisfies_cuts(x_naive, cuts):
        x_naive = project_via_ilp(x_naive, inst, cuts=cuts)
    x_naive = joint_neighborhood_search(x_naive, inst, radius=radius, cuts=cuts, large_b_boost=large_b_boost)

    # MAP+ILP path
    if sigma is None:
        sigma = np.sqrt(np.clip(1 - mu ** 2, 1e-6, None))  # unit-shot-count fallback
    w = mu / np.clip(sigma ** 2, 1e-6, None)
    x_map, _, status = map_decode_ilp(w, inst, cuts=cuts)
    if x_map is None:
        x_map = x_naive.copy()  # degenerate fallback, should not occur (budget_cap>=0)
    x_map = joint_neighborhood_search(x_map, inst, radius=radius, cuts=cuts, large_b_boost=large_b_boost)

    cost_naive = inst.cost(x_naive) if inst.is_feasible(x_naive) and _satisfies_cuts(x_naive, cuts) else np.inf
    cost_map = inst.cost(x_map) if inst.is_feasible(x_map) and _satisfies_cuts(x_map, cuts) else np.inf

    if cost_naive <= cost_map:
        return PortfolioResult(x=x_naive, cost=cost_naive, winner="naive", x_naive=x_naive, x_map=x_map)
    return PortfolioResult(x=x_map, cost=cost_map, winner="map_ilp", x_naive=x_naive, x_map=x_map)
