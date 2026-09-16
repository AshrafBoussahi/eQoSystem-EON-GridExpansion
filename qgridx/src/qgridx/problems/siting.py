"""Phase 0 instance factory: IEEE-14 storage-siting QUBO instances with
domain-wall capacity encoding, seeded from DC-OPF duals (framework Sec 1.5).

Design decision (documented, see report): rather than a separate siting bit
plus an (L-1)-bit sizing chain, each candidate bus gets a single L-bit
domain-wall chain over L+1 states {0 = not sited, 1..L = capacity tier}.
This reproduces every property Sec 1.2/1.5 asks of domain-wall encoding
(single-bit-flip locality, sort-to-repair, linear budget) with one flat chain
per bus, and it reproduces the exact bit budget of framework Sec 1.3
(m = B * L, e.g. B=5, L=4 -> m=20 for IEEE-14).
"""
from dataclasses import dataclass, field

import numpy as np

from qgridx.grid.cases import case14 as case
from qgridx.grid.dcopf import solve_dcopf

DEFAULT_CANDIDATE_BUSES = [4, 5, 9, 10, 14]  # 1-indexed, non-slack load buses
UNIT_CAPEX = 1.0
FIXED_SITING_COST = 0.5
COUPLING_SCALE = 0.6
# Total operational benefit at full capacity level L, for a candidate bus at
# the reference (fixed, cross-instance) locational-value scale of 1.0 -- see
# _LMP_NORM below. Calibrated so a bus's total capex (FIXED_SITING_COST +
# L*UNIT_CAPEX = 4.5 at L=4) is beatable but not trivially so for every bus in
# every instance -- an earlier calibration (benefit negligible vs. capex) made
# "site nothing" always optimal, and a later one (flat capex/benefit curves,
# per-instance max-normalization) made every profitable instance bang-bang to
# full level on exactly one bus. Both were caught by Phase 0 sanity passes and
# are recorded as decisions in the report; this is the third, retained,
# calibration: convex marginal capex + concave marginal benefit (so a level's
# marginal net cost is monotone in the level, keeping the domain-wall's
# threshold structure meaningful) and a fixed norm (so profitability varies
# across instances/buses instead of always picking exactly one "best" bus).
BENEFIT_MAGNITUDE = 6.0
# convex marginal capex per chain position (position 1 also carries FIXED_SITING_COST)
_CAPEX_FACTORS = np.array([1.0, 1.15, 1.35, 1.6, 1.9, 2.25])
# concave marginal benefit weights (front-loaded), renormalized to sum to L in _benefit_weights()
_BENEFIT_FACTORS = np.array([1.6, 1.3, 1.0, 0.75, 0.55, 0.4])


def _capex_factors(L: int) -> np.ndarray:
    if L <= len(_CAPEX_FACTORS):
        return _CAPEX_FACTORS[:L]
    # extrapolate convexly for L beyond the table
    extra = 1.0 + 0.2 * np.arange(L)
    return extra


def _benefit_weights(L: int) -> np.ndarray:
    if L <= len(_BENEFIT_FACTORS):
        w = _BENEFIT_FACTORS[:L]
    else:
        w = 1.0 / (1.0 + 0.3 * np.arange(L))
    return w * L / w.sum()


def _reference_lmp_scale(case_module, pool_buses) -> float:
    """Fixed (not per-instance) normalization scale for the locational LMP
    component, computed once from the nominal-load case over the full
    candidate-bus pool. Using a fixed scale (rather than per-instance max) is
    what lets 0, 1, or several buses be simultaneously profitable depending on
    the instance, instead of always exactly one."""
    load = case_module.BUS[:, 1].copy()
    ptdf = case_module.compute_ptdf()
    res = solve_dcopf(load, ptdf, case=case_module)
    pool_idx = [case_module.bus_index(b) for b in pool_buses]
    locational = res.lmp[pool_idx] - res.lambda_sys
    return float(np.mean(np.abs(locational)))


_LMP_REFERENCE_POOL_CASE14 = [4, 5, 7, 9, 10, 11, 12, 14]  # unchanged from Phase 0/Sprint 4 -- the
# original hardcoded normalization pool; preserved exactly so every prior sprint's IEEE-14 numbers
# stay reproducible. Sprint 8 Track B's IEEE-30 factory needs its own pool (see build_v30_instance).
_LMP_SCALE_CACHE = {}  # keyed by (case_module.__name__, tuple(pool_buses)) -- per-grid cache


def _lmp_scale(case_module=case, pool_buses=None) -> float:
    pool_buses = tuple(pool_buses if pool_buses is not None else _LMP_REFERENCE_POOL_CASE14)
    key = (case_module.__name__, pool_buses)
    if key not in _LMP_SCALE_CACHE:
        _LMP_SCALE_CACHE[key] = _reference_lmp_scale(case_module, pool_buses)
    return _LMP_SCALE_CACHE[key]


@dataclass
class QUBOInstance:
    m: int
    B: int
    L: int
    candidate_buses: list
    c: np.ndarray                # (m,) linear coefficients
    Q: np.ndarray                # (m, m) symmetric, zero diagonal, quadratic coefficients
    budget_weights: np.ndarray   # (m,) linear cost-of-a-1-bit for the budget constraint
    budget_cap: float
    bus_bit_slices: list         # list of (start, end) exclusive, one per bus, len B
    seed: int
    load_scale: np.ndarray
    budget_fraction: float
    meta: dict = field(default_factory=dict)

    def bits_for_bus(self, b_idx: int, x: np.ndarray) -> np.ndarray:
        s, e = self.bus_bit_slices[b_idx]
        return x[s:e]

    def level_from_bits(self, bits: np.ndarray) -> int:
        """Domain-wall decode: level = number of 1s, valid iff bits are a
        monotone (leading-ones) prefix. Caller enforces validity where needed."""
        return int(np.sum(bits))

    def is_monotone(self, bits: np.ndarray) -> bool:
        # valid domain wall: no 0 followed later by a 1
        seen_zero = False
        for v in bits:
            if v == 0:
                seen_zero = True
            elif seen_zero:
                return False
        return True

    def levels_from_x(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.level_from_bits(self.bits_for_bus(b, x)) for b in range(self.B)])

    def cost(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.c + x @ self.Q @ x)

    def budget_used(self, x: np.ndarray) -> float:
        return float(np.asarray(x, dtype=float) @ self.budget_weights)

    def is_feasible(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=int)
        for b in range(self.B):
            if not self.is_monotone(self.bits_for_bus(b, x)):
                return False
        return self.budget_used(x) <= self.budget_cap + 1e-9


def _bus_capex_price_vector(L: int) -> np.ndarray:
    # position 1 bears the fixed siting cost, remaining positions are convex marginal capex
    prices = UNIT_CAPEX * _capex_factors(L)
    prices[0] += FIXED_SITING_COST
    return prices


def build_instance(seed: int,
                    candidate_buses=None,
                    L: int = 4,
                    load_scale_range=(0.7, 1.3),
                    budget_fraction_range=(0.15, 0.5),
                    coupling_scale_mult: float = 1.0,
                    case_module=None,
                    lmp_reference_pool=None,
                    extra_loads: dict = None,
                    benefit_magnitude: float = None,
                    load_scale_vector: np.ndarray = None) -> QUBOInstance:
    """coupling_scale_mult: Sprint 4 (B1) difficulty lever -- multiplies the
    congestion-coupling scale (COUPLING_SCALE) so cross-bus Q interactions can
    be made to dominate more instances than v1's default. 1.0 reproduces v1
    exactly. case_module/lmp_reference_pool: Sprint 8 Track B -- pass
    ieee_case30 (+ its own reference pool) for the cross-grid factory
    extension; both default to IEEE-14's originals, reproducing every prior
    sprint's instances exactly. extra_loads: Sprint 14 C1 (DOE Phase 3
    alignment) -- optional {bus_num: extra_mw} dict, added on top of the
    randomized residential/industrial load *after* scaling, representing a
    synthetic large flexible load (the challenge's own wording: "one or more
    synthetic AI data center loads of the order of 50 to 500 MW added to
    selected buses"). Deliberately not scaled by load_scale_range, since a
    new large-load buildout is a distinct, planner-controlled addition, not
    part of the existing stochastic demand model. None (default) reproduces
    every prior sprint's instances exactly.

    benefit_magnitude: DOE Phase 3 prep (E2) -- overrides the module-level
    BENEFIT_MAGNITUDE, which sets how much operational benefit a unit of
    locational price differential buys. None (default) uses BENEFIT_MAGNITUDE
    = 6.0, reproducing every prior sprint's instances exactly. Why it is now a
    parameter: at 6.0 the profitability threshold for the cheapest chain
    position is lmp_locational > 1.187, which only 1-2 buses per grid clear
    even after the LMP bug fix, so optima site at most one bus and the budget
    constraint never binds -- technically non-trivial but combinatorially thin.
    Raising it lowers every position's threshold proportionally, so several
    buses compete and the budget becomes the active trade-off. Calibrated in
    experiments/e2g_benefit_calibration.py against both failure modes the
    module docstring warns about (all-trivial and all-saturated)."""
    cm = case_module or case
    benefit_mag = BENEFIT_MAGNITUDE if benefit_magnitude is None else float(benefit_magnitude)
    rng = np.random.default_rng(seed)
    candidate_buses = list(candidate_buses or DEFAULT_CANDIDATE_BUSES)
    B = len(candidate_buses)
    m = B * L

    load_scale = rng.uniform(load_scale_range[0], load_scale_range[1], size=cm.N_BUS)
    budget_fraction = float(rng.uniform(*budget_fraction_range))
    if load_scale_vector is not None:
        # E3: deterministic per-bus load scaling supplied by the caller, used to
        # hold an instance's stochastic load SHAPE fixed while sweeping it across
        # scenario LEVELS. The budget_fraction draw above is deliberately left
        # untouched and still consumes the same rng call, so an instance built
        # with a load_scale_vector keeps the same budget cap as the same-seeded
        # instance built without one -- the scenario changes the network
        # conditions, not the capital budget.
        load_scale = np.asarray(load_scale_vector, dtype=float)
        assert load_scale.shape == (cm.N_BUS,), \
            f"load_scale_vector must have shape ({cm.N_BUS},), got {load_scale.shape}"

    load = cm.BUS[:, 1].copy() * load_scale
    if extra_loads:
        for bus_num, extra_mw in extra_loads.items():
            load[cm.bus_index(bus_num)] += extra_mw
    ptdf = cm.compute_ptdf()
    res = solve_dcopf(load, ptdf, case=cm)
    if not res.feasible:
        # extremely rare with these ranges; fall back to nominal load
        load = cm.BUS[:, 1].copy()
        res = solve_dcopf(load, ptdf, case=cm)

    cand_idx = [cm.bus_index(b) for b in candidate_buses]
    lmp_locational = res.lmp[cand_idx] - res.lambda_sys  # congestion-only component

    # fixed (cross-instance) normalization -- see _reference_lmp_scale()
    lmp_locational = lmp_locational / _lmp_scale(cm, lmp_reference_pool)

    price_vec = _bus_capex_price_vector(L)        # (L,) convex marginal capex
    benefit_w = _benefit_weights(L)                # (L,) concave marginal benefit weights

    c = np.zeros(m)
    budget_weights = np.zeros(m)
    bus_bit_slices = []
    for bi in range(B):
        s, e = bi * L, (bi + 1) * L
        bus_bit_slices.append((s, e))
        budget_weights[s:e] = price_vec
        # marginal net cost per chain position = convex capex - concave benefit,
        # so the marginal net cost is monotone increasing in level (threshold structure)
        c[s:e] = price_vec - (benefit_mag * lmp_locational[bi] / L) * benefit_w

    # quadratic congestion coupling between the *first* bit of each bus's chain
    # (the domain-wall "is this bus sited at all" indicator)
    Q = np.zeros((m, m))
    weighted_mu = np.abs(res.mu_line)
    if weighted_mu.sum() > 0:
        weighted_mu = weighted_mu / weighted_mu.max()
    for i in range(B):
        for j in range(i + 1, B):
            overlap = float(np.sum(weighted_mu * ptdf[:, cand_idx[i]] * ptdf[:, cand_idx[j]]))
            qij = COUPLING_SCALE * coupling_scale_mult * overlap
            fi = bus_bit_slices[i][0]
            fj = bus_bit_slices[j][0]
            Q[fi, fj] = qij
            Q[fj, fi] = qij
    # normalize Q the same way as c, bounded dynamic range (ceiling scales with
    # coupling_scale_mult -- otherwise this clamp would silently cancel out the
    # v2 difficulty lever by rescaling every instance back to the same v1 ceiling)
    qmax = np.max(np.abs(Q)) if np.max(np.abs(Q)) > 0 else 1.0
    ceiling = 5.0 * coupling_scale_mult
    if qmax > ceiling:
        Q = Q * (ceiling / qmax)

    budget_cap = budget_fraction * np.sum(budget_weights)

    return QUBOInstance(
        m=m, B=B, L=L, candidate_buses=candidate_buses, c=c, Q=Q,
        budget_weights=budget_weights, budget_cap=budget_cap,
        bus_bit_slices=bus_bit_slices, seed=seed, load_scale=load_scale,
        budget_fraction=budget_fraction,
        meta=dict(lambda_sys=res.lambda_sys, lmp=res.lmp.tolist(),
                   benefit_magnitude=benefit_mag,
                   dcopf_feasible=res.feasible, extra_loads=extra_loads),
    )
