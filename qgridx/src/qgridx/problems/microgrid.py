"""Microgrid islanding decisions inside the same correlation encoding.

A storage-siting instance carries ``B * L`` sizing bits. This module appends
``B`` binaries, one per candidate bus, where ``y_b = 1`` means bus ``b`` is
equipped to island: switchgear plus a controller, so that during an outage
which would otherwise strand it, local storage can serve local load behind an
open boundary. The extended problem has ``m = B * L + B`` decisions and they go
into the *same* Pauli correlation encoding as the sizing bits.

The point is what that costs in qubits. At ``B = 4`` candidate buses and
``L = 10`` tiers the extended problem is ``m = 44`` against a six-qubit
capacity of ``3 * C(6, 2) = 45``: the microgrid decisions are absorbed into
headroom the encoding already had, for zero additional qubits.

Coefficients are derived, not invented. Islanding alone is strictly costly, so
the solver can never earn anything by islanding a bus it has not equipped with
storage. The payoff is a negative quadratic coupling between ``y_b`` and that
bus's "is sited at all" bit, and its size is set by the bus's *measured* outage
exposure: the mean unserved power at that bus over a full N-1 sweep of the
unbuilt network, computed by the same security-constrained DC-OPF used to
evaluate the finished plan. A bus that never sheds under any single outage gets
no islanding value; the most exposed bus gets the most.
"""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from qgridx.grid.contingency import (
    MW_PER_CAPACITY_TIER, compute_ptdf_outage, solve_dcopf_with_shedding,
)

#: A capacity tier is a four-hour battery block, so tier L at a bus specifies
#: ``25 * L`` MW of power and ``100 * L`` MWh of energy. The duration is a
#: stated procurement block, not an optimized parameter.
BESS_DURATION_H = 4.0

#: Controller and switchgear cost per islanded bus, in the normalized units of
#: the siting objective (fixed siting cost 0.5, unit capex 1.0 per tier). About
#: a third of a bus's fixed siting cost.
ISLAND_CAPEX = 0.18

#: Islanding value at the most exposed bus, before exposure scaling. Set so the
#: benefit at that bus is worth roughly three controllers, which makes islanding
#: the right bus profitable and islanding an unexposed bus never profitable.
ISLAND_VALUE = 0.55


def bus_outage_exposure(case_module, load_mw):
    """Mean unserved MW at each bus over a full N-1 sweep of the unbuilt network.

    This is the quantity that makes an islanding decision worth something, and
    it is measured rather than assumed. Outages that disconnect the network
    admit no post-outage PTDF and are skipped here; they are counted separately
    by the security screen itself.

    Returns
    -------
    numpy.ndarray
        Length ``N_BUS``, mean unserved power per bus.
    """
    acc = np.zeros(case_module.N_BUS)
    n = 0
    for br in range(case_module.BRANCH.shape[0]):
        try:
            ptdf_out = compute_ptdf_outage(case_module, br)
        except np.linalg.LinAlgError:
            continue
        r = solve_dcopf_with_shedding(load_mw, ptdf_out, case_module)
        if r.feasible and r.shed_by_bus is not None:
            acc += r.shed_by_bus
            n += 1
    return acc / n if n else acc


def extend_with_islanding(inst, exposure_at_candidates,
                          island_capex=ISLAND_CAPEX, island_value=ISLAND_VALUE):
    """Append one free binary per candidate bus to an existing instance.

    The ``B`` new variables sit after the ``B * L`` domain-wall bits and are
    deliberately left out of ``bus_bit_slices``, so every consumer that iterates
    over chains treats them as unconstrained, which is what they are:
    monotonicity repair, budget repair, tier decode, and the mixed-integer
    baseline's monotonicity rows all skip them automatically.

    The islanding variables carry zero weight in the capital budget. That is a
    modelling decision stated rather than buried: the budget cap in this family
    is the storage build budget, and microgrid controls are a separate operating
    line. It also keeps the budget repair's domain-wall-only greedy decrement
    exactly correct, since no unconstrained variable can push a plan over the
    cap.

    Parameters
    ----------
    inst : QUBOInstance
        The sizing-only instance to extend.
    exposure_at_candidates : array_like
        Mean unserved MW at each candidate bus, from :func:`bus_outage_exposure`.

    Returns
    -------
    QUBOInstance
        A new instance with ``m = B * L + B``. ``meta['islanding_idx']`` gives
        the indices of the new variables.
    """
    B, m0 = inst.B, inst.m
    m = m0 + B

    c = np.concatenate([inst.c, np.full(B, island_capex)])
    bw = np.concatenate([inst.budget_weights, np.zeros(B)])

    Q = np.zeros((m, m))
    Q[:m0, :m0] = inst.Q
    e = np.asarray(exposure_at_candidates, dtype=float)
    scale = e.max() if e.max() > 1e-9 else 1.0
    synergy = island_value * e / scale
    for b in range(B):
        first_bit = inst.bus_bit_slices[b][0]      # "is this bus sited at all"
        Q[m0 + b, first_bit] = -0.5 * synergy[b]   # symmetric halves; x^T Q x
        Q[first_bit, m0 + b] = -0.5 * synergy[b]   # collects -synergy_b

    # The closed-form free-bit optimum used by polish_free_bits and by the
    # decoder's joint search is exact only if the islanding variables do not
    # couple to each other. They do not, by construction above. Asserted so a
    # future coupling term cannot silently invalidate both.
    assert not Q[m0:, m0:].any(), "islanding variables must not couple to each other"

    ext = replace(inst, m=m, c=c, Q=Q, budget_weights=bw, meta=deepcopy(inst.meta))
    ext.meta.update(
        islanding_idx=list(range(m0, m)),
        islanding_synergy=synergy.tolist(),
        island_capex=island_capex,
        mw_per_tier=MW_PER_CAPACITY_TIER,
        bess_duration_h=BESS_DURATION_H,
    )
    return ext


def polish_free_bits(x, inst):
    """Set every unconstrained islanding bit to its exact optimum given the rest.

    The islanding variables do not couple to one another, so each one's optimal
    value depends only on the sizing bits and is available in closed form: turn
    ``y_b`` on exactly when its linear cost plus its interaction with the
    current plan is negative. One pass is exact; there is nothing to iterate.

    Needed because the decoder's local search only ever moves domain-wall
    capacity levels. Without this step the extra variables would stay frozen at
    whatever the raw sign read-out produced.
    """
    idx = inst.meta.get("islanding_idx")
    if not idx:
        return x
    x = np.asarray(x, dtype=int).copy()
    for i in idx:
        delta = inst.c[i] + 2.0 * float(inst.Q[i] @ x) - 2.0 * inst.Q[i, i] * x[i]
        x[i] = 1 if delta < 0 else 0
    return x


def plan_summary(inst, x, candidate_buses):
    """Human-readable reading of a decoded extended plan.

    Returns a dict with the sited tiers in engineering units, the islanded
    buses, and how much load could be served locally behind an open boundary.
    """
    isl_idx = inst.meta.get("islanding_idx", [])
    levels = inst.levels_from_x(x)
    islanded = [candidate_buses[b] for b in range(inst.B)
                if isl_idx and x[isl_idx[b]] == 1]
    return {
        "levels": levels.tolist(),
        "power_mw": {candidate_buses[b]: float(levels[b] * MW_PER_CAPACITY_TIER)
                     for b in range(inst.B) if levels[b] > 0},
        "energy_mwh": {candidate_buses[b]: float(levels[b] * MW_PER_CAPACITY_TIER
                                                 * BESS_DURATION_H)
                       for b in range(inst.B) if levels[b] > 0},
        "islanded_buses": islanded,
        "islanded_and_sited": sum(1 for b in range(inst.B)
                                  if isl_idx and x[isl_idx[b]] == 1 and levels[b] > 0),
    }
