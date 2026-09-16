"""G2 (DOE Phase 3 submission): battery-storage sizing plus microgrid islanding
decisions in one Pauli correlation encoding.

Closes Gap B. Two changes to the decision model, neither of which changes the
quantum layer's interface:

  1. BESS reframing. A domain-wall tier is no longer an abstract "capacity
     level" but a concrete battery bucket: `MW_PER_TIER` of power and
     `MW_PER_TIER * BESS_DURATION_H` of energy. Level 3 at a bus therefore
     means a 75 MW / 300 MWh installation. This is a relabeling of an existing
     quantity, stated explicitly so the plan is readable as an engineering
     specification, and it is the same mapping :mod:`qgridx.grid.contingency` already
     uses to convert a plan into post-outage injection.

  2. Microgrid islanding. One extra binary per candidate bus, y_b = 1 meaning
     the bus is equipped to island -- switchgear plus a controller, so during a
     contingency that would otherwise strand it, local storage can serve local
     load behind an open boundary. The extra variables go into the SAME
     correlation encoding as the sizing bits, appended after the B domain-wall
     chains, so m = B*L + B.

     The point of the exercise is what this costs in qubits. At B=4, L=10 the
     extended problem is m = 44, and the k=2 capacity of a 6-qubit register is
     3*C(6,2) = 45: the islanding decisions are absorbed into spare capacity the
     encoding already had, for zero additional qubits. At B=5, L=9 the extended
     problem is m = 50, which needs 7 qubits (capacity 63). Both are reported.

Coefficients are derived, not invented. Islanding on its own is strictly
costly (`c_y > 0`), so the solver can never earn anything by islanding a bus it
has not equipped with storage. The payoff is a negative quadratic coupling
between y_b and the bus's "is sited at all" domain-wall bit, and its size is
set by that bus's measured outage exposure: the mean unserved MW at bus b over a
full N-1 sweep of the intact plan, computed by the same
security-constrained DC-OPF used in G1. A bus that never sheds under any
single outage gets no islanding value; the bus that sheds most gets the most.

The islanding variables carry zero weight in the capital budget. That is a
modeling decision, stated rather than buried: the budget cap in this family is
the storage build budget, and microgrid controls are a separate operating line.
It also keeps `repair_budget`'s domain-wall-only greedy decrement exactly
correct, since no unconstrained variable can push the plan over the cap.
"""
import multiprocessing as mp
import os
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from qgridx.grid.cases import case14 as ieee_case14, case30 as ieee_case30, case57 as ieee_case57, case118 as ieee_case118
from qgridx.grid.contingency import MW_PER_CAPACITY_TIER, compute_ptdf_outage, plan_to_bus_mw, solve_dcopf_with_shedding
from qgridx.generator.executor import build_unitary_table, execute_batch
from qgridx.generator.train import RegimeAConfig, train_regime_a
from qgridx.generator.vocab import GQEVocab
from qgridx.problems.siting import build_instance
from qgridx.baselines.mip import solve_mip
from qgridx.decoder.repair import joint_neighborhood_search, repair_budget, repair_domain_wall, sign_readout
from qgridx.encoding.loss import correlators
from qgridx.encoding.families import max_capacity, random_assignment
from qgridx.decoder.portfolio import decode_portfolio

from qgridx.utils.paths import results_dir

OUT_DIR = results_dir() / "doe_phase3"

BESS_DURATION_H = 4.0          # a 4-hour battery, the standard utility-scale procurement block
CEIL_MULT = 3
BENEFIT_MAG, BUDGET_RANGE, COUPLING_MULT = 20.0, (0.15, 0.35), 40.0
EVAL_BUDGET = 10_000
TRAIN_SEED = 101

# Islanding economics, in the same normalized units as the siting objective
# (FIXED_SITING_COST = 0.5, UNIT_CAPEX = 1.0 per tier). The controller costs
# about a third of a bus's fixed siting cost; the exposure-scaled benefit at the
# most exposed bus is worth about three controllers, so islanding the right bus
# pays and islanding an unexposed bus never does.
ISLAND_CAPEX = 0.18
ISLAND_VALUE = 0.55

SCREENED = {
    "IEEE-14": [9, 10, 4, 14, 11, 3, 13, 12, 6],
    "IEEE-30": [8, 28, 29, 27, 30, 26, 25, 6, 9],
    "IEEE-57": [16, 12, 10, 51, 9, 50, 55, 11, 43],
    "IEEE-118": [106, 75, 107, 118, 42, 74, 76, 105, 41],
}
MODULES = {"IEEE-14": ieee_case14, "IEEE-30": ieee_case30,
           "IEEE-57": ieee_case57, "IEEE-118": ieee_case118}


def bus_outage_exposure(cm, load_mw):
    """Mean unserved MW at each candidate bus over a full N-1 sweep with no
    storage in place. This is the quantity that makes an islanding decision
    worth something, and it is measured rather than assumed."""
    ptdf_intact = cm.compute_ptdf()
    acc = np.zeros(cm.N_BUS)
    n = 0
    for br in range(cm.BRANCH.shape[0]):
        try:
            ptdf_out = compute_ptdf_outage(cm, br)
        except np.linalg.LinAlgError:
            continue                      # islanding outage: no post-outage PTDF exists
        r = solve_dcopf_with_shedding(load_mw, ptdf_out, cm)
        if r.feasible and r.shed_by_bus is not None:
            acc += r.shed_by_bus
            n += 1
    if n == 0:
        return np.zeros(cm.N_BUS), ptdf_intact
    return acc / n, ptdf_intact


def extend_with_islanding(inst, exposure_at_candidates):
    """Append one free binary per candidate bus to an existing instance.

    The B new variables sit after the B*L domain-wall bits and are deliberately
    left out of `bus_bit_slices`, so every consumer that iterates over chains --
    monotonicity repair, budget repair, level decode, and the MIP's
    monotonicity rows -- treats them as unconstrained, which is what they are.
    """
    B, L, m0 = inst.B, inst.L, inst.m
    m = m0 + B

    c = np.concatenate([inst.c, np.full(B, ISLAND_CAPEX)])
    bw = np.concatenate([inst.budget_weights, np.zeros(B)])   # outside the capital budget

    Q = np.zeros((m, m))
    Q[:m0, :m0] = inst.Q
    e = np.asarray(exposure_at_candidates, dtype=float)
    scale = e.max() if e.max() > 1e-9 else 1.0
    synergy = ISLAND_VALUE * e / scale                        # (B,), zero where never exposed
    for b in range(B):
        first_bit = inst.bus_bit_slices[b][0]                 # "is this bus sited at all"
        Q[m0 + b, first_bit] = -0.5 * synergy[b]              # symmetric halves: x^T Q x picks up
        Q[first_bit, m0 + b] = -0.5 * synergy[b]              # 2*Q_ij, so this totals -synergy_b

    # The closed-form free-bit optimum used by the decoder's joint search and by
    # polish_free_bits below is exact only if the islanding variables do not
    # couple to each other. They do not, by construction above -- asserted so a
    # future coupling term cannot silently invalidate both.
    assert not Q[m0:, m0:].any(), "islanding variables must not couple to each other"

    ext = replace(inst, m=m, c=c, Q=Q, budget_weights=bw, meta=deepcopy(inst.meta))
    ext.meta.update(islanding_idx=list(range(m0, m)), islanding_synergy=synergy.tolist(),
                    island_capex=ISLAND_CAPEX, mw_per_tier=MW_PER_CAPACITY_TIER,
                    bess_duration_h=BESS_DURATION_H)
    return ext


def polish_free_bits(x, inst):
    """Set every unconstrained islanding bit to its exact optimum given the rest.

    The islanding variables do not couple to each other, so each one's optimal
    value depends only on the sizing bits and is available in closed form: flip
    y_b on exactly when its linear cost plus its interaction with the current
    plan is negative. One pass is exact -- there is nothing to iterate.

    Needed because the decoder's local search only ever moves domain-wall
    capacity levels; without this step the extra variables would be frozen at
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


def decode_extended(pi, inst):
    """Full decode for the extended instance: the standard portfolio for the
    constrained sizing bits, then the exact free-bit polish."""
    x_naive = repair_budget(repair_domain_wall(sign_readout(pi.copy()), inst), inst)
    x_naive = polish_free_bits(joint_neighborhood_search(x_naive, inst), inst)
    pr = decode_portfolio(pi, inst)
    x_port = polish_free_bits(pr.x, inst)
    c_n = inst.cost(x_naive) if inst.is_feasible(x_naive) else np.inf
    c_p = inst.cost(x_port) if inst.is_feasible(x_port) else np.inf
    return (x_naive, c_n) if c_n <= c_p else (x_port, c_p)


def _run(task):
    row, threads = task
    torch.set_num_threads(max(1, int(threads)))
    grid = row["grid"]
    cm = MODULES[grid]
    B, L = int(row["B"]), int(row["L"])
    buses = SCREENED[grid][:B]

    base = build_instance(seed=int(row["gen_seed"]), candidate_buses=buses, L=L,
                          budget_fraction_range=BUDGET_RANGE, coupling_scale_mult=COUPLING_MULT,
                          case_module=cm, lmp_reference_pool=SCREENED[grid],
                          benefit_magnitude=BENEFIT_MAG)

    load_mw = cm.BUS[:, 1].copy() * base.load_scale
    exposure_all, _ = bus_outage_exposure(cm, load_mw)
    exposure = exposure_all[[cm.bus_index(b) for b in buses]]

    inst = extend_with_islanding(base, exposure)

    # smallest register whose k=2 capacity holds the extended problem
    n = 2
    while max_capacity(n, 2) < inst.m:
        n += 1
    n_base = 2
    while max_capacity(n_base, 2) < base.m:
        n_base += 1

    assign = random_assignment(n, 2, inst.m, seed=int(row["gen_seed"]))
    vocab = GQEVocab(n=n, max_len=(3 * n + 2) * CEIL_MULT)
    ut = build_unitary_table(vocab)
    cfg = RegimeAConfig(max_evals=EVAL_BUDGET, checkpoints=(EVAL_BUDGET,), seed=TRAIN_SEED)
    res = train_regime_a(inst, vocab, ut, n, assign, [], cfg)
    with torch.no_grad():
        state = execute_batch(torch.tensor(res.best_tokens[None], dtype=torch.long), ut, n)
        pi = correlators(state, assign, n).numpy()[0]
    x, cost = decode_extended(pi, inst)
    cost = min(float(cost), 0.0)

    x_mip, c_mip, mip_ms, status = solve_mip(inst, time_limit_s=60.0)
    c_mip = min(float(c_mip), 0.0) if c_mip is not None else np.nan

    isl_idx = inst.meta["islanding_idx"]
    levels = inst.levels_from_x(x)
    islanded = [buses[b] for b in range(B) if x[isl_idx[b]] == 1]
    plan_mw = plan_to_bus_mw(inst, x)

    # resilience contribution of the islanded subset: load at islanded buses that
    # also carry storage can be served locally behind an open boundary
    served_locally = 0.0
    for b in range(B):
        if x[isl_idx[b]] == 1 and levels[b] > 0:
            bus_load = float(load_mw[cm.bus_index(buses[b])])
            served_locally += min(bus_load, levels[b] * MW_PER_CAPACITY_TIER)

    return dict(
        instance_id=row["instance_id"], grid=grid, B=B, L=L,
        m_base=int(base.m), m_extended=int(inst.m),
        qubits_base=n_base, qubits_extended=n,
        capacity=int(max_capacity(n, 2)), extra_qubits=n - n_base,
        gqe_cost=cost, mip_cost=c_mip,
        exact_match=bool(abs(cost - c_mip) < 1e-6) if np.isfinite(c_mip) else False,
        mip_ms=float(mip_ms) * 1e3 if mip_ms < 1 else float(mip_ms),
        mip_status=status,
        n_sites=int(np.sum(levels > 0)), total_mw=float(levels.sum() * MW_PER_CAPACITY_TIER),
        total_mwh=float(levels.sum() * MW_PER_CAPACITY_TIER * BESS_DURATION_H),
        n_islanded=len(islanded), islanded_buses=str(islanded),
        islanded_and_sited=int(sum(1 for b in range(B)
                                   if x[isl_idx[b]] == 1 and levels[b] > 0)),
        local_served_mw=served_locally,
        max_exposure_mw=float(exposure.max()), plan=str(plan_mw),
        levels=str(levels.tolist()),
    )


def main(smoke=False, per_cfg=3, workers=None, shard=None, tag=""):
    t0 = time.time()
    reg = pd.read_csv(OUT_DIR / "registry_v2" / "instances_v2.csv")
    tasks = []
    for _, grp in reg.groupby("config"):
        for _, r in grp.sort_values("instance_id").head(1 if smoke else per_cfg).iterrows():
            tasks.append((r.to_dict(), 1))
    if smoke:
        tasks = tasks[:2]
    if shard:
        i, nsh = shard
        tasks = [t for j, t in enumerate(tasks) if j % nsh == i]

    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 4
    W = min(workers or avail, len(tasks))
    threads = max(1, avail // max(1, W))
    tasks = [(t[0], threads) for t in tasks]
    print(f"{len(tasks)} instances, {W} workers x {threads} threads", flush=True)

    rows = []
    part = OUT_DIR / f"g2_microgrid{tag}.partial.csv"
    with mp.Pool(W) as pool:
        for j, r in enumerate(pool.imap_unordered(_run, tasks), 1):
            rows.append(r)
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  [{j}/{len(tasks)}] {r['instance_id']} m {r['m_base']}->{r['m_extended']} "
                  f"qubits {r['qubits_base']}->{r['qubits_extended']} "
                  f"gqe={r['gqe_cost']:.3f} mip={r['mip_cost']:.3f} "
                  f"{'EXACT' if r['exact_match'] else '     '} "
                  f"islanded={r['n_islanded']} [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows).sort_values(["grid", "instance_id"])
    out = OUT_DIR / (f"g2_microgrid{tag}_SMOKE.csv" if smoke else f"g2_microgrid{tag}.csv")
    df.to_csv(out, index=False)
    if part.exists():
        part.unlink()
    print(f"\nwrote {out} ({len(df)} rows)")

    print("\n=== qubit cost of adding B islanding decisions ===")
    print(df.groupby(["B", "L"]).agg(
        m_base=("m_base", "first"), m_ext=("m_extended", "first"),
        qubits_base=("qubits_base", "first"), qubits_ext=("qubits_extended", "first"),
        capacity=("capacity", "first"), extra=("extra_qubits", "first"),
        n=("instance_id", "size")).to_string())
    print("\n=== solve quality on the extended problem ===")
    print(df.groupby("grid").agg(
        n=("instance_id", "size"), exact_pct=("exact_match", lambda s: round(100 * s.mean(), 1)),
        mean_gap=("gqe_cost", "mean"), islanded=("n_islanded", "mean"),
        isl_and_sited=("islanded_and_sited", "mean"),
        local_mw=("local_served_mw", "mean")).round(3).to_string())
    print(f"\noverall exact-match: {100*df.exact_match.mean():.1f}% of {len(df)}")
    print(f"BESS mapping: {MW_PER_CAPACITY_TIER} MW / "
          f"{MW_PER_CAPACITY_TIER*BESS_DURATION_H} MWh per tier "
          f"({BESS_DURATION_H} h duration)")
    print(f"Total wall-clock: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    mp.freeze_support()
    w, pc, sh, tg = None, 3, None, ""
    for a in sys.argv:
        if a.startswith("--workers="):
            w = int(a.split("=", 1)[1])
        if a.startswith("--per-config="):
            pc = int(a.split("=", 1)[1])
        if a.startswith("--shard="):
            i, n = a.split("=", 1)[1].split("/")
            sh = (int(i), int(n))
        if a.startswith("--tag="):
            tg = a.split("=", 1)[1]
    main(smoke="--smoke" in sys.argv, per_cfg=pc, workers=w, shard=sh, tag=tg)
