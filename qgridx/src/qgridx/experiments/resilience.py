"""G1 (DOE Phase 3 submission): N-1 security and expected unserved energy for
GQE+PCE siting plans under AI data-center load growth.

Closes two gaps in the paper at once, on the CORRECTED instance family
(`registry_v2`: fixed LMP formula, screened candidate buses, benefit_magnitude
20, budget fraction (0.15, 0.35), coupling 40) rather than the pre-correction
registry the earlier X14 run used.

  Gap A -- resilience. The siting objective sees only a single intact-network
  DC-OPF, so it carries no security information at all. Here every decoded plan
  is pushed through a full N-1 line-outage sweep: for each branch in turn the
  branch is opened (its reactance removed from the susceptance matrix, not just
  its rating zeroed), the post-outage PTDF is rebuilt, and a
  security-constrained DC-OPF with per-bus load shedding is solved. That yields
  the three metrics the challenge names directly:
      * secure fraction    -- share of outages served with zero shedding
      * EUE                -- expected unserved energy over the outage set
      * congestion cost    -- post-outage dispatch cost above the intact case
  Each is computed twice, with and without the sited storage acting as
  zero-marginal-cost injection, so the resilience contribution is a difference
  measured on identical outages rather than an assertion.

  Gap C -- AI load growth. A synthetic data-center load is added at one
  candidate bus over the challenge's own stated range (50-500 MW). It enters
  through `build_instance(extra_loads=...)`, so it reshapes the DC-OPF prices
  and therefore the QUBO coefficients: the siting problem the quantum layer
  sees is genuinely a different problem at each load level, not the same
  problem re-scored.

Grid choice is a physics constraint, not a preference. Serving a 500 MW campus
needs 500 MW of generation headroom, and only two of the four test systems have
it: IEEE-57 (725 MW spare) and IEEE-118 (5,724 MW spare). IEEE-30 has 152 MW
and IEEE-14 has 513 MW before the stochastic load draw eats into it, so neither
can host the upper half of the sweep and both are excluded rather than run and
silently clipped by the factory's infeasibility fallback. `base_feasible` is
recorded per row so no clipped row can pass unnoticed.
"""
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from qgridx.grid.cases import case57 as ieee_case57, case118 as ieee_case118
from qgridx.grid.contingency import MW_PER_CAPACITY_TIER, compute_ptdf_outage, plan_to_bus_mw, solve_dcopf_with_shedding
from qgridx.grid.dcopf import solve_dcopf
from qgridx.generator.executor import build_unitary_table, execute_batch
from qgridx.generator.train import RegimeAConfig, train_regime_a
from qgridx.generator.vocab import GQEVocab
from qgridx.problems.siting import build_instance
from qgridx.decoder.repair import joint_neighborhood_search, repair_budget, repair_domain_wall, sign_readout
from qgridx.encoding.loss import correlators
from qgridx.encoding.families import random_assignment
from qgridx.decoder.portfolio import decode_portfolio

from qgridx.utils.paths import results_dir

OUT_DIR = results_dir() / "doe_phase3"

N_QUBITS, K = 6, 2
CEIL_MULT = 3                       # the gate budget selected in the paper's design sweep
BENEFIT_MAG, BUDGET_RANGE, COUPLING_MULT = 20.0, (0.15, 0.35), 40.0
EVAL_BUDGET = 10_000
TRAIN_SEED = 101

# the challenge's own stated range for a synthetic data-center load
AI_LOAD_MW = (0.0, 50.0, 100.0, 200.0, 350.0, 500.0)

# registry_v2's screened candidate buses (mean congestion-only locational value
# over 40 stochastic draws), restricted to the two grids with the generation
# headroom to serve 500 MW
SCREENED = {
    "IEEE-57": [16, 12, 10, 51, 9, 50, 55, 11, 43],
    "IEEE-118": [106, 75, 107, 118, 42, 74, 76, 105, 41],
}
MODULES = {"IEEE-57": ieee_case57, "IEEE-118": ieee_case118}
# the data-center bus is itself a candidate storage site: a campus co-located
# with a site the planner is already considering is the realistic case, and it
# is also the case where siting can actually answer the new load
AI_BUS = {"IEEE-57": 16, "IEEE-118": 106}


def _load_vector(cm, load_scale, extra_loads):
    load = cm.BUS[:, 1].copy() * load_scale
    if extra_loads:
        for bus_num, mw in extra_loads.items():
            load[cm.bus_index(bus_num)] += mw
    return load


def _islanded_load_mw(cm, outage_branch_idx, load_mw):
    """Load stranded from the slack bus when opening this branch splits the
    network. Such an outage has no post-outage PTDF at all, so the LP cannot be
    posed and the stranded load is unserved by definition."""
    adj = [[] for _ in range(cm.N_BUS)]
    for l, (fb, tb, _x, _r) in enumerate(cm.BRANCH):
        if l == outage_branch_idx:
            continue
        fi, ti = cm.bus_index(fb), cm.bus_index(tb)
        adj[fi].append(ti)
        adj[ti].append(fi)
    slack = cm.bus_index(cm.SLACK_BUS)
    seen, stack = {slack}, [slack]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return float(load_mw[[i for i in range(cm.N_BUS) if i not in seen]].sum())


def n1_sweep(cm, load_mw, plan_mw):
    """Open every branch in turn and solve the security-constrained DC-OPF.

    Returns the three challenge metrics plus the raw per-outage unserved series.
    `secure_frac` counts an outage as secure when the LP serves all load with
    zero shedding; islanding outages are never secure.
    """
    base = solve_dcopf_with_shedding(load_mw, cm.compute_ptdf(), cm, extra_gen_bus_mw=plan_mw)
    base_gen_cost = base.gen_cost if base.feasible else np.nan

    unserved, gen_costs, secure = [], [], []
    n_island = 0
    for br in range(cm.BRANCH.shape[0]):
        try:
            ptdf_out = compute_ptdf_outage(cm, br)
        except np.linalg.LinAlgError:
            unserved.append(_islanded_load_mw(cm, br, load_mw))
            secure.append(False)
            n_island += 1
            continue
        r = solve_dcopf_with_shedding(load_mw, ptdf_out, cm, extra_gen_bus_mw=plan_mw)
        if not r.feasible:
            unserved.append(float(load_mw.sum()))
            secure.append(False)
            continue
        unserved.append(r.unserved_mw)
        gen_costs.append(r.gen_cost)
        secure.append(r.unserved_mw < 1e-6)

    return dict(
        eue_mw=float(np.mean(unserved)),
        max_unserved_mw=float(np.max(unserved)),
        secure_frac=float(np.mean(secure)),
        n_secure=int(np.sum(secure)),
        n_outages=int(cm.BRANCH.shape[0]),
        n_islanding=n_island,
        # congestion cost of the contingency: mean post-outage dispatch cost above
        # the intact-network dispatch, in $/h
        congestion_cost=float(np.mean(gen_costs) - base_gen_cost) if gen_costs else np.nan,
        base_gen_cost=float(base_gen_cost),
    )


def solve_with_gqe(inst, gen_seed):
    """Train the generator on this instance and return the decoded plan."""
    assign = random_assignment(N_QUBITS, K, inst.m, seed=gen_seed)
    vocab = GQEVocab(n=N_QUBITS, max_len=(3 * N_QUBITS + 2) * CEIL_MULT)
    ut = build_unitary_table(vocab)
    cfg = RegimeAConfig(max_evals=EVAL_BUDGET, checkpoints=(EVAL_BUDGET,), seed=TRAIN_SEED)
    res = train_regime_a(inst, vocab, ut, N_QUBITS, assign, [], cfg)

    with torch.no_grad():
        state = execute_batch(torch.tensor(res.best_tokens[None], dtype=torch.long), ut, N_QUBITS)
        pi = correlators(state, assign, N_QUBITS).numpy()[0]

    x_naive = repair_budget(repair_domain_wall(sign_readout(pi.copy()), inst), inst)
    x_naive = joint_neighborhood_search(x_naive, inst)
    pr = decode_portfolio(pi, inst)
    c_naive = inst.cost(x_naive) if inst.is_feasible(x_naive) else np.inf
    x, cost = (x_naive, c_naive) if c_naive <= pr.cost else (pr.x, pr.cost)
    return x, min(float(cost), 0.0)      # do-nothing guard, as everywhere else


def _run(task):
    row, ai_mw, threads = task
    torch.set_num_threads(max(1, int(threads)))
    grid = row["grid"]
    cm = MODULES[grid]
    buses = SCREENED[grid][:int(row["B"])]
    extra = {AI_BUS[grid]: ai_mw} if ai_mw > 0 else None

    inst = build_instance(seed=int(row["gen_seed"]), candidate_buses=buses, L=int(row["L"]),
                          budget_fraction_range=BUDGET_RANGE, coupling_scale_mult=COUPLING_MULT,
                          case_module=cm, lmp_reference_pool=SCREENED[grid],
                          benefit_magnitude=BENEFIT_MAG, extra_loads=extra)

    load_mw = _load_vector(cm, inst.load_scale, extra)
    base_ok = solve_dcopf(load_mw, cm.compute_ptdf(), case=cm).feasible

    x, cost = solve_with_gqe(inst, int(row["gen_seed"]))
    plan_mw = plan_to_bus_mw(inst, x)

    no = n1_sweep(cm, load_mw, None)
    wi = n1_sweep(cm, load_mw, plan_mw or None)

    return dict(
        instance_id=row["instance_id"], grid=grid, ai_load_mw=ai_mw, ai_bus=AI_BUS[grid],
        base_feasible=bool(base_ok), m=int(inst.m),
        total_load_mw=float(load_mw.sum()), siting_cost=cost,
        sited_buses=str(plan_mw), n_sites=len(plan_mw),
        total_sited_mw=float(sum(plan_mw.values())) if plan_mw else 0.0,
        eue_no_storage=no["eue_mw"], eue_with_storage=wi["eue_mw"],
        eue_reduction_mw=no["eue_mw"] - wi["eue_mw"],
        eue_reduction_pct=(100.0 * (no["eue_mw"] - wi["eue_mw"]) / no["eue_mw"]
                           if no["eue_mw"] > 1e-9 else 0.0),
        secure_frac_no_storage=no["secure_frac"], secure_frac_with_storage=wi["secure_frac"],
        n_secure_no=no["n_secure"], n_secure_with=wi["n_secure"], n_outages=no["n_outages"],
        max_unserved_no=no["max_unserved_mw"], max_unserved_with=wi["max_unserved_mw"],
        congestion_cost_no=no["congestion_cost"], congestion_cost_with=wi["congestion_cost"],
        congestion_saving=no["congestion_cost"] - wi["congestion_cost"],
        n_islanding=no["n_islanding"],
    )


def main(smoke=False, per_grid=4, workers=None, levels_mw=None, shard=None, tag="",
         only_grid=None):
    t0 = time.time()
    reg = pd.read_csv(OUT_DIR / "registry_v2" / "instances_v2.csv")
    reg = reg[reg.grid.isin(MODULES)]
    if only_grid:
        reg = reg[reg.grid == only_grid]
    # non-trivial optima only: an instance whose optimum is "site nothing" cannot
    # show a siting response to load growth, so it carries no signal for this test
    reg = reg[reg.optimum_value.abs() > 0.1]

    levels = AI_LOAD_MW[:2] if smoke else (levels_mw or AI_LOAD_MW)
    tasks = []
    for grid, grp in reg.groupby("grid"):
        for _, r in grp.sort_values("instance_id").head(1 if smoke else per_grid).iterrows():
            for ai in levels:
                tasks.append((r.to_dict(), ai, 1))

    if shard:
        i, nsh = shard
        tasks = [t for j, t in enumerate(tasks) if j % nsh == i]
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 4
    W = min(workers or avail, len(tasks))
    threads = max(1, avail // max(1, W))
    tasks = [(t[0], t[1], threads) for t in tasks]
    print(f"{len(tasks)} tasks ({reg.grid.nunique()} grids x {per_grid} instances x "
          f"{len(levels)} AI-load levels), {W} workers x {threads} threads", flush=True)

    rows = []
    part = OUT_DIR / f"g1_resilience{tag}.partial.csv"
    with mp.Pool(W) as pool:
        for j, r in enumerate(pool.imap_unordered(_run, tasks), 1):
            rows.append(r)
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  [{j}/{len(tasks)}] {r['instance_id']} AI={r['ai_load_mw']:.0f}MW "
                  f"sites={r['n_sites']} ({r['total_sited_mw']:.0f}MW) "
                  f"EUE {r['eue_no_storage']:.2f}->{r['eue_with_storage']:.2f}MW "
                  f"secure {r['secure_frac_no_storage']:.3f}->{r['secure_frac_with_storage']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows).sort_values(["grid", "instance_id", "ai_load_mw"])
    out = OUT_DIR / (f"g1_resilience{tag}_SMOKE.csv" if smoke else f"g1_resilience{tag}.csv")
    df.to_csv(out, index=False)
    if part.exists():
        part.unlink()
    print(f"\nwrote {out} ({len(df)} rows)")

    print(f"\ninfeasible base DC-OPF rows: {int((~df.base_feasible).sum())} of {len(df)}")
    print("\n=== siting response and resilience by AI-load level ===")
    print(df.groupby(["grid", "ai_load_mw"]).agg(
        sites=("n_sites", "mean"), sited_MW=("total_sited_mw", "mean"),
        EUE_no=("eue_no_storage", "mean"), EUE_with=("eue_with_storage", "mean"),
        EUE_cut_pct=("eue_reduction_pct", "mean"),
        secure_no=("secure_frac_no_storage", "mean"),
        secure_with=("secure_frac_with_storage", "mean"),
        cong_saving=("congestion_saving", "mean")).round(3).to_string())
    print(f"\nMW per capacity tier: {MW_PER_CAPACITY_TIER}")
    print(f"Total wall-clock: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    mp.freeze_support()
    w, pg, lv, sh, tg, og = None, 4, None, None, "", None
    for a in sys.argv:
        if a.startswith("--workers="):
            w = int(a.split("=", 1)[1])
        if a.startswith("--per-grid="):
            pg = int(a.split("=", 1)[1])
        if a.startswith("--levels="):
            lv = tuple(float(v) for v in a.split("=", 1)[1].split(","))
        if a.startswith("--shard="):
            i, n = a.split("=", 1)[1].split("/")
            sh = (int(i), int(n))
        if a.startswith("--tag="):
            tg = a.split("=", 1)[1]
        if a.startswith("--grid="):
            og = a.split("=", 1)[1]
    main(smoke="--smoke" in sys.argv, per_grid=pg, workers=w, levels_mw=lv, shard=sh, tag=tg,
         only_grid=og)
