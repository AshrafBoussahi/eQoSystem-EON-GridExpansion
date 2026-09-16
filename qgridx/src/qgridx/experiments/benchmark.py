"""E2b/E2h (DOE Phase 3 submission prep): re-run the main benchmark table on the
corrected, non-trivial registry (registry_v2), with every arm at a matched
10,000-evaluation budget.

Why this re-run exists: the published main table (IEEE-14 95.0% / IEEE-30 75.0%
/ IEEE-57 65.0% / IEEE-118 100%) was computed on instances where 79.4% of
optima were "site nothing" (IEEE-30 and IEEE-57: 100%), and on coefficients
built from a DC-OPF LMP expression that was wrong by a mean of 21.7 $/MWh.
Both are fixed in registry_v2 (see e2d/e2e for the LMP fix and its
finite-difference validation; e2f/e2g for the instance rebuild and calibration).
Trivial rate is now 4.4%, optima site 2.7-4.95 of 4-5 candidate buses, and the
budget cap binds on ~85% of instances.

Arms, all on identical instances:
  MIP        exact certified optimum (already in the registry, HiGHS)
  GQE+PCE    generative circuit synthesis, 10,000 circuit evaluations
  SA         simulated annealing, 10,000 evaluations   (reused unchanged from
  tabu       tabu search, 10,000 evaluations            paper_s2_w3a_*)
  greedy-1   single-start repair + joint local search
  greedy-20  best of 20 random restarts
  blind      10,000 random correlator vectors through the IDENTICAL decoder

Framing note carried over verbatim from Sprint X.16, because it still applies
and matters for how the submission reads: SA/tabu/greedy have DIRECT access to
the full QUBO (c, Q) on every evaluation. GQE+PCE never sees c or Q -- it only
receives a scalar decoded-cost reward after its candidate passes through the
6-qubit PCE measurement compression and the classical decoder. This is not
"same information, different search algorithm." The classical arms answer a
complementary question: are these instances classically easy when given full
information? The blind arm is the one that isolates the circuit's own
contribution, since it shares the decoder and the budget but carries no signal.

Protocol matches the published table's structure exactly: 10 instances per
config, 2 configs per grid (20 per grid, pooled), 2 training seeds per instance.
"""
import sys
import time
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd


from qgridx.baselines.heuristics import simulated_annealing, tabu_search
from qgridx.grid.cases import case14 as ieee_case14, case30 as ieee_case30, case57 as ieee_case57, case118 as ieee_case118
from qgridx.generator.executor import build_unitary_table
from qgridx.generator.reward import cheap_decode_cost
from qgridx.generator.train import RegimeAConfig, train_regime_a
from qgridx.generator.vocab import GQEVocab
from qgridx.problems.siting import build_instance
from qgridx.decoder.repair import joint_neighborhood_search, repair_budget, repair_domain_wall
from qgridx.encoding.families import random_assignment

from qgridx.utils.paths import results_dir

OUT_DIR = results_dir() / "doe_phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_BUDGET = 10_000
N_INST_PER_CONFIG = 10
SEEDS = [101, 102]
N_QUBITS, K = 6, 2
# must match experiments/e2f_build_nontrivial_registry.py exactly, or the
# rebuilt instances will not be the registry's instances. Enforced by an
# assertion against each row's recorded MIP optimum in rebuild().
BENEFIT_MAG = 20.0
BUDGET_RANGE = (0.15, 0.35)
COUPLING_MULT = 40.0

SCREENED = {
    "IEEE-14": [9, 10, 4, 14, 11, 3, 13, 12, 6],
    "IEEE-30": [8, 28, 29, 27, 30, 26, 25, 6, 9],
    "IEEE-57": [16, 12, 10, 51, 9, 50, 55, 11, 43],
    "IEEE-118": [106, 75, 107, 118, 42, 74, 76, 105, 41],
}
MODULES = {"IEEE-14": ieee_case14, "IEEE-30": ieee_case30,
           "IEEE-57": ieee_case57, "IEEE-118": ieee_case118}


def rebuild(row):
    """Rebuild the registry instance from its seed. Asserts the rebuilt QUBO
    reproduces the registry's recorded MIP optimum, so any calibration drift
    between this script and e2f fails loudly instead of silently benchmarking a
    different problem."""
    grid = row.grid
    buses = SCREENED[grid][:int(row.B)]
    inst = build_instance(seed=int(row.gen_seed), candidate_buses=buses, L=int(row.L),
                          budget_fraction_range=BUDGET_RANGE,
                          coupling_scale_mult=COUPLING_MULT,
                          case_module=MODULES[grid], lmp_reference_pool=SCREENED[grid],
                          benefit_magnitude=BENEFIT_MAG)
    x_opt = np.array(eval(row.levels_at_opt)) if isinstance(row.levels_at_opt, str) else None
    if x_opt is not None:
        # reconstruct the optimum bitstring from its per-bus levels and check cost
        x = np.zeros(inst.m, dtype=int)
        for bi, lev in enumerate(x_opt):
            s, _ = inst.bus_bit_slices[bi]
            x[s:s + int(lev)] = 1
        got, want = inst.cost(x), float(row.optimum_value)
        assert abs(got - want) < 1e-6, (
            f"{row.instance_id}: rebuilt instance does not match registry "
            f"(cost at recorded optimum {got:.6f} vs recorded {want:.6f}) -- "
            f"calibration constants in this script disagree with e2f")
    return inst


def greedy_restarts(inst, n_restarts, seed):
    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(n_restarts):
        x = repair_budget(repair_domain_wall(rng.integers(0, 2, size=inst.m), inst), inst)
        x = joint_neighborhood_search(x.copy(), inst, max_rounds=5)
        best = min(best, inst.cost(x))
    return best


def blind_matched(inst, budget, seed):
    """Matched-budget blind control: `budget` random correlator vectors pushed
    through the identical decoder GQE's reward uses, best-of kept. Stronger (and
    fairer) than the 15-trial blind controls earlier sprints used."""
    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(budget):
        pi = rng.uniform(-1, 1, size=inst.m)
        cost, _ = cheap_decode_cost(pi, inst, None)
        if cost < best:
            best = cost
    return best


def _run_instance(task):
    """One instance = one independent unit of work (all arms, all GQE seeds).
    Runs in a worker process; torch is pinned to 1 thread per worker so N workers
    do not oversubscribe the machine and thrash."""
    import torch
    torch.set_num_threads(1)
    row_dict, seeds, budget = task
    row = pd.Series(row_dict)
    inst = rebuild(row)
    opt = float(row.optimum_value)
    assignment = random_assignment(N_QUBITS, K, inst.m, seed=int(row.gen_seed))
    vocab = GQEVocab(n=N_QUBITS)
    ut = build_unitary_table(vocab)   # ~0.1 s at n=6, cheap enough to build per worker

    sa = simulated_annealing(inst, budget, seed=int(row.gen_seed))
    tb = tabu_search(inst, budget, seed=int(row.gen_seed))
    sa_cost = sa[0] if isinstance(sa, tuple) else sa
    tb_cost = tb[0] if isinstance(tb, tuple) else tb
    g1 = greedy_restarts(inst, 1, int(row.gen_seed))
    g20 = greedy_restarts(inst, 20, int(row.gen_seed))
    bl = blind_matched(inst, budget, int(row.gen_seed))

    out = []
    for s in seeds:
        cfg_gqe = RegimeAConfig(max_evals=budget, checkpoints=(budget,), seed=s)
        res = train_regime_a(inst, vocab, ut, N_QUBITS, assignment, [], cfg_gqe)
        out.append(dict(
            instance_id=row.instance_id, grid=row.grid, config=row.config, m=int(row.m),
            B=int(row.B), L=int(row.L), gqe_seed=s, optimum=opt,
            gqe_cost=float(res.best_cost), sa_cost=float(sa_cost), tabu_cost=float(tb_cost),
            greedy1_cost=float(g1), greedy20_cost=float(g20), blind_cost=float(bl),
            gqe_exact=bool(abs(res.best_cost - opt) < 1e-6),
            sa_exact=bool(abs(sa_cost - opt) < 1e-6),
            tabu_exact=bool(abs(tb_cost - opt) < 1e-6),
            greedy1_exact=bool(abs(g1 - opt) < 1e-6),
            greedy20_exact=bool(abs(g20 - opt) < 1e-6),
            blind_exact=bool(abs(bl - opt) < 1e-6),
            gqe_gap=float(res.best_cost - opt), blind_gap=float(bl - opt),
            n_gates=int((res.best_tokens != vocab.EOS_ID).sum() - 1),
        ))
    return out


def main(smoke=False, workers=None):
    t0 = time.time()
    reg = pd.read_csv(OUT_DIR / "registry_v2" / "instances_v2.csv")

    n_inst = 2 if smoke else N_INST_PER_CONFIG
    seeds = SEEDS[:1] if smoke else SEEDS
    budget = 200 if smoke else EVAL_BUDGET

    tasks = []
    for cfg, grp in reg.groupby("config"):
        for _, row in grp.sort_values("instance_id").head(n_inst).iterrows():
            tasks.append((row.to_dict(), seeds, budget))

    suffix = "_SMOKE" if smoke else ""
    out = OUT_DIR / f"e2h_main_table_v2{suffix}.csv"
    part = OUT_DIR / f"e2h_main_table_v2{suffix}.partial.csv"

    n_workers = workers or max(1, min(len(tasks), (os.cpu_count() or 4) - 4))
    print(f"{len(tasks)} instances x {len(seeds)} seeds, budget={budget}, {n_workers} workers")

    rows = []
    if n_workers == 1:
        for t in tasks:
            rows.extend(_run_instance(t))
    else:
        # imap_unordered + incremental checkpoint: partial results survive a crash,
        # which the first (serial, unbuffered, 5.5 h) attempt did not provide
        with mp.Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_instance, tasks), 1):
                rows.extend(res)
                pd.DataFrame(rows).to_csv(part, index=False)
                r0 = res[0]
                print(f"  [{i}/{len(tasks)}] {r0['instance_id']}: opt={r0['optimum']:.4f} "
                      f"GQE={r0['gqe_cost']:.4f} SA={r0['sa_cost']:.4f} tabu={r0['tabu_cost']:.4f} "
                      f"blind={r0['blind_cost']:.4f} [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if part.exists():
        part.unlink()
    print(f"\nWrote {out} ({len(df)} rows)")

    print("\n=== EXACT-MATCH RATE BY GRID (registry_v2, matched 10k budget) ===")
    cols = ["gqe_exact", "sa_exact", "tabu_exact", "greedy20_exact", "greedy1_exact", "blind_exact"]
    print(df.groupby("grid")[cols].mean().round(3).to_string())
    print("\n=== pooled ===")
    print(df[cols].mean().round(3).to_string())
    print("\n=== mean residual gap to optimum (continuous view) ===")
    print(df.groupby("grid")[["gqe_gap", "blind_gap"]].mean().to_string())
    print(f"\n=== GQE gate count (right-sizing check) ===")
    print(df.groupby("grid")["n_gates"].agg(["mean", "std", "min", "max"]).to_string())
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    mp.freeze_support()
    w = None
    for a in sys.argv:
        if a.startswith("--workers="):
            w = int(a.split("=", 1)[1])
    main(smoke="--smoke" in sys.argv, workers=w)
