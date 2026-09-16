"""Sprint 2, WS-3a: strengthened certification via simulated annealing and
tabu search (Sprint 1's unresolved G8/G12). Adds two heuristic-resistance
tiers to the registry's existing "greedy1 < 50%" certification (S1.0a):
non-trivial (greedy-1 < 50%, already established) and heuristic-resistant
(SA & tabu < 50% success rate too, new this sprint).

Move space: both heuristics operate on the domain-wall "level" representation
per bus (level in [0,L], monotone leading-ones encoding) rather than raw bit
flips, since an arbitrary bit flip can break domain-wall monotonicity;
level +/-1 moves are always monotonicity-preserving by construction. Budget
feasibility is enforced by rejecting (not repairing) any move that would
exceed budget_cap -- matching a genuine local-search's inability to "cheat"
past the constraint.

10,000 cost-evaluation budget each (SA and tabu), matching the plan.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qgridx.analysis.stats import wilson_ci

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "paper_sprint2" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUDGET_EVALS = 10000
TABU_TENURE = 10
SA_T0, SA_TMIN = 2.0, 0.01


def levels_to_x(levels, L, B, m):
    x = np.zeros(m, dtype=int)
    ptr = 0
    for b in range(B):
        x[ptr:ptr + int(levels[b])] = 1
        ptr += L
    return x


def _feasible_random_start(inst, rng):
    L, B, m = inst.L, inst.B, inst.m
    levels = rng.integers(0, L + 1, size=B)
    x = levels_to_x(levels, L, B, m)
    guard = 0
    while inst.budget_used(x) > inst.budget_cap + 1e-9 and levels.sum() > 0 and guard < 10 * B:
        b = rng.integers(0, B)
        if levels[b] > 0:
            levels[b] -= 1
            x = levels_to_x(levels, L, B, m)
        guard += 1
    return levels


def simulated_annealing(inst, budget_evals, seed):
    rng = np.random.default_rng(seed)
    L, B, m = inst.L, inst.B, inst.m
    levels = _feasible_random_start(inst, rng)
    x = levels_to_x(levels, L, B, m)
    cur_cost = inst.cost(x)
    best_cost, best_levels = cur_cost, levels.copy()
    evals = 0
    step = 0
    while evals < budget_evals:
        step += 1
        T = SA_T0 * (SA_TMIN / SA_T0) ** min(step / budget_evals, 1.0)
        b = rng.integers(0, B)
        delta = rng.choice([-1, 1])
        nl = levels[b] + delta
        if nl < 0 or nl > L:
            continue
        new_levels = levels.copy()
        new_levels[b] = nl
        x_new = levels_to_x(new_levels, L, B, m)
        if inst.budget_used(x_new) > inst.budget_cap + 1e-9:
            continue
        new_cost = inst.cost(x_new)
        evals += 1
        d = new_cost - cur_cost
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            levels, cur_cost = new_levels, new_cost
            if cur_cost < best_cost:
                best_cost, best_levels = cur_cost, levels.copy()
    return best_cost


def tabu_search(inst, budget_evals, seed, tenure=TABU_TENURE):
    rng = np.random.default_rng(seed)
    L, B, m = inst.L, inst.B, inst.m
    levels = _feasible_random_start(inst, rng)
    x = levels_to_x(levels, L, B, m)
    cur_cost = inst.cost(x)
    best_cost, best_levels = cur_cost, levels.copy()
    tabu = {}
    evals = 0
    step = 0
    while evals < budget_evals:
        step += 1
        candidates = []
        for b in range(B):
            for delta in (-1, 1):
                nl = levels[b] + delta
                if nl < 0 or nl > L:
                    continue
                new_levels = levels.copy()
                new_levels[b] = nl
                x_new = levels_to_x(new_levels, L, B, m)
                if inst.budget_used(x_new) > inst.budget_cap + 1e-9:
                    continue
                c = inst.cost(x_new)
                evals += 1
                is_tabu = tabu.get((b, nl), -1) > step
                candidates.append((c, b, nl, is_tabu))
                if evals >= budget_evals:
                    break
            if evals >= budget_evals:
                break
        if not candidates:
            break
        non_tabu = [cand for cand in candidates if not cand[3]]
        pool = non_tabu if non_tabu else candidates
        c, b, nl, _ = min(pool, key=lambda t: t[0])
        tabu[(b, int(levels[b]))] = step + tenure
        levels = levels.copy()
        levels[b] = nl
        cur_cost = c
        if cur_cost < best_cost:
            best_cost, best_levels = cur_cost, levels.copy()
    return best_cost


def main():
    t0 = time.time()
    registry = pd.read_csv(ROOT / "results" / "paper_sprint1" / "registry" / "instances.csv")

    # Reconstruct QUBOInstance objects the same way every other Sprint-1/2 script does
    from qgridx.grid.cases import case14 as ieee_case14, case30 as ieee_case30, case57 as ieee_case57, case118 as ieee_case118
    from qgridx.problems.siting import build_instance

    pools = {
        "IEEE-14": (ieee_case14, [4, 5, 9, 10, 14, 3, 7, 11, 12, 13, 1, 2, 6, 8]),
        "IEEE-30": (ieee_case30, [4, 7, 10, 12, 15, 17, 19, 21, 24]),
        "IEEE-57": (ieee_case57, [3, 8, 13, 20, 25, 30, 35, 40, 45, 50, 55]),
        "IEEE-118": (ieee_case118, [3, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84]),
    }

    rows = []
    for (grid, cfg), group in registry.groupby(["grid", "config"]):
        if cfg == "compression_k3":
            continue  # different pool/case_module convention, out of scope for this pass
        if grid not in pools:
            continue
        mod, pool = pools[grid]
        B, L = int(group.iloc[0]["B"]), int(group.iloc[0]["L"])
        buses = pool[:B]
        for _, row in group.iterrows():
            inst = build_instance(seed=int(row["gen_seed"]), candidate_buses=buses, L=L,
                                   budget_fraction_range=(0.5, 0.8), coupling_scale_mult=4.0,
                                   case_module=mod, lmp_reference_pool=pool)
            oracle_cost = row["optimum_value"]
            sa_cost = simulated_annealing(inst, BUDGET_EVALS, seed=int(row["gen_seed"]) + 1)
            tabu_cost = tabu_search(inst, BUDGET_EVALS, seed=int(row["gen_seed"]) + 2)
            rows.append(dict(instance_id=row["instance_id"], grid=grid, config=cfg,
                              sa_exact=abs(sa_cost - oracle_cost) < 1e-6,
                              tabu_exact=abs(tabu_cost - oracle_cost) < 1e-6,
                              greedy1_rate=row["cert_greedy1_rate"]))
        print(f"{grid}/{cfg}: done ({time.time()-t0:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "W3a_sa_tabu_certification.csv", index=False)

    print("\n=== SA / tabu success rate by config, with n and Wilson CI (10k-eval budget each) ===")
    summary_rows = []
    for (grid, cfg), g in df.groupby(["grid", "config"]):
        sa_est = wilson_ci(int(g.sa_exact.sum()), len(g))
        tabu_est = wilson_ci(int(g.tabu_exact.sum()), len(g))
        greedy1 = g.greedy1_rate.iloc[0]
        heuristic_resistant = sa_est.value < 0.5 and tabu_est.value < 0.5
        tier = "heuristic-resistant" if heuristic_resistant else "non-trivial only"
        print(f"{grid}/{cfg}: greedy1={greedy1:.1%}  SA={sa_est}  tabu={tabu_est}  tier={tier}")
        summary_rows.append(dict(grid=grid, config=cfg, greedy1_rate=greedy1,
                                  sa_rate=sa_est.value, tabu_rate=tabu_est.value, tier=tier))
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "W3a_certification_tiers.csv", index=False)

    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
