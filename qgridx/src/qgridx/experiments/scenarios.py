"""E3b (DOE Phase 3 submission prep): build scenario-weighted siting QUBOs from
the RTS-GMLC scenario set, i.e. the two-stage stochastic formulation the
challenge asks for ("multi-scenario uncertainty in load and generation").

Formulation. Siting/sizing is a FIRST-STAGE decision: one x must be chosen
before the operating scenario is known. Operational value is scenario-dependent.
Taking the expectation over the scenario set gives

    minimize  E_s[ cost_s(x) ]  =  sum_s p_s ( c_s' x + x' Q_s x )
                               =  ( sum_s p_s c_s )' x  +  x' ( sum_s p_s Q_s ) x

so the expected-value problem is itself a QUBO with c = sum_s p_s c_s and
Q = sum_s p_s Q_s. That is what this script constructs. This matters for the
submission because it means multi-scenario uncertainty costs NOTHING extra at
the quantum layer: the qubit count, circuit depth, shot budget and measurement
settings are all identical to the single-scenario case. Scenario richness is
absorbed entirely in classical pre-processing. For a planner that is the
practically relevant property -- scenario coverage is limited by DC-OPF solve
time, not by quantum resources.

Per-scenario network conditions come from re-solving DC-OPF at that scenario's
load level, with the instance's own stochastic per-bus load SHAPE held fixed
(so scenarios vary the operating condition, not the instance identity). Each
scenario therefore has its own LMPs and congestion duals, hence its own c_s and
Q_s.

Also reports, because they are the interesting planner-facing numbers:
  * how much the optimal siting decision CHANGES between the deterministic
    mean-load instance and the scenario-weighted one (does uncertainty move the
    answer? if not, the scenario machinery is not earning its keep)
  * the value of the stochastic solution: how the scenario-weighted decision
    performs under the tail scenario versus the deterministic decision
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


from qgridx.grid.cases import case14 as ieee_case14, case30 as ieee_case30, case57 as ieee_case57, case118 as ieee_case118
from qgridx.problems.siting import build_instance
from qgridx.baselines.mip import solve_mip
from qgridx.utils.seeding import stable_seed

from qgridx.utils.paths import results_dir

OUT_DIR = results_dir() / "doe_phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCREENED = {
    "IEEE-14": [9, 10, 4, 14, 11, 3, 13, 12, 6],
    "IEEE-30": [8, 28, 29, 27, 30, 26, 25, 6, 9],
    "IEEE-57": [16, 12, 10, 51, 9, 50, 55, 11, 43],
    "IEEE-118": [106, 75, 107, 118, 42, 74, 76, 105, 41],
}
MODULES = {"IEEE-14": ieee_case14, "IEEE-30": ieee_case30,
           "IEEE-57": ieee_case57, "IEEE-118": ieee_case118}

B, L = 5, 9
BENEFIT_MAG, BUDGET_RANGE, COUPLING_MULT = 20.0, (0.15, 0.35), 40.0
N_INSTANCES = 10


def base_shape(mod, seed):
    """The instance's own stochastic per-bus load shape, normalized to mean 1, so
    a scenario's net-load level multiplies it cleanly."""
    rng = np.random.default_rng(seed)
    shape = rng.uniform(0.7, 1.3, size=mod.N_BUS)
    return shape / shape.mean()


def build_weighted(grid, mod, seed, scen):
    """Probability-weighted c and Q over the scenario set."""
    buses = SCREENED[grid][:B]
    shape = base_shape(mod, seed)
    c_acc, Q_acc = None, None
    per_scen = []
    for _, s in scen.iterrows():
        lsv = shape * float(s.net_load_scale)
        inst = build_instance(seed=seed, candidate_buses=buses, L=L,
                              budget_fraction_range=BUDGET_RANGE,
                              coupling_scale_mult=COUPLING_MULT,
                              case_module=mod, lmp_reference_pool=SCREENED[grid],
                              benefit_magnitude=BENEFIT_MAG,
                              load_scale_vector=lsv)
        p = float(s.prob)
        c_acc = inst.c * p if c_acc is None else c_acc + inst.c * p
        Q_acc = inst.Q * p if Q_acc is None else Q_acc + inst.Q * p
        per_scen.append((s.scenario_id, p, inst))
    # the weighted instance reuses the last build's structure (identical B/L/
    # budget/slices by construction) with expectation-weighted coefficients
    weighted = per_scen[-1][2]
    weighted.c = c_acc
    weighted.Q = Q_acc
    return weighted, per_scen, shape


def main():
    scen = pd.read_csv(OUT_DIR / "e3a_scenarios.csv")
    print(f"Scenario set: {len(scen)} scenarios, probabilities sum to {scen.prob.sum():.6f}")
    tail = scen[scen.kind == "tail"].iloc[0]
    print(f"Tail scenario '{tail.scenario_id}': p={tail.prob:.3f}, net_load={tail.net_load_scale:.3f}x mean\n")

    t0 = time.time()
    rows = []
    for grid, mod in MODULES.items():
        for i in range(N_INSTANCES):
            seed = stable_seed(f"e3_{grid}", "scen", i)
            wt, per_scen, shape = build_weighted(grid, mod, seed, scen)
            x_wt, cost_wt, _, st_wt = solve_mip(wt, time_limit_s=60.0)
            if st_wt != "optimal":
                continue
            lev_wt = wt.levels_from_x(np.asarray(x_wt, dtype=int))

            # deterministic mean-load counterpart (single scenario at net_load=1)
            det = build_instance(seed=seed, candidate_buses=SCREENED[grid][:B], L=L,
                                 budget_fraction_range=BUDGET_RANGE,
                                 coupling_scale_mult=COUPLING_MULT,
                                 case_module=mod, lmp_reference_pool=SCREENED[grid],
                                 benefit_magnitude=BENEFIT_MAG,
                                 load_scale_vector=shape)
            x_det, cost_det, _, st_det = solve_mip(det, time_limit_s=60.0)
            if st_det != "optimal":
                continue
            lev_det = det.levels_from_x(np.asarray(x_det, dtype=int))

            # value of the stochastic solution, evaluated under the tail scenario
            tail_inst = [ps[2] for ps in per_scen if ps[0] == tail.scenario_id][0]
            tail_cost_wt = tail_inst.cost(np.asarray(x_wt, dtype=int))
            tail_cost_det = tail_inst.cost(np.asarray(x_det, dtype=int))

            rows.append(dict(
                grid=grid, instance=i, seed=seed,
                cost_weighted=cost_wt, cost_deterministic=cost_det,
                levels_weighted=str(lev_wt.tolist()), levels_deterministic=str(lev_det.tolist()),
                decision_changed=bool(not np.array_equal(lev_wt, lev_det)),
                n_sites_weighted=int((lev_wt > 0).sum()), n_sites_det=int((lev_det > 0).sum()),
                total_levels_weighted=int(lev_wt.sum()), total_levels_det=int(lev_det.sum()),
                tail_cost_stochastic_decision=float(tail_cost_wt),
                tail_cost_deterministic_decision=float(tail_cost_det),
                vss=float(tail_cost_det - tail_cost_wt),  # >0 => stochastic decision better in tail
            ))
        sub = [r for r in rows if r["grid"] == grid]
        print(f"{grid:9s}: decision changed on {np.mean([r['decision_changed'] for r in sub]):5.1%} of instances  "
              f"sites {np.mean([r['n_sites_det'] for r in sub]):.2f}(det) -> "
              f"{np.mean([r['n_sites_weighted'] for r in sub]):.2f}(stoch)  "
              f"mean VSS in tail = {np.mean([r['vss'] for r in sub]):+.4f}  [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "e3b_scenario_weighted.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'e3b_scenario_weighted.csv'} ({len(df)} instances)")
    print("\n=== summary ===")
    print(f"scenarios per instance         : {len(scen)} (12 clusters + 1 correlated tail)")
    print(f"DC-OPF solves per instance     : {len(scen)}")
    print(f"quantum resources per instance : UNCHANGED vs single-scenario "
          f"(n=6 qubits, k=2, m={B*L}, 3 measurement settings)")
    print(f"siting decision changed by uncertainty: {df.decision_changed.mean():.1%} of instances")
    print(f"mean value of stochastic solution under the tail scenario: {df.vss.mean():+.4f}")
    print(f"  (positive = the scenario-weighted decision costs less than the "
          f"deterministic one when the tail actually occurs)")
    print(f"instances where stochastic decision is strictly better in tail: "
          f"{(df.vss > 1e-9).mean():.1%}; strictly worse: {(df.vss < -1e-9).mean():.1%}")
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
