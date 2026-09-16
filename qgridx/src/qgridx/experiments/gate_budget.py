"""E2o (DOE Phase 3 prep): final lever sweep for GQE+PCE before writing up.

Established by E2l/E2m (all at matched 10,000 evaluations, registry_v2):
  * Gate ceiling is the binding constraint. 3x default took exact-match
    23.8% -> 48.8% on 80 instances (p=2.4e-07), reaching statistical parity with
    the unconstrained vector-blind control (p=0.402) but still behind SA
    (65.0%, p=0.00047) and tabu (53.8%, p=0.033).
  * ceil3x STILL saturates: 59.0/59 gates on all 80 instances. The constraint has
    not actually been relieved, only loosened.
  * Not levers: extra qubits (capacity headroom), margin-aware reward,
    local-search rounds, graph-aware assignment on its own.
  * Marginal: k=3 (p=0.050); graph-aware helps only combined with a raised ceiling.

Five hypotheses tested here, each budget-matched at 10,000 evaluations so no arm
buys quality with extra compute.

H-A  CEILING 6x / 8x. ceil3x saturates, so keep raising until either the model
     stops saturating (it has found a preferred depth) or quality plateaus (depth
     is no longer the constraint). This is the direct continuation.

H-B  CEILING x k=3 COMBINED. Both helped alone; k=3 widens the reachable
     sign-pattern set at fixed qubit count while depth enriches the state. If the
     two mechanisms are independent the gains should partially add.

H-C  CEILING x GRAPH-AWARE COMBINED. graph_ceil2x matched the best 24-instance
     median (1.000) while graph alone did nothing, suggesting the assignment only
     pays off once the circuit is deep enough to exploit it.

H-D  RICHER ANGLE GRID. The vocabulary ships 8 fixed angles
     (+-pi/3, pi/4, pi/5, pi/8). Angle resolution is an expressivity axis
     ORTHOGONAL to depth: more reachable rotations per gate means a richer state
     at the same gate count. Never varied in any reported run. Tested at 16
     angles.

H-E  BUDGET SPLIT ACROSS RESTARTS. The vector-blind control's strength comes from
     10,000 INDEPENDENT samples, while GQE spends its whole budget refining one
     policy from one initialisation. 5 independent trainings of 2,000 evaluations
     each (same 10,000 total) tests whether exploration breadth beats depth of
     refinement -- directly motivated by why blind is competitive at all.

Primary metric: median approximation ratio (cost/optimum). The do-nothing guard
(cost := min(cost, 0)) is applied to every arm.
"""
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from qgridx.grid.cases import case14 as ieee_case14, case30 as ieee_case30, case57 as ieee_case57, case118 as ieee_case118
from qgridx.generator.executor import build_unitary_table
from qgridx.generator.train import RegimeAConfig, train_regime_a
from qgridx.generator.vocab import GQEVocab
from qgridx.problems.siting import build_instance
from qgridx.encoding.families import graph_aware_assignment, max_capacity, random_assignment

from qgridx.utils.paths import results_dir

OUT_DIR = results_dir() / "doe_phase3"

BENEFIT_MAG, BUDGET_RANGE, COUPLING_MULT = 20.0, (0.15, 0.35), 40.0
N_QUBITS = 6
EVAL_BUDGET = 10_000
SEED = 101
SHARD = (0, 1)   # (index, total); set by --shard=i/n

SCREENED = {
    "IEEE-14": [9, 10, 4, 14, 11, 3, 13, 12, 6],
    "IEEE-30": [8, 28, 29, 27, 30, 26, 25, 6, 9],
    "IEEE-57": [16, 12, 10, 51, 9, 50, 55, 11, 43],
    "IEEE-118": [106, 75, 107, 118, 42, 74, 76, 105, 41],
}
MODULES = {"IEEE-14": ieee_case14, "IEEE-30": ieee_case30,
           "IEEE-57": ieee_case57, "IEEE-118": ieee_case118}

PI = np.pi
ANGLES_8 = None      # None -> vocabulary default (8 angles)
ANGLES_16 = [PI / 2, -PI / 2, PI / 3, -PI / 3, PI / 4, -PI / 4, PI / 5, -PI / 5,
             PI / 6, -PI / 6, PI / 8, -PI / 8, PI / 12, -PI / 12, 3 * PI / 4, -3 * PI / 4]

# label: (ceiling multiplier, assignment, k, angle_grid, n_restarts)
# graph-aware is excluded: already measured at 2x in E2m and marginal there
# (no effect alone, p=0.31). Everything else is kept so the ceiling plateau can
# actually be located -- ceil3x saturates at 59/59, so 6x and 8x are both needed
# to tell "more depth helps" from "depth is no longer the constraint".
# Per-instance checkpointing is on regardless, so an interrupted run keeps its
# completed instances.
CONFIGS = {
    "ceil3x_ref":     (3, "random", 2, ANGLES_8, 1),    # reference: current best
    "ceil6x":         (6, "random", 2, ANGLES_8, 1),    # H-A  more depth
    "ceil8x":         (8, "random", 2, ANGLES_8, 1),    # H-A  find the plateau
    "ceil3x_k3":      (3, "random", 3, ANGLES_8, 1),    # H-B  depth x higher-body
    "ceil3x_ang16":   (3, "random", 2, ANGLES_16, 1),   # H-D  angle resolution
    "ceil3x_split5":  (3, "random", 2, ANGLES_8, 5),    # H-E  breadth vs refinement
}


def rebuild(row):
    return build_instance(seed=int(row["gen_seed"]),
                          candidate_buses=SCREENED[row["grid"]][:int(row["B"])],
                          L=int(row["L"]), budget_fraction_range=BUDGET_RANGE,
                          coupling_scale_mult=COUPLING_MULT,
                          case_module=MODULES[row["grid"]],
                          lmp_reference_pool=SCREENED[row["grid"]],
                          benefit_magnitude=BENEFIT_MAG)


def _run(task):
    """ONE (instance, config) cell. Deliberately fine-grained: an earlier version
    did all 6 configs per task, so a worker had to run ~26 gate-units (tens of
    minutes) before emitting anything, and every interruption of the pod lost the
    entire run with an empty partial file. One cell per task means the partial CSV
    grows every few minutes and any uptime window produces usable results."""
    row, label, budget, threads = task
    torch.set_num_threads(max(1, int(threads)))
    mult, assign_kind, k, angles, n_restarts = CONFIGS[label]
    inst = rebuild(row)
    opt = float(row["optimum_value"])
    gseed = int(row["gen_seed"])
    if max_capacity(N_QUBITS, k) < inst.m:
        return None
    assign = (graph_aware_assignment(N_QUBITS, k, inst.m, inst.Q, seed=gseed)
              if assign_kind == "graph"
              else random_assignment(N_QUBITS, k, inst.m, seed=gseed))
    vocab = GQEVocab(n=N_QUBITS, angle_grid=angles, max_len=(3 * N_QUBITS + 2) * mult)
    ut = build_unitary_table(vocab)
    # budget is SPLIT across restarts so every arm sees exactly `budget` evals
    per = max(1, budget // n_restarts)
    best, gates = np.inf, np.nan
    for r in range(n_restarts):
        cfg = RegimeAConfig(max_evals=per, checkpoints=(per,), seed=SEED + 1000 * r)
        res = train_regime_a(inst, vocab, ut, N_QUBITS, assign, [], cfg)
        if float(res.best_cost) < best:
            best = float(res.best_cost)
            gates = int((res.best_tokens != vocab.EOS_ID).sum() - 1)
    return dict(instance_id=row["instance_id"], grid=row["grid"], config=row["config"],
                m=int(inst.m), optimum=opt, arm=label,
                cost=min(best, 0.0), gates=gates,
                ceiling=vocab.max_len - 1, vocab_size=vocab.vocab_size)


def main(smoke=False, workers=None, per_cfg=3):
    t0 = time.time()
    reg = pd.read_csv(OUT_DIR / "registry_v2" / "instances_v2.csv")
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 4
    budget = 320 if smoke else EVAL_BUDGET
    per_cfg = 1 if smoke else per_cfg

    # --shard=i/n splits the independent cells across several pods. Each shard
    # writes its own partial/output so they never race on the same file; merge
    # locally afterwards (the cell table is long-format, so a concat is enough).
    shard_i, shard_n = SHARD
    stag = "" if shard_n == 1 else f"_s{shard_i}of{shard_n}"
    sfx = ("_SMOKE" if smoke else "") + stag
    part = OUT_DIR / f"e2o_final_levers{sfx}.partial.csv"
    out = OUT_DIR / f"e2o_final_levers{sfx}.csv"

    instances = []
    for cfg, grp in reg.groupby("config"):
        for _, r in grp.sort_values("instance_id").head(per_cfg).iterrows():
            instances.append(r.to_dict())

    # resume: skip (instance, arm) cells already present in the partial
    done = set()
    rows = []
    if part.exists():
        prev = pd.read_csv(part)
        rows = prev.to_dict("records")
        done = {(r["instance_id"], r["arm"]) for r in rows}
        print(f"resuming: {len(done)} cells already complete", flush=True)

    cells = [(r, lab) for r in instances for lab in CONFIGS
             if (r["instance_id"], lab) not in done]
    # cheapest arms first, so an interruption loses the least work
    cells.sort(key=lambda c: CONFIGS[c[1]][0] * CONFIGS[c[1]][4])
    if shard_n > 1:
        cells = [c for i, c in enumerate(cells) if i % shard_n == shard_i]
        print(f"shard {shard_i}/{shard_n}: {len(cells)} cells assigned", flush=True)

    W = min(workers or avail, max(1, len(cells)))
    threads = max(1, avail // max(1, W))
    tasks = [(r, lab, budget, threads) for r, lab in cells]
    print(f"{len(instances)} instances x {len(CONFIGS)} arms = {len(cells)} cells to run, "
          f"budget={budget}, {W} workers x {threads} threads "
          f"(= {W*threads} of {avail} cores)", flush=True)

    if tasks:
        with mp.Pool(W) as pool:
            for j, r in enumerate(pool.imap_unordered(_run, tasks), 1):
                if r is None:
                    continue
                rows.append(r)
                pd.DataFrame(rows).to_csv(part, index=False)   # checkpoint EVERY cell
                print(f"  [{j}/{len(tasks)}] {r['instance_id']:28s} {r['arm']:14s} "
                      f"opt={r['optimum']:8.3f} cost={r['cost']:8.3f} gates={r['gates']}"
                      f" [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} cells)")
    summarize(df)
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s")


def summarize(df):
    """Aggregate the long-format cell table. Works on a partial file too, which is
    the point of the per-cell layout."""
    if df.empty:
        print("no cells yet")
        return
    piv = df.pivot_table(index="instance_id", columns="arm", values="cost")
    opt = df.groupby("instance_id").optimum.first()
    piv = piv.join(opt.rename("optimum"))
    nt = piv[piv.optimum.abs() > 1e-9]
    res = []
    for arm in sorted(df.arm.unique()):
        if arm not in piv:
            continue
        sub = df[df.arm == arm]
        col = piv[arm].dropna()
        o = piv.loc[col.index, "optimum"]
        ntm = o.abs() > 1e-9
        res.append(dict(arm=arm, n=len(col),
                        median_ratio=round(float((col[ntm] / o[ntm]).median()), 4),
                        mean_gap=round(float((col - o).mean()), 4),
                        exact_pct=round(100 * float(np.mean(np.abs(col - o) < 1e-6)), 1),
                        mean_gates=round(float(sub.gates.mean()), 1),
                        ceiling=int(sub.ceiling.iloc[0]),
                        saturated=bool(abs(sub.gates.mean() - sub.ceiling.iloc[0]) < 1.0)))
    print("\n=== PRIMARY: median approximation ratio (1.000 = optimal) ===")
    print(pd.DataFrame(res).sort_values("median_ratio", ascending=False).to_string(index=False))
    if "ceil3x_ref" in piv:
        print("\n=== paired vs ceil3x_ref (instances where both arms completed) ===")
        try:
            from scipy.stats import wilcoxon
            for arm in sorted(df.arm.unique()):
                if arm == "ceil3x_ref" or arm not in piv:
                    continue
                both = piv[[arm, "ceil3x_ref"]].dropna()
                if len(both) < 3:
                    print(f"   {arm:16s} only {len(both)} paired instances -- skipping")
                    continue
                a, b = both[arm], both["ceil3x_ref"]
                better, worse = int((a < b - 1e-9).sum()), int((a > b + 1e-9).sum())
                try:
                    _, p = wilcoxon(a, b)
                    ps = f"{p:.3g}"
                except Exception:
                    ps = "all-tied"
                print(f"   {arm:16s} n={len(both):2d} better {better:2d}, worse {worse:2d}, "
                      f"ties {len(both)-better-worse:2d}, mean_delta={float((a-b).mean()):+.4f}, p={ps}")
        except Exception as e:
            print("   wilcoxon unavailable:", e)


if __name__ == "__main__":
    mp.freeze_support()
    w, pc = None, 3
    for a in sys.argv:
        if a.startswith("--shard="):
            _i, _n = a.split("=", 1)[1].split("/")
            SHARD = (int(_i), int(_n))
        if a.startswith("--workers="):
            w = int(a.split("=", 1)[1])
        if a.startswith("--per-config="):
            pc = int(a.split("=", 1)[1])
    main(smoke="--smoke" in sys.argv, workers=w, per_cfg=pc)
