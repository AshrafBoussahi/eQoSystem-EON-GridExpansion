"""E10 supplement — does a full-circle, finer angle vocabulary lift the ceilings?

Phase A found that the 25-token pool (angles 0, ±π/16, ±π/8, ±π/4, ±π/2) is what stops discrete
circuits from reproducing the teacher's correlations. Here two vocabularies are compared with the same
search and the same budgets:

* ``pool25``  — the current pool;
* ``pool94``  — full circle, π/16 spacing: R_a(±kπ/16), k = 1..16, three axes, identity (±π coincide).

Two tests: (i) representability search (correlation distance) at 8 layers for the held-out teacher
targets; (ii) the plain reward-driven search of the earlier work (shaped reward, 20k evaluations,
several seeds) — the question being whether the 0.89–0.90 plateau was a vocabulary artefact.
"""

from __future__ import annotations

import argparse
import sys
from math import pi
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, benchmark_instance, cubic_cset  # noqa: E402
from e1_budget_curves import shot_decoded  # noqa: E402

from genpce.pool import native_chain_pool  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train import EvolutionConfig, EvolutionarySearch, Reward  # noqa: E402
from genpce.train.distill import CorrelationTargetReward, VQACorpus, reconstruction_metrics  # noqa: E402

OUT = RESULTS / "e10"
POOLS = {"pool25": dict(angles=(pi / 2, pi / 4, pi / 8, pi / 16)), "pool94": dict(angles=tuple(k * pi / 16 for k in range(1, 17)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="reg3:0")
    ap.add_argument("--pools", nargs="+", default=["pool94", "pool25"])
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--reward-seeds", type=int, default=5)
    ap.add_argument("--skip-reward", action="store_true")
    ap.add_argument("--skip-ceiling", action="store_true")
    args = ap.parse_args()
    fam, iseed = args.instance.split(":")
    inst = benchmark_instance(fam, 60, int(iseed))
    cset = cubic_cset(inst.m)
    ev = StatevectorEvaluator(cset)
    bk = inst.best_known()
    seq_len = args.L * cset.n
    corpus = VQACorpus.load(str(OUT / f"corpus_{inst.name}.npz"))
    targets = pd.read_csv(OUT / f"targets_{inst.name}.csv")
    held = targets[targets.holdout & (targets.stage == "final")]
    ceil_path, rew_path = OUT / f"vocab_ceiling_{inst.name}_L{args.L}.csv", OUT / f"vocab_reward_{inst.name}_L{args.L}.csv"
    for pname in args.pools:
        pool = native_chain_pool(cset.n, **POOLS[pname])
        print(f"{pname}: {pool.size} tokens", flush=True)
        if not args.skip_ceiling:
            for t in held.itertuples():
                c_star = corpus.corr[t.corpus_index]
                cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, seed=int(t.seed), max_mutations=2, log_every=10, record_database=False)
                es = EvolutionarySearch(inst, cset, pool, ev, CorrelationTargetReward.for_target(inst, cset, c_star), cfg)
                t0 = perf_counter()
                df = es.run()
                met = reconstruction_metrics(es.best["corr"], c_star, inst)
                curve = [(int(r.evaluations), float(-r.best_score)) for r in df.itertuples()]
                row = {"pool": pname, "tokens": pool.size, "target": t.target, "teacher_layers": t.teacher_layers, "discrete_layers": args.L, **met,
                       "D_c_at_5000": float(np.interp(5000, *zip(*curve))), "seconds": perf_counter() - t0}
                pd.DataFrame([row]).to_csv(ceil_path, mode="a", header=not ceil_path.exists(), index=False)
                print(f"  ceiling {pname} {t.target:14s} D_c={met['D_c']:.4f} A_x={met['A_x']:.3f} r={met['ratio']:.3f} (target {met['target_ratio']:.3f}) |c|={met['median_abs_corr']:.3f} {row['seconds']:.0f}s", flush=True)
        if not args.skip_reward:
            for seed in range(args.reward_seeds):
                cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, seed=seed, max_mutations=2, log_every=5, record_database=False)
                es = EvolutionarySearch(inst, cset, pool, ev, Reward.shaped(inst, cset, 0.5), cfg)
                marks = {}
                t0 = perf_counter()
                df = es.run()
                sd = shot_decoded(inst, cset, pool.to_circuit(es.best["tokens"]), shots_list=(1000,), reps=20)
                r5k = float(np.interp(5000, df.evaluations, np.maximum.accumulate(df.ratio_best)))
                row = {"pool": pname, "tokens": pool.size, "seed": seed, "ratio_5k": r5k, "ratio_final": es.best["cut"] / bk, "ratio_1000shots": sd[1000][0] / bk,
                       "median_abs_corr": float(np.median(np.abs(es.best["corr"][: inst.m]))), "seconds": perf_counter() - t0}
                pd.DataFrame([row]).to_csv(rew_path, mode="a", header=not rew_path.exists(), index=False)
                print(f"  reward  {pname} seed={seed} ratio@5k={r5k:.3f} final={row['ratio_final']:.3f} 1000sh={row['ratio_1000shots']:.3f} |c|={row['median_abs_corr']:.3f} {row['seconds']:.0f}s", flush=True)
    if ceil_path.exists():
        c = pd.read_csv(ceil_path)
        print("\nceiling by pool:\n", c.groupby("pool")[["D_c", "D_c_at_5000", "A_x", "ratio", "target_ratio", "median_abs_corr"]].mean().round(4).to_string())
    if rew_path.exists():
        r = pd.read_csv(rew_path)
        print("\nreward search by pool:\n", r.groupby("pool")[["ratio_5k", "ratio_final", "ratio_1000shots", "median_abs_corr"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
