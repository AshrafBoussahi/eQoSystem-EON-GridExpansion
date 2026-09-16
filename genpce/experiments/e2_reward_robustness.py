"""E2 — reward design vs shot-noise robustness for GenPCE-ES.

Variants of the reward (all normalised by the graph-intrinsic bound ν; the optimum is never used):
exact cut; cut + λ·relaxed for several λ; cut + relaxed + margin; margin only. For each we report the
exact ratio, the ratio after one bit-swap pass, the ratio decoded from 1000- and 4000-shot estimates
(mean over 20 repetitions), and the median correlator magnitude of the best circuit.

Example::

    python -u experiments/e2_reward_robustness.py --sizes 60 252 --seeds 0 1 --budget 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, benchmark_instance, cubic_cset  # noqa: E402
from e1_budget_curves import shot_decoded  # noqa: E402

from genpce.pool import native_chain_pool  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train import EvolutionConfig, EvolutionarySearch, Reward  # noqa: E402

VARIANTS = {
    "exact": dict(relaxed_weight=0.0, margin_weight=0.0),
    "shaped_0.5": dict(relaxed_weight=0.5, margin_weight=0.0),
    "shaped_1": dict(relaxed_weight=1.0, margin_weight=0.0),
    "shaped_2": dict(relaxed_weight=2.0, margin_weight=0.0),
    "shaped_4": dict(relaxed_weight=4.0, margin_weight=0.0),
    "shaped_16": dict(relaxed_weight=16.0, margin_weight=0.0),
    "relaxed_only": dict(relaxed_weight=1000.0, margin_weight=0.0),
    "shaped_0.5+margin_0.3": dict(relaxed_weight=0.5, margin_weight=0.3),
    "margin_0.3": dict(relaxed_weight=0.0, margin_weight=0.3),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["reg3", "er4"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[60, 252])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--mu", type=int, default=10)
    ap.add_argument("--lam", type=int, default=50)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e2")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"reward_{args.tag}.csv"
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev.instance, prev.variant, prev.run))

    for family in args.families:
        for m in args.sizes:
            for seed in args.seeds:
                inst = benchmark_instance(family, m, seed)
                cset = cubic_cset(inst.m)
                bk = inst.best_known()
                pool = native_chain_pool(cset.n)
                evaluator = StatevectorEvaluator(cset)
                seq_len = int(round(args.seq_mult * cset.n))
                for variant in args.variants:
                    for run in range(args.runs):
                        if (inst.name, variant, run) in done:
                            continue
                        reward = Reward.shaped(inst, cset, **VARIANTS[variant])
                        es = EvolutionarySearch(inst, cset, pool, evaluator, reward, EvolutionConfig(seq_len=seq_len, budget=args.budget, mu=args.mu, lam=args.lam, seed=run))
                        df = es.run()
                        cut, cut_ls = es.best_with_local_search()
                        sd = shot_decoded(inst, cset, pool.to_circuit(es.best["tokens"]))
                        row = {"family": family, "m": inst.m, "n": cset.n, "instance": inst.name, "variant": variant, "run": run, "seq_mult": args.seq_mult,
                               "best_known": bk, "ratio_raw": cut / bk, "ratio_ls": cut_ls / bk,
                               "ratio_1000shots": sd[1000][0] / bk, "ratio_1000shots_min": sd[1000][1] / bk,
                               "ratio_4000shots": sd[4000][0] / bk, "ratio_4000shots_min": sd[4000][1] / bk,
                               "median_abs_corr": float(np.median(np.abs(es.best["corr"][: inst.m]))),
                               "min_abs_corr": float(np.min(np.abs(es.best["corr"][: inst.m]))), "seconds": df.seconds.iloc[-1]}
                        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
                        print(f"{inst.name:16s} {variant:22s} run={run} r_raw={row['ratio_raw']:.3f} r_ls={row['ratio_ls']:.3f} "
                              f"1000sh={row['ratio_1000shots']:.3f} 4000sh={row['ratio_4000shots']:.3f} med|c|={row['median_abs_corr']:.3f} {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
