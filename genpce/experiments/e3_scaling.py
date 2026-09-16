"""E3 — how the quantum budget needed by GenPCE-ES scales with the problem size m.

Runs the evolutionary search with a large budget on instances of increasing m and records the full
best-so-far curve, so that "circuit executions to reach a raw ratio of 0.8 / 0.85 / 0.9" can be read
off and compared with the parameter-shift cost of PCE-VQA (E1 curves). Shot-decoded quality of the
final circuit is recorded as in E1.

Example::

    python -u experiments/e3_scaling.py --sizes 105 168 252 --budget 100000
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

THRESHOLDS = (0.75, 0.8, 0.85, 0.9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["reg3"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[105, 168, 252])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--budget", type=int, default=100000)
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--relaxed-weight", type=float, default=0.5)
    ap.add_argument("--mu", type=int, default=10)
    ap.add_argument("--lam", type=int, default=50)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e3")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_path, final_path = args.out_dir / f"curves_{args.tag}.csv", args.out_dir / f"final_{args.tag}.csv"

    for family in args.families:
        for m in args.sizes:
            for seed in args.seeds:
                inst = benchmark_instance(family, m, seed)
                cset = cubic_cset(inst.m)
                bk = inst.best_known()
                pool = native_chain_pool(cset.n)
                evaluator = StatevectorEvaluator(cset)
                seq_len = int(round(args.seq_mult * cset.n))
                for run in range(args.runs):
                    es = EvolutionarySearch(inst, cset, pool, evaluator, Reward.shaped(inst, cset, args.relaxed_weight),
                                            EvolutionConfig(seq_len=seq_len, budget=args.budget, mu=args.mu, lam=args.lam, seed=run, log_every=5, record_database=False))
                    marks = {t: None for t in THRESHOLDS}

                    def cb(rec, marks=marks):
                        for t in THRESHOLDS:
                            if marks[t] is None and rec.get("ratio_best", 0) >= t:
                                marks[t] = rec["evaluations"]
                                print(f"  {inst.name} run={run}: ratio {t} at {rec['evaluations']} evaluations ({rec['seconds']:.0f}s)", flush=True)

                    df = es.run(callback=cb)
                    df.insert(0, "run", run); df.insert(0, "instance", inst.name); df.insert(0, "n", cset.n); df.insert(0, "m", inst.m); df.insert(0, "family", family)
                    df.to_csv(curves_path, mode="a", header=not curves_path.exists(), index=False)
                    cut, cut_ls = es.best_with_local_search()
                    sd = shot_decoded(inst, cset, pool.to_circuit(es.best["tokens"]))
                    row = {"family": family, "m": inst.m, "n": cset.n, "instance": inst.name, "run": run, "best_known": bk, "budget": args.budget,
                           "ratio_raw": cut / bk, "ratio_ls": cut_ls / bk, "ratio_1000shots": sd[1000][0] / bk, "ratio_4000shots": sd[4000][0] / bk,
                           "median_abs_corr": float(np.median(np.abs(es.best["corr"][: inst.m]))), **{f"evals_to_{t}": marks[t] for t in THRESHOLDS},
                           "seconds": df.seconds.iloc[-1]}
                    pd.DataFrame([row]).to_csv(final_path, mode="a", header=not final_path.exists(), index=False)
                    print(f"{inst.name:16s} run={run} budget={args.budget} r_raw={cut/bk:.3f} r_ls={cut_ls/bk:.3f} 1000sh={sd[1000][0]/bk:.3f} med|c|={row['median_abs_corr']:.3f} marks={marks} {df.seconds.iloc[-1]:.0f}s", flush=True)


if __name__ == "__main__":
    main()
