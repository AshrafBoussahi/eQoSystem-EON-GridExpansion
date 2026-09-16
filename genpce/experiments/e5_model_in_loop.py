"""E5 — does a learned proposal distribution help the per-instance search?

Three rungs at equal quantum budget on the same instances:

* ``es``          — mutation-only (μ+λ) search;
* ``es+pbil``     — half of the offspring from an independent-slot distribution (PBIL) updated
                    online from the elite archive;
* ``es+gpt``      — half of the offspring from an autoregressive transformer fine-tuned online on
                    the elite archive (dropout, weight decay, few steps per generation).

Example::

    python -u experiments/e5_model_in_loop.py --sizes 60 252 --budget 20000 --runs 2
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
from genpce.train import EvolutionConfig, EvolutionarySearch, PBILProposal, Reward, TransformerProposal  # noqa: E402

THRESHOLDS = (0.75, 0.8, 0.85, 0.9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["reg3", "er4"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[60, 252])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--model-fraction", type=float, default=0.5)
    ap.add_argument("--methods", nargs="+", default=["es", "es+pbil", "es+gpt"])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e5")
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
                for method in args.methods:
                    for run in range(args.runs):
                        frac = 0.0 if method == "es" else args.model_fraction
                        prop = None
                        if method == "es+pbil":
                            prop = PBILProposal(seq_len, pool.size, seed=run)
                        elif method == "es+gpt":
                            prop = TransformerProposal(seq_len, pool.size, seed=run)
                        cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, model_fraction=frac, seed=run, log_every=5)
                        es = EvolutionarySearch(inst, cset, pool, evaluator, Reward.shaped(inst, cset), cfg, proposal=prop)
                        marks = {t: None for t in THRESHOLDS}

                        def cb(rec, marks=marks):
                            for t in THRESHOLDS:
                                if marks[t] is None and rec.get("ratio_best", 0) >= t:
                                    marks[t] = rec["evaluations"]

                        df = es.run(callback=cb)
                        df.insert(0, "run", run); df.insert(0, "method", method); df.insert(0, "instance", inst.name); df.insert(0, "m", inst.m); df.insert(0, "family", family)
                        df.to_csv(curves_path, mode="a", header=not curves_path.exists(), index=False)
                        cut, cut_ls = es.best_with_local_search()
                        sd = shot_decoded(inst, cset, pool.to_circuit(es.best["tokens"]))
                        extra = {}
                        if method == "es+pbil":
                            extra["proposal_entropy"] = prop.entropy()
                        elif method == "es+gpt":
                            extra["proposal_last_nll"] = prop.last_loss
                        row = {"family": family, "m": inst.m, "n": cset.n, "instance": inst.name, "method": method, "run": run, "best_known": bk,
                               "ratio_raw": cut / bk, "ratio_ls": cut_ls / bk, "ratio_1000shots": sd[1000][0] / bk, "ratio_4000shots": sd[4000][0] / bk,
                               "median_abs_corr": float(np.median(np.abs(es.best["corr"][: inst.m]))), **{f"evals_to_{t}": marks[t] for t in THRESHOLDS},
                               "seconds": df.seconds.iloc[-1], **extra}
                        pd.DataFrame([row]).to_csv(final_path, mode="a", header=not final_path.exists(), index=False)
                        print(f"{inst.name:16s} {method:8s} run={run} r_raw={cut/bk:.3f} 1000sh={sd[1000][0]/bk:.3f} med|c|={row['median_abs_corr']:.3f} marks={marks} {extra} {df.seconds.iloc[-1]:.0f}s", flush=True)


if __name__ == "__main__":
    main()
