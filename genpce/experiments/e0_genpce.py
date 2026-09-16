"""E0-B — GenPCE pilot on the benchmark instances, with a random-circuit control.

For every (family, m, instance seed) we train the generator from ``--runs`` model seeds and log the
full learning curve (best raw / post-processed ratio vs. quantum evaluations) plus gate statistics
of the best circuit. The control samples uniformly random token sequences with the same quantum
budget. Databases of (tokens, correlators) are saved for later transfer experiments.

Example::

    python -u experiments/e0_genpce.py --families reg3 er4 --sizes 60 105 168 252 --seeds 0 1 2 \
        --seq-mult 8 --epochs 300 --runs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, benchmark_instance, cubic_cset, write_json  # noqa: E402

from genpce.pool import native_chain_pool  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train import GenPCEConfig, GenPCETrainer, random_search  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["reg3", "er4"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[60, 105, 168, 252])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--runs", type=int, default=2, help="independent model seeds per instance")
    ap.add_argument("--seq-mult", type=float, default=8.0, help="sequence length N = seq_mult · n")
    ap.add_argument("--seq-len", type=int, default=None, help="explicit N (overrides --seq-mult)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-sample", type=int, default=50)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--reward", default="cut", choices=["cut", "relaxed", "shaped"])
    ap.add_argument("--loss", default="grpo", choices=["grpo", "logit_matching", "grpo+lm"])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e0")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    curves_path = args.out_dir / f"genpce_curves_{args.tag}.csv"
    summary_path = args.out_dir / f"genpce_summary_{args.tag}.csv"
    random_path = args.out_dir / f"random_curves_{args.tag}.csv"
    db_dir = args.out_dir / "databases"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        done = set(zip(prev.instance, prev.run))

    for family in args.families:
        for m in args.sizes:
            for seed in args.seeds:
                inst = benchmark_instance(family, m, seed)
                cset = cubic_cset(inst.m)
                bk = inst.best_known()
                pool = native_chain_pool(cset.n)
                evaluator = StatevectorEvaluator(cset)
                seq_len = args.seq_len or int(round(args.seq_mult * cset.n))
                budget = args.epochs * args.n_sample

                if not args.no_random and (inst.name, -1) not in done:
                    rs = random_search(inst, cset, pool, evaluator, seq_len=seq_len, total=budget, seed=seed)
                    rs.insert(0, "instance", inst.name)
                    rs.insert(0, "m", inst.m)
                    rs.insert(0, "family", family)
                    rs.to_csv(random_path, mode="a", header=not random_path.exists(), index=False)
                    last = rs.iloc[-1]
                    pd.DataFrame([{
                        "family": family, "m": inst.m, "n": cset.n, "instance": inst.name, "run": -1,
                        "method": "random", "seq_len": seq_len, "evaluations": int(last.evaluations),
                        "ratio_raw": last.ratio_best, "ratio_ls": last.ratio_best_ls, "best_epoch": -1,
                        "two_qubit": None, "non_clifford": None, "seconds": None,
                    }]).to_csv(summary_path, mode="a", header=not summary_path.exists(), index=False)
                    print(f"{inst.name:16s} RANDOM  N={seq_len:3d} evals={budget:6d} r_raw={last.ratio_best:.3f} r_ls={last.ratio_best_ls:.3f}", flush=True)

                for run in range(args.runs):
                    if (inst.name, run) in done:
                        continue
                    cfg = GenPCEConfig(
                        seq_len=seq_len, epochs=args.epochs, n_sample=args.n_sample, n_iter=args.n_iter,
                        lr=args.lr, reward=args.reward, loss=args.loss, d_model=args.d_model,
                        n_layers=args.n_layers, n_heads=4, seed=100 * run + seed,
                    )
                    trainer = GenPCETrainer(inst, cset, pool, evaluator, cfg)
                    df = trainer.run()
                    df.insert(0, "run", run)
                    df.insert(0, "instance", inst.name)
                    df.insert(0, "m", inst.m)
                    df.insert(0, "family", family)
                    df.to_csv(curves_path, mode="a", header=not curves_path.exists(), index=False)
                    counts = pool.gate_counts(trainer.best["tokens"])
                    pd.DataFrame([{
                        "family": family, "m": inst.m, "n": cset.n, "instance": inst.name, "run": run,
                        "method": "genpce", "seq_len": seq_len, "evaluations": trainer.evaluations,
                        "ratio_raw": trainer.best["cut"] / bk, "ratio_ls": trainer.best["cut_ls"] / bk,
                        "best_epoch": trainer.best["epoch"], "two_qubit": counts["two_qubit"],
                        "non_clifford": counts["non_clifford"], "seconds": df.seconds.iloc[-1],
                    }]).to_csv(summary_path, mode="a", header=not summary_path.exists(), index=False)
                    db_dir.mkdir(parents=True, exist_ok=True)
                    trainer.database.save(str(db_dir / f"{inst.name}_run{run}_{args.tag}.npz"))
                    write_json(db_dir / f"{inst.name}_run{run}_{args.tag}_best.json", {
                        "tokens": trainer.best["tokens"].tolist(), "cut": trainer.best["cut"],
                        "cut_ls": trainer.best["cut_ls"], "config": cfg.to_dict(),
                    })
                    print(
                        f"{inst.name:16s} GENPCE run={run} N={seq_len:3d} evals={trainer.evaluations:6d} "
                        f"r_raw={trainer.best['cut']/bk:.3f} r_ls={trainer.best['cut_ls']/bk:.3f} "
                        f"(epoch {trainer.best['epoch']}) 2q={counts['two_qubit']} nC={counts['non_clifford']} "
                        f"{df.seconds.iloc[-1]:5.0f}s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
