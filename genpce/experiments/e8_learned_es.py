"""E8 — Stage 2: does a learned mutation policy accelerate the evolutionary search?

Same (μ+λ) elitist engine, same quantum budget, same single-token edit operator; only the
*proposal* of edits differs:

* ``es``            — all offspring are uniformly random single-token edits;
* ``es+learned``    — ``--learned-fraction`` of the offspring are edits sampled from a
                      :class:`MutationScorer` trained **online** from every evaluated child
                      (parent, edit, ΔR) — starts untrained ("cold");
* ``es+pretrained`` — the scorer is first trained on the complete-neighbourhood dataset of the
                      *same* instance (E7) and keeps learning online (upper-bound / warm variant);
* ``es+transfer``   — scorer pretrained on the neighbourhood dataset of a *different* instance.

Metrics: best-so-far curve vs circuit executions, evaluations to reach 0.8/0.85/0.9, final raw
ratio, shot-decoded ratio. Several runs per instance; medians reported.

Example::

    python -u experiments/e8_learned_es.py --m 60 --seeds 0 1 --runs 3 --budget 20000
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
from genpce.train import EvolutionConfig, EvolutionarySearch, LearnedMutationProposal, MutationDataset, MutationScorer, Reward, ScorerConfig, train_scorer  # noqa: E402

THRESHOLDS = (0.75, 0.8, 0.85, 0.9)


def pretrained_scorer(data_path: Path, vocab: int, seq_len: int, seed: int) -> MutationScorer:
    data = MutationDataset.load(str(data_path))
    cfg = ScorerConfig(loss="mse", seed=seed)
    model = MutationScorer(vocab, seq_len, cfg)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.P)
    n_val = max(8, data.P // 5)
    train_scorer(model, data, cfg, train_idx=perm[n_val:], val_idx=perm[:n_val])
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--families", nargs="+", default=["reg3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--relaxed-weight", type=float, default=0.5)
    ap.add_argument("--learned-fraction", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--update-every", type=int, default=3, help="generations between online scorer updates")
    ap.add_argument("--steps-per-update", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=["es", "es+learned", "es+pretrained", "es+transfer"])
    ap.add_argument("--e7-dir", type=Path, default=RESULTS / "e7")
    ap.add_argument("--e7-tag", default="m60")
    ap.add_argument("--transfer-seed", type=int, default=1, help="instance whose E7 dataset pretrains the transfer scorer")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e8")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_path, final_path = args.out_dir / f"curves_{args.tag}.csv", args.out_dir / f"final_{args.tag}.csv"
    done = set()
    if final_path.exists():
        prev = pd.read_csv(final_path)
        done = set(zip(prev.instance, prev.method, prev.run))

    for family in args.families:
        for seed in args.seeds:
            inst = benchmark_instance(family, args.m, seed)
            cset = cubic_cset(inst.m)
            bk = inst.best_known()
            pool = native_chain_pool(cset.n)
            evaluator = StatevectorEvaluator(cset)
            seq_len = int(round(args.seq_mult * cset.n))
            reward = Reward.shaped(inst, cset, args.relaxed_weight)
            for method in args.methods:
                for run in range(args.runs):
                    if (inst.name, method, run) in done:
                        continue
                    prop = None
                    frac = 0.0
                    if method != "es":
                        frac = args.learned_fraction
                        if method == "es+learned":
                            model = MutationScorer(pool.size, seq_len, ScorerConfig(seed=run))
                        elif method == "es+pretrained":
                            model = pretrained_scorer(args.e7_dir / f"mutations_{inst.name}_{args.e7_tag}.npz", pool.size, seq_len, run)
                        elif method == "es+transfer":
                            other = benchmark_instance(family, args.m, args.transfer_seed if seed != args.transfer_seed else args.seeds[0])
                            model = pretrained_scorer(args.e7_dir / f"mutations_{other.name}_{args.e7_tag}.npz", pool.size, seq_len, run)
                        else:
                            raise ValueError(method)
                        prop = LearnedMutationProposal(model, temperature=args.temperature, epsilon=args.epsilon, steps_per_update=args.steps_per_update, seed=run)
                    cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, seed=100 * run + seed, learned_fraction=frac, single_token_mutations=True, log_every=5, record_database=False, learned_update_every=args.update_every)
                    es = EvolutionarySearch(inst, cset, pool, evaluator, reward, cfg, mutation_proposal=prop)
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
                    row = {"family": family, "m": inst.m, "n": cset.n, "instance": inst.name, "method": method, "run": run, "best_known": bk,
                           "ratio_raw": cut / bk, "ratio_ls": cut_ls / bk, "ratio_1000shots": sd[1000][0] / bk,
                           "median_abs_corr": float(np.median(np.abs(es.best["corr"][: inst.m]))), **{f"evals_to_{t}": marks[t] for t in THRESHOLDS},
                           "seconds": df.seconds.iloc[-1]}
                    pd.DataFrame([row]).to_csv(final_path, mode="a", header=not final_path.exists(), index=False)
                    print(f"{inst.name:14s} {method:14s} run={run} r_raw={cut/bk:.3f} 1000sh={sd[1000][0]/bk:.3f} marks={marks} {df.seconds.iloc[-1]:.0f}s", flush=True)

    fin = pd.read_csv(final_path)
    summ = fin.groupby(["instance", "method"]).agg(ratio_raw=("ratio_raw", "median"), ratio_max=("ratio_raw", "max"), e80=("evals_to_0.8", "median"), e85=("evals_to_0.85", "median"),
                                                  e90=("evals_to_0.9", "median"), n90=("evals_to_0.9", lambda s: int(s.notna().sum())), n=("run", "size"))
    print(summ.round(3).to_string(), flush=True)
    summ.to_csv(args.out_dir / f"summary_{args.tag}.csv")


if __name__ == "__main__":
    main()
