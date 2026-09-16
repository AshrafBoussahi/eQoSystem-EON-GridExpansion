"""E4 — amortisation across instances: does a transformer prior learned from other instances help?

Protocol (all instances share n, k, m and the token space; the optimum is never used in training):

1. **Sources.** Run GenPCE-ES (shaped reward) on ``--n-sources`` instances per family; keep the top
   ``--elites-per-source`` sequences of each (ranked by their own measured reward) and the full
   evaluated databases.
2. **Prior.** Train a GPT by maximum likelihood on the union of elites with a held-out split
   (memorisation check: validation NLL/token vs. training NLL vs. uniform ``ln|vocab|``).
3. **Held-out targets.** For each of ``--n-targets`` unseen instances per family, at equal budget:
   * ``cold``        — ES from random sequences;
   * ``prior_init``  — sample ``--init-samples`` sequences from the prior, evaluate them (counted),
                       seed the population with the best ``mu``, continue with mutations only;
   * ``prior_prop``  — cold start, but ``--model-fraction`` of every generation's offspring are
                       sampled from the fixed prior;
   * ``db_rescore``  — re-score all source databases under the target at zero quantum cost and seed
                       half of the population with the best re-scored circuits (rest random).
   Also recorded: mean/max quality and correlator magnitude of 500 prior samples vs 500 random
   sequences on each target (the "does the prior write better PCE circuits for unseen instances"
   question), and evaluations-to-threshold for every method.

Example::

    python -u experiments/e4_transfer.py --m 60 --n-sources 6 --n-targets 3 --budget 20000
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

from genpce.pce import decode_signs  # noqa: E402
from genpce.pool import native_chain_pool  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train import CircuitDatabase, EvolutionConfig, EvolutionarySearch, ModelProposal, PriorConfig, Reward, mean_token_entropy, train_prior  # noqa: E402

THRESHOLDS = (0.8, 0.85, 0.9)


def hits(df: pd.DataFrame) -> dict:
    out = {}
    for t in THRESHOLDS:
        ok = df[df.ratio_best >= t]
        out[f"hit_{t}"] = int(ok.evaluations.iloc[0]) if len(ok) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--families", nargs="+", default=["reg3", "er4"])
    ap.add_argument("--n-sources", type=int, default=6)
    ap.add_argument("--n-targets", type=int, default=3)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--elites-per-source", type=int, default=200)
    ap.add_argument("--init-samples", type=int, default=200)
    ap.add_argument("--model-fraction", type=float, default=0.3)
    ap.add_argument("--prior-beta", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e4")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # shared geometry (all instances have the same m -> same n, k, pool)
    probe = benchmark_instance(args.families[0], args.m, 0)
    cset = cubic_cset(probe.m)
    pool = native_chain_pool(cset.n)
    evaluator = StatevectorEvaluator(cset)
    seq_len = int(round(args.seq_mult * cset.n))
    mu = 10

    # ---------------------------------------------------------------- 1. sources
    elites, source_rows, databases = [], [], []
    for family in args.families:
        for seed in range(args.n_sources):
            inst = benchmark_instance(family, args.m, seed)
            es = EvolutionarySearch(inst, cset, pool, evaluator, Reward.shaped(inst, cset), EvolutionConfig(seq_len=seq_len, budget=args.budget, mu=mu, seed=seed))
            df = es.run()
            e = np.unique(es.elites(3 * args.elites_per_source), axis=0)[: args.elites_per_source]
            elites.append(e)
            databases.append(es.database)
            source_rows.append({"instance": inst.name, "ratio_best": df.ratio_best.iloc[-1], "median_abs_corr": df.median_abs_corr.iloc[-1], **hits(df)})
            print(f"SOURCE {inst.name:16s} best={df.ratio_best.iloc[-1]:.3f} med|c|={df.median_abs_corr.iloc[-1]:.3f} hits={hits(df)}", flush=True)
    pd.DataFrame(source_rows).to_csv(args.out_dir / f"sources_{args.tag}.csv", index=False)
    val_elites = elites[-1]  # the last source is held out entirely for the memorisation check
    elites = np.concatenate(elites[:-1])
    np.save(args.out_dir / f"elites_{args.tag}.npy", elites)

    # ---------------------------------------------------------------- 2. prior (validated on an unseen source)
    model, info = train_prior(elites, pool.size, PriorConfig(seed=0), val_elites=val_elites)
    ent = mean_token_entropy(model, elites[:200])
    print(f"PRIOR trained on {len(elites)} elites: val NLL/token={info['best_val_nll_per_token']:.3f} (uniform {info['uniform_nll_per_token']:.3f}), "
          f"train NLL/token={info['history'][-1]['train_nll']:.3f}, steps={info['steps']}, mean token entropy={ent:.2f} nats", flush=True)
    torch.save(model.state_dict(), args.out_dir / f"prior_{args.tag}.pt")
    pd.DataFrame(info["history"]).to_csv(args.out_dir / f"prior_history_{args.tag}.csv", index=False)

    # merged source database for zero-cost re-scoring
    merged = CircuitDatabase()
    for db in databases:
        merged.tokens.extend(db.tokens)
        merged.corr.extend(db.corr)
        merged.epoch.extend(db.epoch)
    merged_corr = np.stack(merged.corr).astype(np.float64)
    merged_tokens = np.stack(merged.tokens)

    # ---------------------------------------------------------------- 3. targets
    rows, curves = [], []
    rng = np.random.default_rng(123)
    for family in args.families:
        for seed in range(args.n_sources, args.n_sources + args.n_targets):
            tgt = benchmark_instance(family, args.m, seed)
            bk = tgt.best_known()
            reward = Reward.shaped(tgt, cset)

            # prior samples vs random (no learning on the target)
            prior_toks = ModelProposal(model, seq_len, args.prior_beta, seed=7)(500)
            rand_toks = rng.integers(0, pool.size, size=(500, seq_len))
            pc = evaluator.evaluate(pool.to_circuits(prior_toks))
            rc = evaluator.evaluate(pool.to_circuits(rand_toks))
            _, pcut = reward(pc)
            _, rcut = reward(rc)
            sample_stats = {
                "prior_mean": float(pcut.mean() / bk), "prior_max": float(pcut.max() / bk), "prior_median_abs_corr": float(np.median(np.abs(pc))),
                "random_mean": float(rcut.mean() / bk), "random_max": float(rcut.max() / bk), "random_median_abs_corr": float(np.median(np.abs(rc))),
            }
            print(f"TARGET {tgt.name:16s} prior samples: mean={sample_stats['prior_mean']:.3f} max={sample_stats['prior_max']:.3f} med|c|={sample_stats['prior_median_abs_corr']:.3f} | "
                  f"random: mean={sample_stats['random_mean']:.3f} max={sample_stats['random_max']:.3f} med|c|={sample_stats['random_median_abs_corr']:.3f}", flush=True)

            # zero-cost re-scoring of all source data
            rescored = tgt.cut_values(decode_signs(merged_corr[:, : tgt.m]))
            top = np.argsort(rescored)[::-1]
            rescore_best = float(rescored[top[0]] / bk)

            for run in range(args.runs):
                for method in ("cold", "prior_init", "prior_prop", "db_rescore"):
                    cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, mu=mu, seed=100 * run + seed,
                                          model_fraction=args.model_fraction if method == "prior_prop" else 0.0, model_beta=args.prior_beta)
                    init, proposal, offset = None, None, 0
                    if method == "prior_init":
                        cand = ModelProposal(model, seq_len, args.prior_beta, seed=run)(args.init_samples)
                        s, _ = reward(evaluator.evaluate(pool.to_circuits(cand)))
                        init = cand[np.argsort(s)[::-1][:mu]]
                        offset = args.init_samples  # these evaluations are charged to the method
                        cfg.budget = args.budget - offset
                    elif method == "prior_prop":
                        proposal = ModelProposal(model, seq_len, args.prior_beta, seed=run)
                    elif method == "db_rescore":
                        init = np.concatenate([merged_tokens[top[: mu // 2]], rng.integers(0, pool.size, size=(mu - mu // 2, seq_len))])
                    es = EvolutionarySearch(tgt, cset, pool, evaluator, reward, cfg, proposal=proposal, init_population=init)
                    df = es.run()
                    df["evaluations"] = df["evaluations"] + offset
                    for r in df.itertuples():
                        curves.append({"target": tgt.name, "family": family, "method": method, "run": run, "evaluations": r.evaluations, "ratio": r.ratio_best})
                    cut, cut_ls = es.best_with_local_search()
                    rec = {"target": tgt.name, "family": family, "method": method, "run": run, "ratio_raw": cut / bk, "ratio_ls": cut_ls / bk,
                           "median_abs_corr": float(np.median(np.abs(es.best["corr"][: tgt.m]))), "rescore_best": rescore_best, **hits(df), **sample_stats}
                    rows.append(rec)
                    print(f"  {tgt.name:16s} {method:11s} run={run} final={cut/bk:.3f} (+LS {cut_ls/bk:.3f}) hits={hits(df)}", flush=True)
            pd.DataFrame(rows).to_csv(args.out_dir / f"targets_{args.tag}.csv", index=False)
            pd.DataFrame(curves).to_csv(args.out_dir / f"curves_{args.tag}.csv", index=False)

    summary = pd.DataFrame(rows).groupby("method").agg(
        ratio_raw=("ratio_raw", "mean"), ratio_ls=("ratio_ls", "mean"),
        hit_0_8=("hit_0.8", "median"), hit_0_85=("hit_0.85", "median"), hit_0_9=("hit_0.9", "median"),
        n_hit_0_9=("hit_0.9", lambda s: int(s.notna().sum())), n=("ratio_raw", "size"),
    )
    print(summary.round(3).to_string(), flush=True)
    summary.to_csv(args.out_dir / f"summary_{args.tag}.csv")


if __name__ == "__main__":
    main()
