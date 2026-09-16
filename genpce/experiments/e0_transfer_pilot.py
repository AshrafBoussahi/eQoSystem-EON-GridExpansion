"""E0-C — transfer pilot: is the quantum data instance-agnostic in practice?

1. Run an evolutionary search (μ+λ, token mutations) on a *source* instance while recording every
   evaluated circuit's correlator vector in a :class:`CircuitDatabase`.
2. For each *target* instance (same n, k, m):
   (a) re-score the database under the target objective at zero quantum cost — best re-scored ratio;
   (b) warm-start the search on the target from the top-μ re-scored circuits vs a cold start, at the
       same budget — evaluations to reach a ratio threshold;
   (c) train the transformer by elite imitation on the source database's top fraction and sample
       from it: mean ratio of its samples on the target vs uniformly random sequences (does the
       model learn instance-independent "correlation-rich" structure?).

Example::

    python -u experiments/e0_transfer_pilot.py --source reg3:60:0 --targets reg3:60:1 reg3:60:2 er4:60:0
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

from genpce.model import GPT, GPTConfig, sample_sequences, sequence_log_probs  # noqa: E402
from genpce.pce import decode_signs  # noqa: E402
from genpce.pool import native_chain_pool  # noqa: E402
from genpce.problems import one_pass_bit_swap  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train import CircuitDatabase  # noqa: E402


def parse(spec: str):
    fam, m, seed = spec.split(":")
    return fam, int(m), int(seed)


def evolve(inst, cset, pool, evaluator, *, seq_len, budget, mu=10, lam=50, seed=0, init=None, db=None, thresholds=(0.8, 0.85, 0.9)):
    """(μ+λ)-ES with 1–2 token mutations; returns (best_tokens, best_corr, curve rows, hits)."""
    rng = np.random.default_rng(seed)
    bk = inst.best_known()  # used ONLY to express thresholds/curves as approximation ratios

    def fit(seqs):
        corr = evaluator.evaluate(pool.to_circuits(seqs))
        if db is not None:
            db.push(seqs, corr, 0)
        # Selection is by raw cut value; dividing by the constant bk only rescales for reporting
        # and cannot change any ranking (the search never sees the optimum).
        return inst.cut_values(decode_signs(corr[:, : inst.m])) / bk, corr

    pop = rng.integers(0, pool.size, size=(mu, seq_len)) if init is None else np.asarray(init)[:mu]
    f, c = fit(pop)
    evals = len(pop)
    hits = {t: None for t in thresholds}
    rows = []
    while evals < budget:
        kids = pop[rng.integers(0, mu, size=lam)].copy()
        k = rng.integers(1, 3, size=lam)
        for i in range(lam):
            pos = rng.choice(seq_len, size=k[i], replace=False)
            kids[i, pos] = rng.integers(0, pool.size, size=k[i])
        fk, ck = fit(kids)
        evals += lam
        allp, allf, allc = np.concatenate([pop, kids]), np.concatenate([f, fk]), np.concatenate([c, ck])
        order = np.argsort(allf)[::-1][:mu]
        pop, f, c = allp[order], allf[order], allc[order]
        for t in thresholds:
            if hits[t] is None and f[0] >= t:
                hits[t] = evals
        rows.append({"evaluations": evals, "best": float(f[0]), "mu_mean": float(f.mean())})
    return pop[0], c[0], rows, hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="reg3:60:0")
    ap.add_argument("--targets", nargs="+", default=["reg3:60:1", "reg3:60:2", "er4:60:0"])
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--model-samples", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e0")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fam, m, seed = parse(args.source)
    src = benchmark_instance(fam, m, seed)
    cset = cubic_cset(src.m)
    pool = native_chain_pool(cset.n)
    evaluator = StatevectorEvaluator(cset)
    seq_len = int(round(args.seq_mult * cset.n))

    # 1. source search with database recording
    db = CircuitDatabase()
    best_tok, best_c, rows, hits = evolve(src, cset, pool, evaluator, seq_len=seq_len, budget=args.budget, seed=0, db=db)
    src_best = src.cut_value(decode_signs(best_c[: src.m])) / src.best_known()
    print(f"SOURCE {src.name}: ES best raw={src_best:.3f} | database size {len(db)} | hits {hits}", flush=True)
    db.save(str(args.out_dir / f"transfer_db_{src.name}.npz"))

    # (c) model trained offline on the source elites
    scores_src = db.rescore(src) / src.best_known()
    elite_idx = np.argsort(scores_src)[::-1][: max(50, len(db) // 20)]
    elite_tokens = torch.as_tensor(np.stack([db.tokens[i] for i in elite_idx]))
    model = GPT(GPTConfig(vocab_size=pool.size, max_len=seq_len, d_model=128, n_layers=4, n_heads=4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    torch.manual_seed(0)
    for step in range(300):
        idx = torch.randint(0, len(elite_tokens), (min(64, len(elite_tokens)),))
        logp, _ = sequence_log_probs(model, elite_tokens[idx], beta=1.0)
        loss = -logp.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  elite-imitation model trained on {len(elite_tokens)} source elites (final NLL/token {float(loss)/seq_len:.3f})", flush=True)

    records = []
    rng = np.random.default_rng(1)
    for spec in args.targets:
        tfam, tm, tseed = parse(spec)
        tgt = benchmark_instance(tfam, tm, tseed)
        bk = tgt.best_known()
        # (a) zero-cost re-scoring
        rescored = db.rescore(tgt) / bk
        top = np.argsort(rescored)[::-1]
        best_rescored = float(rescored[top[0]])
        best_rescored_ls = tgt.cut_value(one_pass_bit_swap(tgt, decode_signs(np.asarray(db.corr[top[0]], dtype=np.float64)[: tgt.m]))) / bk
        # (b) warm vs cold search
        init = np.stack([db.tokens[i] for i in top[:10]])
        _, _, rows_w, hits_w = evolve(tgt, cset, pool, evaluator, seq_len=seq_len, budget=args.budget, seed=1, init=init)
        _, _, rows_c, hits_c = evolve(tgt, cset, pool, evaluator, seq_len=seq_len, budget=args.budget, seed=1)
        # (c) model samples vs random
        for beta in (1.0, 2.0):
            toks, _ = sample_sequences(model, args.model_samples, seq_len, beta, generator=torch.Generator().manual_seed(0))
            corr = evaluator.evaluate(pool.to_circuits(toks.numpy()))
            r_model = tgt.cut_values(decode_signs(corr[:, : tgt.m])) / bk
            rand = rng.integers(0, pool.size, size=(args.model_samples, seq_len))
            corr_r = evaluator.evaluate(pool.to_circuits(rand))
            r_rand = tgt.cut_values(decode_signs(corr_r[:, : tgt.m])) / bk
            rec = {
                "source": src.name, "target": tgt.name, "beta": beta,
                "best_rescored": best_rescored, "best_rescored_ls": best_rescored_ls,
                "warm_final": rows_w[-1]["best"], "cold_final": rows_c[-1]["best"],
                **{f"warm_hit_{t}": hits_w[t] for t in hits_w}, **{f"cold_hit_{t}": hits_c[t] for t in hits_c},
                "model_mean": float(r_model.mean()), "model_max": float(r_model.max()),
                "model_median_abs_corr": float(np.median(np.abs(corr))),
                "random_mean": float(r_rand.mean()), "random_max": float(r_rand.max()),
                "random_median_abs_corr": float(np.median(np.abs(corr_r))),
            }
            records.append(rec)
            print(
                f"TARGET {tgt.name} beta={beta}: rescored best={best_rescored:.3f} (+LS {best_rescored_ls:.3f}) | "
                f"warm final={rec['warm_final']:.3f} hits={hits_w} | cold final={rec['cold_final']:.3f} hits={hits_c} | "
                f"model samples mean={rec['model_mean']:.3f} max={rec['model_max']:.3f} med|c|={rec['model_median_abs_corr']:.3f} | "
                f"random mean={rec['random_mean']:.3f} max={rec['random_max']:.3f} med|c|={rec['random_median_abs_corr']:.3f}",
                flush=True,
            )
    pd.DataFrame(records).to_csv(args.out_dir / f"transfer_pilot_{src.name}.csv", index=False)


if __name__ == "__main__":
    main()
