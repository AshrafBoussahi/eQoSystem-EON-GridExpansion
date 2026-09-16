"""E10 supplement — learning curve of the forward map U -> c on *random* circuits.

The pair sets from the search are clustered around a few basins; here the training data are uniform
random token sequences, and the held-out set is also random, so this is the cleanest question:
"with N examples, how well can a transformer regress all m correlators of an unseen circuit?"
Baselines: predicting the training mean; a linear (ridge) model on one-hot tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, benchmark_instance, cubic_cset  # noqa: E402

from genpce.pce.correlators import decode_signs  # noqa: E402
from genpce.pool import native_chain_pool  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train.distill import CorrelatorPredictor, train_forward  # noqa: E402

OUT = RESULTS / "e10"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="reg3:0")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000, 80000])
    ap.add_argument("--test", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    fam, iseed = args.instance.split(":")
    inst = benchmark_instance(fam, 60, int(iseed))
    cset = cubic_cset(inst.m)
    pool = native_chain_pool(cset.n)
    ev = StatevectorEvaluator(cset)
    seq_len = args.L * cset.n
    rng = np.random.default_rng(0)
    cache = OUT / f"random_pairs_{inst.name}_L{args.L}.npz"
    N = max(args.sizes) + args.test
    if cache.exists() and len(np.load(cache)["tokens"]) >= N:
        z = np.load(cache); tok, corr = z["tokens"][:N], z["corr"][:N]
    else:
        tok = rng.integers(0, pool.size, size=(N, seq_len))
        t0 = perf_counter()
        corr = np.concatenate([ev.evaluate(pool.to_circuits(tok[i : i + 2000])).astype(np.float32) for i in range(0, N, 2000)])
        np.savez_compressed(cache, tokens=tok, corr=corr)
        print(f"evaluated {N} random circuits in {perf_counter()-t0:.0f}s", flush=True)
    test = np.arange(N - args.test, N)
    Yt = corr[test][:, : inst.m].astype(np.float64)
    xt = decode_signs(Yt)
    rows = []
    for n_train in args.sizes:
        tr = np.arange(n_train)
        val, tr = tr[: max(500, n_train // 10)], tr[max(500, n_train // 10):]
        # ridge on one-hot tokens (linear baseline)
        X1 = np.zeros((N, seq_len * pool.size), dtype=np.float32)
        X1[np.arange(N)[:, None], np.arange(seq_len)[None, :] * pool.size + tok] = 1.0
        A = X1[tr]; lam = 1.0
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ corr[tr])
        p_lin = X1[test] @ W
        mse_lin = float(np.mean((p_lin[:, : inst.m] - Yt) ** 2))
        base = float(np.mean((corr[tr][:, : inst.m].mean(0) - Yt) ** 2))
        model = CorrelatorPredictor(pool.size, seq_len, cset.m, d_model=args.d_model, n_layers=args.n_layers)
        t0 = perf_counter()
        info = train_forward(model, tok, corr, train_idx=tr, val_idx=val, max_steps=args.steps, patience=1000, seed=0)
        with torch.no_grad():
            p = np.concatenate([model(torch.as_tensor(tok[test[i : i + 1024]], dtype=torch.long)).numpy() for i in range(0, len(test), 1024)])[:, : inst.m]
        from scipy.stats import spearmanr
        sp = float(np.mean([spearmanr(p[i], Yt[i]).correlation for i in range(len(test))]))
        row = {"L": args.L, "n_train": len(tr), "mse_mean_baseline": base, "mse_ridge": mse_lin, "mse_transformer": float(np.mean((p - Yt) ** 2)),
               "sign_acc_transformer": float(np.mean(decode_signs(p) == xt)), "sign_acc_ridge": float(np.mean(decode_signs(p_lin[:, : inst.m]) == xt)),
               "spearman_transformer": sp, "best_val_mse": info["best_val_mse"], "steps": info["steps"], "seconds": perf_counter() - t0}
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"forward_curve_{inst.name}_L{args.L}.csv", index=False)


if __name__ == "__main__":
    main()
