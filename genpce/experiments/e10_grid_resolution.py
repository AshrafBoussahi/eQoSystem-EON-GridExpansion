"""E10 supplement — which angle grid would let a discrete circuit keep the refined correlations?

For every searched-best circuit (from the ceiling stage) the angles are refined continuously with the
axes fixed, then snapped to full-circle grids of spacing 2π/M for several M, and re-simulated. The
current pool is {0, ±π/16, ±π/8, ±π/4, ±π/2} per axis (no angle beyond π/2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, benchmark_instance, cubic_cset  # noqa: E402
from e10_phase_a import load_pairs  # noqa: E402

from genpce.pool import native_chain_pool  # noqa: E402
from genpce.train.distill import SingleAxisBrickworkCZ, VQACorpus, reconstruction_metrics, refine_and_snap  # noqa: E402

OUT = RESULTS / "e10"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="reg3:0")
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--M", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    torch.set_num_threads(1)
    fam, iseed = args.instance.split(":")
    inst = benchmark_instance(fam, 60, int(iseed))
    cset = cubic_cset(inst.m)
    pool = native_chain_pool(cset.n)
    corpus = VQACorpus.load(str(OUT / f"corpus_{inst.name}.npz"))
    targets = pd.read_csv(OUT / f"targets_{inst.name}.csv")
    rows = []
    for L in args.depths:
        tok, corr, tid, _ = load_pairs(inst, L)
        for t in targets.itertuples():
            c_star = corpus.corr[t.corpus_index].astype(np.float64)
            sel = np.flatnonzero(tid == t.target)
            d = np.mean((corr[sel][:, : inst.m].astype(np.float64) - c_star[: inst.m]) ** 2, axis=1)
            out = refine_and_snap(pool, cset, tok[sel[int(np.argmin(d))]], c_star, steps=args.steps)
            sim = SingleAxisBrickworkCZ(cset.n, L, cset, out["axes"])
            row = {"target": t.target, "holdout": t.holdout, "teacher_layers": t.teacher_layers, "discrete_layers": L, "target_ratio": t.target_cut / inst.best_known(),
                   "D_c_search": float(d.min()), "D_c_refined": out["D_c_refined"], "D_c_pool_grid": float(np.mean((pool_eval(pool, cset, out["snapped_tokens"]) - c_star[: cset.m]) ** 2))}
            for M in args.M:
                th = np.round(out["angles_refined"] / (2 * np.pi / M)) * (2 * np.pi / M)
                with torch.no_grad():
                    c = sim.correlators(torch.tensor(th)).numpy()
                met = reconstruction_metrics(c, c_star, inst)
                row[f"D_c_M{M}"], row[f"A_x_M{M}"], row[f"ratio_M{M}"] = met["D_c"], met["A_x"], met["ratio"]
            rows.append(row)
            print(f"L={L} {t.target:16s} search={row['D_c_search']:.4f} refined={row['D_c_refined']:.4f} pool-grid={row['D_c_pool_grid']:.4f} | "
                  + " ".join(f"M{M}={row[f'D_c_M{M}']:.4f}" for M in args.M), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"grid_resolution_{inst.name}.csv", index=False)
    cols = ["D_c_search", "D_c_refined", "D_c_pool_grid"] + [f"D_c_M{M}" for M in args.M]
    print("\nmean over targets:")
    print(df.groupby("discrete_layers")[cols + [f"A_x_M{M}" for M in args.M] + [f"ratio_M{M}" for M in args.M] + ["target_ratio"]].mean().round(4).T.to_string())


def pool_eval(pool, cset, tokens):
    from genpce.sim import StatevectorEvaluator
    return StatevectorEvaluator(cset).evaluate([pool.to_circuit(tokens)])[0]


if __name__ == "__main__":
    main()
