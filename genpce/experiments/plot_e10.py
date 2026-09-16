"""Summarise and draw E10 (Phase A): teacher corpus, representability ceiling, forward and inverse tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

DEPTH_COLOR = {4: "#ff7f0e", 6: "#2ca02c", 8: "#1f77b4"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="reg3_m60_s0")
    ap.add_argument("--dir", type=Path, default=RESULTS / "e10")
    args = ap.parse_args()
    d, name = args.dir, args.instance

    ceil = pd.concat([pd.read_csv(p) for p in sorted(d.glob(f"ceiling_{name}_L*.csv"))], ignore_index=True)
    print("=== representability ceiling (search on correlation distance, 20k evaluations per target)")
    g = ceil.groupby(["discrete_layers", "teacher_layers", "stage"]).agg(n=("target", "size"), target_r=("target_ratio", "mean"), D_c=("D_c", "mean"),
                                                                          D_c_random=("random_D_c_mean", "mean"), A_x=("A_x", "mean"), r=("ratio", "mean"),
                                                                          r_1000=("ratio_1000shots", "mean"), abs_c=("median_abs_corr", "mean"), target_abs_c=("target_median_abs_corr", "mean"))
    print(g.round(3).to_string())
    g.round(4).to_csv(d / f"summary_ceiling_{name}.csv")

    fwd = {}
    for p in sorted(d.glob(f"forward_{name}_L*.json")):
        L = int(p.stem.split("_L")[-1])
        fwd[L] = json.loads(p.read_text())
    if fwd:
        print("\n=== forward model U -> c (held-out search pairs / random circuits)")
        rows = []
        for L, r in fwd.items():
            for split in ("test_holdout_search", "test_random", "train_fit"):
                rows.append({"L": L, "split": split, **{k: v for k, v in r[split].items()}})
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    inv = pd.concat([pd.read_csv(p) for p in sorted(d.glob(f"inverse_{name}_L*.csv"))], ignore_index=True) if list(d.glob(f"inverse_{name}_L*.csv")) else None
    if inv is not None:
        print("\n=== inverse model c* -> U on held-out teacher targets (mean over targets)")
        cols = ["D_c", "A_x", "ratio", "median_abs_corr"]
        piv = inv.groupby(["discrete_layers", "method", "K"])[cols + ["target_ratio"]].mean()
        print(piv.round(3).to_string())
        piv.round(4).to_csv(d / f"summary_inverse_{name}.csv")
        # paired: generator vs random at the same K, same target
        print("\n=== paired differences (generator − random), same target and K")
        for L in sorted(inv.discrete_layers.unique()):
            s = inv[inv.discrete_layers == L]
            base = s[s.method == "random"].set_index(["target", "K"])
            for meth in sorted(m for m in s.method.unique() if m.startswith("generator")):
                a = s[s.method == meth].set_index(["target", "K"])
                idx = a.index.intersection(base.index)
                dd = (a.loc[idx, cols] - base.loc[idx, cols]).groupby(level="K").agg(["mean", "std"])
                print(f"L={L} {meth}\n{dd.round(4).to_string()}")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for L, s in ceil.groupby("discrete_layers"):
        ax.scatter(s.random_D_c_mean, s.D_c, color=DEPTH_COLOR.get(L, "k"), label=f"{L} discrete layers", s=28, alpha=0.8)
    lim = [0, ceil.random_D_c_mean.max() * 1.05]
    ax.plot(lim, lim, "k:", lw=1, label="random circuit")
    ax.plot(lim, [x / 10 for x in lim], "k--", lw=0.8, label="10× below random")
    ax.set_xlabel("D_c of a random circuit (≈ mean c*²)"); ax.set_ylabel("best D_c found by search (20k evaluations)")
    ax.set_title("representability ceiling per target"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for L, s in ceil.groupby("discrete_layers"):
        ax.scatter(s.target_ratio, s.ratio, color=DEPTH_COLOR.get(L, "k"), label=f"{L} discrete layers", s=28, alpha=0.8)
    ax.plot([0.6, 1], [0.6, 1], "k:", lw=1)
    ax.set_xlabel("teacher target cut ratio"); ax.set_ylabel("cut ratio of the best-matching discrete circuit")
    ax.set_title("does matching the correlations keep the cut?"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    if inv is not None:
        order = ["random", "nearest_train", "generator_T1.0", "generator_T0.7", "generator+filter_T1.0", "search_ceiling"]
        Ls = sorted(inv.discrete_layers.unique())
        width = 0.8 / len(Ls)
        for j, L in enumerate(Ls):
            s = inv[inv.discrete_layers == L]
            vals = []
            for meth in order:
                ss = s[(s.method == meth) & (s.K == (0 if meth == "search_ceiling" else s.K.max()))]
                vals.append(ss.D_c.mean() if len(ss) else np.nan)
            ax.bar(np.arange(len(order)) + j * width, vals, width, color=DEPTH_COLOR.get(L, "k"), label=f"{L} discrete layers")
        ax.set_xticks(np.arange(len(order)) + width * (len(Ls) - 1) / 2)
        ax.set_xticklabels([o.replace("generator", "gen").replace("_T", " T=") for o in order], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("D_c on held-out teacher targets (best of 100)"); ax.set_title("inverse test: c* → U"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / "figs" / f"e10_phase_a_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
