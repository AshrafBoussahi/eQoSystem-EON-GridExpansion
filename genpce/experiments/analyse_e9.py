"""Merge the parallel confirmation shards, run the paired statistics, and draw the figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402
from e9_confirmation import analyse  # noqa: E402

STYLE = {"es_1_2": ("#7f7f7f", "search (1–2 edits, baseline)"), "es_single": ("#bcbd22", "search (single edits only)"),
         "es_learned": ("#1f77b4", "search + learned edits (online)"), "es_transfer_dc": ("#9467bd", "search + learned edits (pretrained elsewhere)")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RESULTS / "e9")
    ap.add_argument("--shards", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    finals = [pd.read_csv(args.dir / f"final_{s}.csv") for s in args.shards if (args.dir / f"final_{s}.csv").exists()]
    curves = [pd.read_csv(args.dir / f"curves_{s}.csv") for s in args.shards if (args.dir / f"curves_{s}.csv").exists()]
    if not finals:
        print("no shards found")
        return
    f = pd.concat(finals, ignore_index=True)
    c = pd.concat(curves, ignore_index=True)
    f.to_csv(args.dir / f"final_{args.tag}.csv", index=False)
    c.to_csv(args.dir / f"curves_{args.tag}.csv", index=False)
    print(f"merged {len(f)} runs over {f.instance.nunique()} instances × {f.run.nunique()} seeds × {f.arm.nunique()} arms\n")

    # per-instance means, so the reader can see consistency rather than a pooled average
    piv = f.pivot_table(index="instance", columns="arm", values="ratio_raw", aggfunc="mean").round(3)
    print("=== mean final ratio per instance")
    print(piv.to_string())
    piv.to_csv(args.dir / f"per_instance_{args.tag}.csv")
    analyse(args.dir / f"final_{args.tag}.csv", args.budget)

    # figure: median best-so-far curve per arm (pooled over instances and seeds)
    grid = np.geomspace(50, args.budget, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, (color, label) in STYLE.items():
        s = c[c.arm == arm]
        if s.empty:
            continue
        per_run = []
        for _, ss in s.groupby(["instance", "run"]):
            ss = ss.sort_values("evaluations")
            per_run.append(np.interp(grid, ss.evaluations, np.maximum.accumulate(ss.ratio_best)))
        M = np.stack(per_run)
        axes[0].plot(grid, np.median(M, axis=0), color=color, lw=1.8, label=label)
        axes[0].fill_between(grid, np.quantile(M, 0.25, axis=0), np.quantile(M, 0.75, axis=0), color=color, alpha=0.15)
    axes[0].set_xscale("log"); axes[0].set_xlabel("circuit executions"); axes[0].set_ylabel("best approximation ratio")
    axes[0].set_title("median best-so-far (IQR shaded)"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8, loc="lower right")

    base = f[f.arm == "es_1_2"].set_index(["instance", "run"]).ratio_raw
    offs, labels, colors = [], [], []
    for arm, (color, label) in STYLE.items():
        if arm == "es_1_2" or f[f.arm == arm].empty:
            continue
        a = f[f.arm == arm].set_index(["instance", "run"]).ratio_raw
        idx = a.index.intersection(base.index)
        offs.append((a.loc[idx] - base.loc[idx]).values); labels.append(label.replace("search + ", "")); colors.append(color)
    if offs:
        bp = axes[1].boxplot(offs, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.4)
        axes[1].axhline(0, color="k", lw=1, ls=":")
        axes[1].set_ylabel("paired difference in final ratio\n(arm − baseline, same instance & seed)")
        axes[1].set_title("paired effect vs the unrestricted search"); axes[1].grid(alpha=0.3, axis="y")
        plt.setp(axes[1].get_xticklabels(), rotation=12, ha="right", fontsize=8)
    fig.tight_layout()
    out = RESULTS / "figs" / f"e9_confirmation_{args.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
