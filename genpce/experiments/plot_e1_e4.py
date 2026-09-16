"""Figures and tables for E1 (quality vs quantum budget) and E4 (transfer / amortisation)."""

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

STYLE = {
    "genpce_es": dict(color="#1f77b4", label="GenPCE-ES (discrete, gradient-free)"),
    "random": dict(color="#7f7f7f", label="random circuits"),
    "pce_vqa_L4": dict(color="#ff7f0e", label="PCE-VQA, 4 layers (parameter shift)"),
    "pce_vqa_L8": dict(color="#d62728", label="PCE-VQA, 8 layers (parameter shift)"),
}


def _step_curve(sub: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Best-so-far ratio on a common evaluation grid, averaged over runs."""
    out = []
    for _, s in sub.groupby("run"):
        s = s.sort_values("evaluations")
        vals = np.interp(grid, s.evaluations, np.maximum.accumulate(s.ratio), left=np.nan)
        out.append(vals)
    return np.nanmean(np.stack(out), axis=0)


def plot_e1(curves: pd.DataFrame, final: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    instances = list(dict.fromkeys(curves.instance))
    ncol = min(4, len(instances))
    nrow = int(np.ceil(len(instances) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), instances):
        sub = curves[curves.instance == name]
        xmax = sub.evaluations.max()
        grid = np.geomspace(50, xmax, 200)
        for method, st in STYLE.items():
            s = sub[sub.method == method]
            if s.empty:
                continue
            ax.plot(grid, _step_curve(s, grid), color=st["color"], lw=1.6, label=st["label"])
        ax.axhline(16 / 17, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(0.5, 1.02)
        ax.set_title(f"{name}  (n = {int(sub.n.iloc[0])})", fontsize=9)
        ax.set_xlabel("circuit executions (× 3 settings)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("best approximation ratio")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    for ax in axes.ravel()[len(instances):]:
        ax.axis("off")
    fig.suptitle("E1 — solution quality vs quantum budget")
    fig.tight_layout()
    fig.savefig(out_dir / "e1_quality_vs_budget.png", dpi=150)
    plt.close(fig)

    # matched-budget table: ratio reached by each method at the ES budget
    rows = []
    for (m, method), s in curves.groupby(["m", "method"]):
        budget = curves[(curves.m == m) & (curves.method == "genpce_es")].evaluations.max()
        vals = []
        for (inst, run), ss in s.groupby(["instance", "run"]):
            ss = ss.sort_values("evaluations")
            ok = ss[ss.evaluations <= budget]
            vals.append(ok.ratio.max() if len(ok) else np.nan)
        rows.append({"m": m, "method": method, f"ratio@{'ES budget'}": np.nanmean(vals), "final ratio": s.groupby(["instance", "run"]).ratio.max().mean(),
                     "final evaluations": s.groupby(["instance", "run"]).evaluations.max().mean()})
    table = pd.DataFrame(rows)
    fin = final.groupby(["m", "method"]).agg(ratio_raw=("ratio_raw", "mean"), ratio_ls=("ratio_ls", "mean"),
                                             ratio_1000shots=("ratio_1000shots", "mean"), ratio_4000shots=("ratio_4000shots", "mean"),
                                             median_abs_corr=("median_abs_corr", "mean")).reset_index()
    table = table.merge(fin, on=["m", "method"], how="left")
    return table


def plot_e4(curves: pd.DataFrame, targets: pd.DataFrame, out_dir: Path, tag: str) -> None:
    methods = ["cold", "db_rescore", "prior_init", "prior_prop"]
    colors = dict(zip(methods, ["#7f7f7f", "#2ca02c", "#1f77b4", "#9467bd"]))
    grid = np.geomspace(50, curves.evaluations.max(), 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in methods:
        s = curves[curves.method == method]
        if s.empty:
            continue
        per_target = [_step_curve(ss.rename(columns={"run": "run"}), grid) for _, ss in s.groupby("target")]
        ax.plot(grid, np.nanmean(np.stack(per_target), axis=0), color=colors[method], lw=1.8, label=method)
    ax.set_xscale("log")
    ax.set_xlabel("circuit executions on the new instance (× 3 settings)")
    ax.set_ylabel("best approximation ratio (mean over held-out targets)")
    ax.axhline(16 / 17, color="k", ls=":", lw=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("E4 — amortisation across instances")
    fig.tight_layout()
    fig.savefig(out_dir / f"e4_transfer_{tag}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-tag", default="main")
    ap.add_argument("--e4-tag", default="m60")
    args = ap.parse_args()
    figs = RESULTS / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    e1c, e1f = RESULTS / "e1" / f"curves_{args.e1_tag}.csv", RESULTS / "e1" / f"final_{args.e1_tag}.csv"
    if e1c.exists() and e1f.exists():
        table = plot_e1(pd.read_csv(e1c), pd.read_csv(e1f), figs)
        md = table.round(3).to_markdown(index=False)
        (RESULTS / "e1" / f"summary_{args.e1_tag}.md").write_text(md)
        print(md)
    e4c, e4t = RESULTS / "e4" / f"curves_{args.e4_tag}.csv", RESULTS / "e4" / f"targets_{args.e4_tag}.csv"
    if e4c.exists() and e4t.exists():
        plot_e4(pd.read_csv(e4c), pd.read_csv(e4t), figs, args.e4_tag)


if __name__ == "__main__":
    main()
