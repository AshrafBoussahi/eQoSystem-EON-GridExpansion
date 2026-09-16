"""Figures for the E0 pilot: PCE-VQA depth scan, GenPCE learning curves, matched-budget comparison.

Reads ``results/e0/*.csv`` written by ``e0_pce_vqa.py`` and ``e0_genpce.py``; writes PNGs to
``results/e0/figs`` and a Markdown summary table to ``results/e0/summary.md``.
"""

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

COLORS = {"genpce": "#1f77b4", "random": "#7f7f7f", "pce_vqa": "#d62728"}


def plot_pce_vqa_depth_scan(df: pd.DataFrame, out: Path) -> None:
    sizes = sorted(df.m.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.6 * len(sizes), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, sizes):
        sub = df[df.m == m]
        for fam, mk in (("reg3", "o"), ("er4", "s")):
            s = sub[sub.family == fam]
            if s.empty:
                continue
            g = s.groupby("two_qubit")
            ax.errorbar(g.ratio_raw.mean().index, g.ratio_raw.mean(), yerr=g.ratio_raw.std(), marker=mk, ls="-", color=COLORS["pce_vqa"], label=f"{fam} raw")
            ax.errorbar(g.ratio_ls.mean().index, g.ratio_ls.mean(), yerr=g.ratio_ls.std(), marker=mk, ls="--", color=COLORS["pce_vqa"], alpha=0.6, label=f"{fam} +bit-swap")
        ax.axhline(16 / 17, color="k", ls=":", lw=1, label="16/17 hardness")
        ax.set_title(f"m = {m} (n = {int(sub.n.iloc[0])})")
        ax.set_xlabel("two-qubit (MS) gates")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("approximation ratio (mean ± sd)")
    axes[0].legend(fontsize=7)
    fig.suptitle("E0-A  PCE-VQA (Sciorilli-style brickwork + relaxed loss, Adam)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_genpce_curves(curves: pd.DataFrame, rand: pd.DataFrame | None, out: Path) -> None:
    instances = list(dict.fromkeys(curves.instance))
    ncol = 4
    nrow = int(np.ceil(len(instances) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), instances):
        sub = curves[curves.instance == name]
        for run, s in sub.groupby("run"):
            ax.plot(s.evaluations, s.ratio_best, color=COLORS["genpce"], alpha=0.9, lw=1.2, label="GenPCE best (raw)" if run == 0 else None)
            ax.plot(s.evaluations, s.ratio_best_ls, color=COLORS["genpce"], alpha=0.5, lw=1, ls="--", label="GenPCE best +bit-swap" if run == 0 else None)
            ax.plot(s.evaluations, s.ratio_epoch_mean, color="#2ca02c", alpha=0.6, lw=0.8, label="GenPCE epoch mean" if run == 0 else None)
        if rand is not None:
            r = rand[rand.instance == name]
            if not r.empty:
                ax.plot(r.evaluations, r.ratio_best, color=COLORS["random"], lw=1.2, label="random circuits best")
        ax.axhline(16 / 17, color="k", ls=":", lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("circuit evaluations")
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("approximation ratio")
    axes[0, 0].legend(fontsize=7)
    for ax in axes.ravel()[len(instances):]:
        ax.axis("off")
    fig.suptitle("E0-B  GenPCE learning curves vs random-circuit control")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def summary_table(vqa: pd.DataFrame | None, gen: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if gen is not None:
        for (fam, m), s in gen.groupby(["family", "m"]):
            g = s[s.method == "genpce"]
            r = s[s.method == "random"]
            rows.append({
                "family": fam, "m": m, "n": int(s.n.iloc[0]),
                "GenPCE raw (mean)": g.ratio_raw.mean(), "GenPCE +LS (mean)": g.ratio_ls.mean(),
                "GenPCE raw (best run)": g.ratio_raw.max(),
                "GenPCE evals": int(g.evaluations.mean()) if not g.empty else None,
                "GenPCE 2q gates": g.two_qubit.mean(), "GenPCE non-Clifford": g.non_clifford.mean(),
                "random raw (mean)": r.ratio_raw.mean(), "random +LS (mean)": r.ratio_ls.mean(),
            })
    table = pd.DataFrame(rows)
    if vqa is not None and not table.empty:
        best_depth = vqa.groupby(["family", "m", "two_qubit"]).ratio_raw.mean().reset_index()
        idx = best_depth.groupby(["family", "m"]).ratio_raw.idxmax()
        bd = best_depth.loc[idx].rename(columns={"two_qubit": "VQA best depth (2q)", "ratio_raw": "VQA raw (mean @best depth)"})
        table = table.merge(bd, on=["family", "m"], how="left")
        ls = vqa.groupby(["family", "m"]).ratio_ls.mean().rename("VQA +LS (mean, all depths)").reset_index()
        table = table.merge(ls, on=["family", "m"], how="left")
        ce = vqa.groupby(["family", "m"]).circuit_equiv.mean().rename("VQA circuit-equiv (mean)").reset_index()
        table = table.merge(ce, on=["family", "m"], how="left")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    ap.add_argument("--dir", type=Path, default=RESULTS / "e0")
    args = ap.parse_args()
    figs = args.dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    vqa = pd.read_csv(args.dir / "pce_vqa.csv") if (args.dir / "pce_vqa.csv").exists() else None
    curves_p = args.dir / f"genpce_curves_{args.tag}.csv"
    curves = pd.read_csv(curves_p) if curves_p.exists() else None
    rand_p = args.dir / f"random_curves_{args.tag}.csv"
    rand = pd.read_csv(rand_p) if rand_p.exists() else None
    summ_p = args.dir / f"genpce_summary_{args.tag}.csv"
    gen = pd.read_csv(summ_p) if summ_p.exists() else None
    if vqa is not None:
        plot_pce_vqa_depth_scan(vqa, figs / "e0a_pce_vqa_depth_scan.png")
    if curves is not None:
        plot_genpce_curves(curves, rand, figs / f"e0b_genpce_curves_{args.tag}.png")
    table = summary_table(vqa, gen)
    if not table.empty:
        md = table.to_markdown(index=False, floatfmt=".3f")
        (args.dir / f"summary_{args.tag}.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
