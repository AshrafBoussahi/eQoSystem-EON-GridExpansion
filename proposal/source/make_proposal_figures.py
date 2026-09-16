"""Figures for the eQoSystem E.ON Phase 1 concept proposal.

Every data point is read from result files produced earlier; nothing here runs
a new experiment. Outputs are vector PDF in ./figures.
"""
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
REPO = HERE.parents[1]
GIC = REPO / "qgridx" / "src" / "qgridx"
GEN = REPO / "genpce"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
C_Q = "#2166AC"; C_Q2 = "#4393C3"; C_CL = "#B2182B"; C_CL2 = "#D6604D"
C_NEU = "#878787"; C_ACC = "#1A9850"; C_AI = "#7B3294"
TEXTW = 6.9   # text width in inches at 1.8 cm margins on A4


def fig_workflow():
    """Hybrid pipeline: which layer does what, and what crosses the boundary."""
    fig, ax = plt.subplots(figsize=(TEXTW, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec, tfs=8.6, lfs=7.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#222222")
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.0 - 3.0 * i, ln, ha="center", va="center",
                    fontsize=lfs, color="#333333")

    def arrow(x0, y0, x1, y1, color="#444444"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                     lw=1.0, color=color, shrinkA=1, shrinkB=1))

    top = 25.0; h = 16.0; w = 23.0; gap = 2.5
    xs = [0.5 + i * (w + gap) for i in range(4)]
    box(xs[0], top, w, h, "Grid physics (classical)",
        ["pandapower power flow", "scenario set, N-1 outages",
         "congestion sensitivities"], "#F4F6FA", C_NEU)
    box(xs[1], top, w, h, "Coefficients (classical)",
        ["one quadratic program", "for all scenarios:",
         r"$c=\sum_s p_s c_s,\; Q=\sum_s p_s Q_s$"], "#F4F6FA", C_NEU)
    box(xs[2], top, w, h, "Quantum layer",
        [r"$m \leq 3\binom{n}{k}$ lines on $n$ qubits",
         "discrete native-gate circuit", "3 measurement settings"], "#E8F0FA", C_Q)
    box(xs[3], top, w, h, "Decoder (classical)",
        ["sign read-out, repair", "budget and radiality", "polish, N-1 screen"],
        "#F4F6FA", C_NEU)
    for i in range(3):
        arrow(xs[i] + w + 0.3, top + h / 2, xs[i + 1] - 0.3, top + h / 2)

    ay, ah, ax0, aw = 6.0, 11.0, 24.0, 52.0
    box(ax0, ay, aw, ah, "Circuit writer (AI, classical)",
        ["transformer generator or evolutionary search over gate tokens",
         "trained on the cost of the decoded, feasible plan"], "#F3ECF8", C_AI)
    # decoder -> writer (cost), writer -> quantum layer (next circuit)
    arrow(xs[3] + w * 0.5, top - 0.4, ax0 + aw - 6.0, ay + ah + 0.4, color=C_AI)
    ax.text(ax0 + aw + 1.5, ay + ah + 2.6, "decoded plan cost", ha="left", va="center",
            fontsize=8.5, color=C_AI)
    arrow(ax0 + 6.0, ay + ah + 0.4, xs[2] + w * 0.5, top - 0.4, color=C_AI)
    ax.text(ax0 - 1.5, ay + ah + 2.6, "next circuit", ha="right", va="center",
            fontsize=8.5, color=C_AI)
    ax.text(99.5, 1.8, "Output: lines to build, congestion before and after, cost",
            ha="right", va="center", fontsize=9.0, color="#222222", style="italic")
    fig.savefig(OUT / "fig_workflow.pdf")
    plt.close(fig)


def fig_evidence():
    """(a) measured compression across the executed configurations;
    (b) solution quality against quantum budget at 858 variables."""
    r7 = pd.read_csv(GIC / "results/doe_phase3/e7_resource_table.csv")
    r7 = r7.drop_duplicates(subset=["n_qubits", "m_decisions"]).sort_values("m_decisions")
    m = r7.m_decisions.to_numpy(float); n = r7.n_qubits.to_numpy(float)

    curves = pd.read_csv(GEN / "results/e1/curves_all.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 2.6), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1.15]})

    a1.plot(m, m / n, "o-", color=C_Q, lw=1.6, ms=5, label="Executed end to end")
    a1.plot([180], [180 / 12], "s", color=C_ACC, ms=6, label="Rigetti Ankaa-3 (winning run)")
    a1.plot([4500], [4500 / 14], "D", mfc="white", mec=C_CL, mew=1.5, ms=7,
            label="4,500 decisions on 14 qubits (capacity)")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Binary decisions $m$")
    a1.set_ylabel("Decisions per qubit")
    a1.set_ylim(4, 1500); a1.set_xlim(30, 40000)
    a1.grid(alpha=0.25, lw=0.5)
    a1.legend(frameon=False, loc="upper left", handlelength=1.5, borderpad=0.1, fontsize=8.5)
    for mi, ni in [(45, 6), (105, 7), (9009, 15)]:
        a1.annotate(f"{mi} on {ni}", xy=(mi, mi / ni), textcoords="offset points",
                    xytext=(7, -10), ha="left", fontsize=8.5, color=C_Q)
    a1.set_title("(a) Register size against decision count", fontsize=10, loc="left")

    grid = np.logspace(np.log10(50), np.log10(6e5), 120)
    lab = {"genpce_es": ("Discrete PCE search (ours)", C_Q, "-"),
           "pce_vqa_L8": ("Variational PCE, 8 layers", C_CL, "-"),
           "pce_vqa_L4": ("Variational PCE, 4 layers", C_CL2, "--"),
           "random": ("Random circuits", C_NEU, ":")}
    sub = curves[curves.m == 858]
    for meth, (name, col, ls) in lab.items():
        runs = []
        for (_, _), g in sub[sub.method == meth].groupby(["instance", "run"]):
            g = g.sort_values("evaluations")
            x = g.evaluations.to_numpy(float); y = np.maximum.accumulate(g.ratio.to_numpy(float))
            yy = np.interp(grid, x, y, left=np.nan, right=np.nan)
            runs.append(yy)
        if not runs:
            continue
        R = np.vstack(runs)
        mean = np.nanmean(R, axis=0)
        ok = np.sum(~np.isnan(R), axis=0) >= max(1, R.shape[0] // 2)
        a2.plot(grid[ok], mean[ok], ls, color=col, lw=1.6, label=name)
    a2.set_xscale("log")
    a2.set_xlabel("Circuit executions (each is 3 measurement settings)")
    a2.set_ylabel("Approximation ratio")
    a2.set_ylim(0.55, 0.76)
    a2.axvline(5000, color="#999999", lw=0.8, ls="--")
    a2.text(5000, 0.556, " 5,000", fontsize=8.5, color="#666666", va="bottom")
    a2.grid(alpha=0.25, lw=0.5)
    a2.legend(frameon=False, loc="upper left", fontsize=8.5, handlelength=1.8)
    a2.set_title("(b) 858 variables on 13 qubits, mean over instances and runs",
                 fontsize=10, loc="left")
    fig.savefig(OUT / "fig_evidence.pdf")
    plt.close(fig)


def fig_capacity():
    """Capacity of the registers the PoC will use."""
    fig, ax = plt.subplots(figsize=(TEXTW * 0.62, 2.3))
    ns = np.arange(6, 41)
    for k, col, ls in [(2, C_Q, "-"), (3, C_Q2, "--"), (4, C_ACC, ":")]:
        ax.plot(ns, [3 * comb(int(x), k) for x in ns], ls, color=col, lw=1.6, label=f"$k={k}$")
    ax.axhline(2000, color=C_CL, lw=0.9, ls="-.")
    ax.text(6.3, 2600, "2,000 candidate lines", fontsize=8.5, color=C_CL)
    ax.set_yscale("log")
    ax.set_xlabel("Qubits $n$"); ax.set_ylabel(r"Capacity $3\binom{n}{k}$")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "fig_capacity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_workflow()
    fig_evidence()
    fig_capacity()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
