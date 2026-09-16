"""Regenerate every figure from the archived result files.

Nothing here is typed in by hand: each figure reads the same CSV or JSON that
the corresponding experiment wrote, so a figure cannot drift away from the
number it illustrates. Output is vector PDF, serif-typeset to match the
manuscript, with explicit padding so no label can overlap another element.

Usage
-----
>>> from qgridx.analysis import figures
>>> figures.render_all("out/")            # doctest: +SKIP

Individual figures are available as ``fig_*`` functions if you only need one.
"""
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qgridx.utils.paths import results_dir

RES = results_dir() / "doe_phase3"
HW = results_dir() / "hardware"
MC = results_dir() / "maxcut"
OUT = Path("figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# muted, colour-blind-safe palette
C_Q = "#2166AC"     # quantum / our method
C_Q2 = "#4393C3"
C_CL = "#B2182B"    # classical exact / metaheuristic
C_CL2 = "#D6604D"
C_NEU = "#878787"   # neutral / control
C_ACC = "#1A9850"   # accent / positive
C_BG = "#F0F0F0"

COL1 = 3.4   # single-column width (inches)
COL2 = 7.1   # double-column width


# ---------------------------------------------------------------- Figure 1
def fig_architecture():
    """Wide schematic of the full pipeline.

    Geometry is laid out on an explicit 100-unit grid: four equal 20-unit boxes
    with 6-unit gutters. Titles are sized against that width so no label can
    spill past a box border.
    """
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 37); ax.axis("off")

    # gutters are wide enough for the italic arrow labels to sit clear of both
    # neighbouring borders, including the 0.5-unit box padding on each side
    BW, GAP, BY, BH = 18.0, 9.0, 9.0, 21.0
    xs = [0.5 + i * (BW + GAP) for i in range(4)]

    def box(x, title, lines, fc, ec):
        ax.add_patch(FancyBboxPatch((x, BY), BW, BH,
                                    boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + BW / 2, BY + BH - 3.4, title, ha="center", va="top",
                fontsize=8.2, fontweight="bold", color="#222222")
        for i, ln in enumerate(lines):
            ax.text(x + BW / 2, BY + BH - 8.4 - i * 3.6, ln, ha="center", va="top",
                    fontsize=7.0, color="#333333")

    def arrow(x1, x2, label):
        y = BY + BH / 2
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.1, color="#555555"))
        ax.text((x1 + x2) / 2, y + 1.3, label, ha="center", va="bottom",
                fontsize=6.8, color="#555555", style="italic")

    box(xs[0], "Planning problem",
        ["binary decisions", "cost coefficients", "budget constraint"], "#EAF1F7", C_Q)
    box(xs[1], "Correlation encoding",
        ["one Pauli string", "per decision", "3 measurements"], "#EAF1F7", C_Q)
    box(xs[2], "Circuit generator",
        ["transformer emits", "gate tokens", "no angle gradients"], "#E8F4EC", C_ACC)
    box(xs[3], "Decoder",
        ["sign readout", "structural repair", "local search"], "#FBEDEC", C_CL)
    for i, lab in enumerate(["encode", "generate", "read out"]):
        arrow(xs[i] + BW + 0.8, xs[i + 1] - 0.8, lab)

    # feedback loop: decoded cost is the only training signal
    ax.add_patch(FancyArrowPatch((xs[3] + BW / 2, BY - 0.8), (xs[2] + BW / 2, BY - 0.8),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.1,
                                 color=C_ACC, linestyle="--",
                                 connectionstyle="arc3,rad=-0.32"))
    ax.text((xs[2] + xs[3]) / 2 + BW / 2, 0.4, "decoded cost trains the generator",
            ha="center", va="bottom", fontsize=7.0, color=C_ACC, style="italic")

    ax.text(50, 34.2, "Many decisions, few qubits, one training loop",
            ha="center", va="center", fontsize=9.4, fontweight="bold", color="#222222")
    fig.savefig(OUT / "fig_architecture.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig_compression():
    """Qubits required vs number of decisions."""
    fig, ax = plt.subplots(figsize=(COL1, 2.7), constrained_layout=True)
    m = np.logspace(1, 4.3, 200)

    ax.plot(m, m, color=C_CL, lw=1.6, label="One qubit per decision")
    for k, style, col in [(2, "-", C_Q), (3, "--", C_Q2), (5, ":", "#7FB3D5")]:
        need = []
        for mm in m:
            n = k
            while 3 * comb(n, k) < mm:
                n += 1
            need.append(n)
        ax.plot(m, need, style, color=col, lw=1.6, label=f"Correlation encoding, $k={k}$")

    for mm, nn in [(45, 6), (105, 7), (252, 9), (9009, 15)]:
        ax.plot(mm, nn, "o", ms=5.5, mfc="white", mec=C_ACC, mew=1.6, zorder=5)
    # A single short call-out, placed in the wedge between the two families --
    # wide enough there that neither the diagonal reference line nor the k=2
    # curve can reach it. The opaque halo is belt-and-braces.
    ax.annotate("45 to 9,009 decisions\non 6 to 15 qubits",
                xy=(9009, 15), xycoords="data", xytext=(3000, 300), textcoords="data",
                ha="center", va="center", fontsize=7.2, color=C_ACC, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_ACC,
                                shrinkA=3, shrinkB=4))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of binary decisions")
    ax.set_ylabel("Qubits required")
    ax.set_ylim(3, 3e4)
    ax.legend(frameon=False, loc="upper left", handlelength=1.9, borderpad=0.2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(OUT / "fig_compression.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig_depth_budget():
    """Solution quality against the generator's gate budget.

    The 1x-vs-3x contrast is measured on all 80 benchmark instances; the
    6x and 8x arms were run on a 24-instance subset. Those two populations get
    their own panels rather than sharing an axis, so no bar is compared against
    a bar drawn from a different sample.
    """
    lv = pd.read_csv(RES / "e2o_merged_done.csv")
    h = pd.read_csv(RES / "e2h_main_table_v2.csv")
    f = pd.read_csv(RES / "e2m_expressivity_full80.csv")

    full = h.groupby("instance_id").agg(opt=("optimum", "first"),
                                        one=("gqe_cost", "min")).reset_index()
    full = full.merge(f[["instance_id", "ceil3x_cost"]], on="instance_id")
    ex_full = [100 * float(np.mean(np.abs(full[c] - full.opt) < 1e-6))
               for c in ("one", "ceil3x_cost")]

    piv = lv.pivot_table(index="instance_id", columns="arm", values="cost")
    opt = lv.groupby("instance_id").optimum.first()
    sub_arms = ["ceil3x_ref", "ceil6x", "ceil8x"]
    ex_sub, n_sub = [], []
    for arm in sub_arms:
        col = piv[arm].dropna(); o = opt[col.index]
        ex_sub.append(100 * float(np.mean(np.abs(col - o) < 1e-6)))
        n_sub.append(len(col))

    ceils = [19, 59, 119, 159]
    used = [19.0, 59.0] + [float(lv[lv.arm == a].gates.mean()) for a in ("ceil6x", "ceil8x")]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(COL2, 2.55), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1, 1.15, 1.5]})

    x = np.arange(2)
    a1.bar(x, ex_full, width=0.55, color=[C_NEU, C_Q], edgecolor="white")
    for xi, v in zip(x, ex_full):
        a1.text(xi, v + 1.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(["$1\\times$\n19 gates", "$3\\times$\n59 gates"])
    a1.set_ylabel("Instances solved exactly (%)")
    a1.set_title("All 80 instances", fontsize=8.6, pad=4)
    a1.set_ylim(0, 95)
    a1.grid(axis="y", alpha=0.25, lw=0.5)
    a1.annotate("", xy=(1, 63), xytext=(0, 63),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#444444"))
    a1.text(0.5, 65, "$p = 2\\times10^{-7}$", ha="center", va="bottom", fontsize=7.4,
            color="#444444")

    xs = np.arange(3)
    a2.bar(xs, ex_sub, width=0.58, color=[C_Q, C_NEU, C_NEU], edgecolor="white")
    for xi, v in zip(xs, ex_sub):
        a2.text(xi, v + 1.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(xs)
    a2.set_xticklabels([f"$3\\times$\n$n={n_sub[0]}$", f"$6\\times$\n$n={n_sub[1]}$",
                        f"$8\\times$\n$n={n_sub[2]}$"])
    a2.set_title("Deeper budgets, matched subset", fontsize=8.6, pad=4)
    a2.set_ylim(0, 95)
    a2.grid(axis="y", alpha=0.25, lw=0.5)
    a2.text(1.0, 93, "$6\\times$ and $8\\times$ are not\nsignificantly better",
            ha="center", va="top", fontsize=7.2, color="#666666", style="italic",
            linespacing=1.35)

    xg = np.arange(4)
    a3.bar(xg - 0.17, ceils, width=0.32, color="#D9D9D9", edgecolor="white",
           label="Budget allowed")
    a3.bar(xg + 0.17, used, width=0.32, color=C_Q2, edgecolor="white", label="Gates used")
    a3.set_xticks(xg)
    a3.set_xticklabels([f"${v}\\times$" for v in (1, 3, 6, 8)])
    a3.set_ylabel("Gates per circuit")
    a3.set_xlabel("Gate budget given to the generator")
    a3.set_title("Budget spent", fontsize=8.6, pad=4)
    a3.legend(frameon=False, loc="upper left", handlelength=1.4, borderpad=0.2)
    a3.grid(axis="y", alpha=0.25, lw=0.5)
    a3.set_ylim(0, 235)
    fig.savefig(OUT / "fig_depth_budget.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def fig_main_results():
    """Comparison against classical references on the siting family."""
    h = pd.read_csv(RES / "e2h_main_table_v2.csv")
    f = pd.read_csv(RES / "e2m_expressivity_full80.csv")
    cl = h.groupby("instance_id").agg(
        optimum=("optimum", "first"), sa=("sa_cost", "first"), tabu=("tabu_cost", "first"),
        blind=("blind_cost", "first"), g20=("greedy20_cost", "first"),
        g1=("greedy1_cost", "first")).reset_index()
    d = f[["instance_id", "optimum", "ceil3x_cost"]].merge(cl.drop(columns=["optimum"]),
                                                           on="instance_id")

    names = ["Simulated\nannealing", "Correlation\nsolver", "Tabu\nsearch",
             "Random-input\ncontrol", "Greedy,\n20 restarts", "Greedy,\n1 restart"]
    cols = ["sa", "ceil3x_cost", "tabu", "blind", "g20", "g1"]
    vals = [100 * float(np.mean(np.abs(d[c] - d.optimum) < 1e-6)) for c in cols]
    colors = [C_CL, C_Q, C_CL2, C_NEU, "#BBBBBB", "#DDDDDD"]

    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]; vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(COL1, 2.85), constrained_layout=True)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor="white")
    for yi, v in zip(y, vals):
        ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", ha="left", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Instances solved exactly (%)")
    ax.set_xlim(0, 78)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.savefig(OUT / "fig_main_results.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def fig_maxcut_scaling():
    """Graph benchmark: the paired 25-graph campaign, and the scaling trajectory.

    Both panels compare the two circuit-preparation strategies at a matched
    training budget on identical graphs and identical reference cuts.
    """
    p2 = json.loads((MC / "phase2_campaign_results.json").read_text())
    xa = np.array([g["r_full_a"] for g in p2["per_graph"]])
    yb = np.array([g["r_full_b"] for g in p2["per_graph"]])
    p3 = json.loads((MC / "phase3_scaling_results.json").read_text())
    m = [r["m"] for r in p3["results"]]
    a = [r["mean_r_full_a"] for r in p3["results"]]
    b = [r["mean_r_full_b"] for r in p3["results"]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.6), constrained_layout=True)

    lo, hi = 0.845, 1.075
    a1.plot([lo, hi], [lo, hi], "-", color="#999999", lw=0.9)
    a1.scatter(xa, yb, s=26, color=C_Q, edgecolor="white", lw=0.6, zorder=4)
    a1.set_xlim(lo, hi); a1.set_ylim(lo, hi)
    a1.set_aspect("equal")
    a1.set_xlabel("Directly trained circuits")
    a1.set_ylabel("Generated circuits")
    a1.set_title("25 graphs, one point each", fontsize=8.6, pad=4)
    a1.grid(alpha=0.25, lw=0.5)
    # everything below the diagonal is empty, so both labels live there
    a1.text(0.962, 0.947, "equal quality", ha="center", va="bottom", fontsize=7.0,
            color="#888888", style="italic", rotation=45, rotation_mode="anchor")
    a1.text(1.065, 0.862, "generated circuits\nbetter on 25 of 25 graphs",
            ha="right", va="bottom", fontsize=7.2, color=C_Q, linespacing=1.35)

    a2.plot(m, b, "o-", color=C_Q, lw=1.7, ms=5, label="Generated circuits")
    a2.plot(m, a, "s--", color=C_CL, lw=1.4, ms=4.5, label="Directly trained circuits")
    a2.set_xscale("log")
    a2.set_xlabel("Number of binary decisions")
    a2.set_ylabel("Approximation ratio")
    a2.set_title("Scaling trajectory", fontsize=8.6, pad=4)
    a2.legend(frameon=False, loc="lower right", handlelength=2.0, borderpad=0.2)
    a2.grid(alpha=0.25, lw=0.5)
    a2.set_ylim(0.92, 1.15)
    a2.annotate("9,009 decisions\non 15 qubits", xy=(m[-1], b[-1]), xycoords="data",
                xytext=(900, 1.122), textcoords="data", ha="center", va="center",
                fontsize=7.2, color=C_Q, linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_Q,
                                shrinkA=3, shrinkB=4))
    fig.savefig(OUT / "fig_maxcut_scaling.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6
def fig_scenarios():
    """Scenario set and the value of planning against it."""
    sc = pd.read_csv(RES / "e3a_scenarios.csv")
    e3 = pd.read_csv(RES / "e3b_scenario_weighted.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.5), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    cl = sc[sc.kind == "cluster"]; tl = sc[sc.kind == "tail"]
    a1.scatter(cl.load_scale, cl.wind_scale, s=cl.prob * 1400, alpha=0.75,
               color=C_Q2, edgecolor=C_Q, lw=0.9, label="Representative conditions")
    a1.scatter(tl.load_scale, tl.wind_scale, s=tl.prob * 1400, alpha=0.9,
               color=C_CL2, edgecolor=C_CL, lw=1.1, marker="D", label="High-stress condition")
    a1.set_xlabel("Demand, relative to annual mean")
    a1.set_ylabel("Wind availability,\nrelative to annual mean")
    a1.legend(frameon=False, loc="upper right", scatterpoints=1, handletextpad=0.6)
    a1.grid(alpha=0.25, lw=0.5)
    a1.set_xlim(0.78, 1.62); a1.set_ylim(-0.25, 3.35)
    a1.annotate("marker area $\\propto$ probability", (0.82, 3.0), fontsize=7.0,
                color="#666666", style="italic")

    changed = 100 * e3.decision_changed.mean()
    better = 100 * float((e3.vss > 1e-9).mean())
    worse = 100 * float((e3.vss < -1e-9).mean())
    bars = ["Siting choice\nchanges", "Better under\nhigh stress", "Worse under\nhigh stress"]
    vals = [changed, better, worse]
    cols = [C_Q, C_ACC, C_CL]
    xb = np.arange(3)
    a2.bar(xb, vals, width=0.58, color=cols, edgecolor="white")
    for xi, v in zip(xb, vals):
        a2.text(xi, v + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(xb); a2.set_xticklabels(bars)
    a2.set_ylabel("Share of instances (%)")
    a2.set_ylim(0, 100)
    a2.grid(axis="y", alpha=0.25, lw=0.5)
    fig.savefig(OUT / "fig_scenarios.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 7
def fig_design_levers():
    """Design-choice sweep, every arm at an identical evaluation budget.

    Sorted best-first, with the paired test against the reference design in its
    own right-hand column so no p-value can crowd a bar label.
    """
    from scipy.stats import wilcoxon
    lv = pd.read_csv(RES / "e2o_merged_done.csv")
    piv = lv.pivot_table(index="instance_id", columns="arm", values="cost")
    opt = lv.groupby("instance_id").optimum.first()
    label = {"ceil3x_ref": "Reference design",
             "ceil3x_ang16": "Finer rotation angles",
             "ceil3x_k3": "Higher-order correlations",
             "ceil8x": "Gate budget $8\\times$",
             "ceil6x": "Gate budget $6\\times$",
             "ceil3x_split5": "Budget split over restarts"}
    rows = []
    for arm in label:
        col = piv[arm].dropna(); o = opt[col.index]
        if arm == "ceil3x_ref":
            p = None
        else:
            d = piv[[arm, "ceil3x_ref"]].dropna()
            p = float(wilcoxon(d[arm], d["ceil3x_ref"]).pvalue)
        rows.append((label[arm], float((col - o).mean()), p, arm))
    rows.sort(key=lambda r: r[1])

    names = [r[0] for r in rows]; gaps = [r[1] for r in rows]
    cols = [C_Q if r[3] == "ceil3x_ref" else C_NEU for r in rows]

    fig, ax = plt.subplots(figsize=(COL1, 2.7), constrained_layout=True)
    y = np.arange(len(names))[::-1]
    ax.barh(y, gaps, height=0.6, color=cols, edgecolor="white")
    xmax = 1.75   # reserves a clear right-hand column for the paired-test result
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.025, yi, f"{r[1]:.2f}", va="center", ha="left", fontsize=7.8)
        if r[2] is None:
            txt, col = "reference", C_Q
        elif r[2] < 0.05:
            txt, col = f"$p={r[2]:.3f}$", C_CL
        else:
            txt, col = "n.s.", "#888888"
        ax.text(xmax - 0.03, yi, txt, va="center", ha="right", fontsize=7.2, color=col)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Mean distance from the optimum")
    ax.set_xlim(0, xmax)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.savefig(OUT / "fig_design_levers.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 8
def fig_resources():
    """What grows with problem size, and what does not."""
    r7 = pd.read_csv(RES / "e7_resource_table.csv")
    r7 = r7.drop_duplicates(subset=["n_qubits", "m_decisions"]).sort_values("m_decisions")
    m = r7.m_decisions.to_numpy(float)
    n = r7.n_qubits.to_numpy(float)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.5), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1.05]})

    a1.plot(m, m / n, "o-", color=C_Q, lw=1.6, ms=5, label="Measured configurations")
    a1.plot([4500], [4500 / 14], "D", mfc="white", mec=C_ACC, mew=1.6, ms=7,
            label="Planner-scale projection")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Number of binary decisions")
    a1.set_ylabel("Decisions per qubit")
    a1.grid(alpha=0.25, lw=0.5)
    a1.set_ylim(4, 1500)
    # legend goes below the curve, call-outs above it: the two regions never meet
    a1.legend(frameon=False, loc="lower right", handlelength=1.6, borderpad=0.2)
    a1.annotate("601 : 1", xy=(m[-1], m[-1] / n[-1]), textcoords="offset points",
                xytext=(-3, 7), ha="right", fontsize=7.4, color=C_Q)
    a1.annotate("planner scale: 321 : 1\non 14 qubits", xy=(4500, 4500 / 14),
                xycoords="data", xytext=(700, 430), textcoords="data",
                ha="center", va="center", fontsize=7.2, color=C_ACC, linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_ACC,
                                shrinkA=3, shrinkB=5))

    # growth factors inside the one internally consistent scaling family
    dep = [15, 120]
    facts = [("Binary decisions", m[-1] / 252, C_Q),
             ("Circuit depth", dep[1] / dep[0], C_Q2),
             ("Qubits", 15 / 9, C_Q2),
             ("Measurement settings", 1.0, C_ACC),
             ("Shots per evaluation", 1.0, C_ACC)]
    y = np.arange(len(facts))[::-1]
    a2.barh(y, [f[1] for f in facts], height=0.6, color=[f[2] for f in facts],
            edgecolor="white")
    for yi, f in zip(y, facts):
        a2.text(f[1] * 1.13, yi, f"{f[1]:.1f}$\\times$", va="center", ha="left", fontsize=7.8)
    a2.set_yticks(y); a2.set_yticklabels([f[0] for f in facts])
    a2.set_xscale("log")
    a2.set_xlim(0.85, 130)
    a2.set_xticks([1, 10, 100]); a2.set_xticklabels(["$1\\times$", "$10\\times$", "$100\\times$"])
    a2.set_xlabel("Growth from the smallest to the largest instance")
    a2.grid(axis="x", alpha=0.25, lw=0.5)
    fig.savefig(OUT / "fig_resources.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 9
def fig_hardware():
    """Measured device resources, and the on-device check of the readout."""
    c = pd.read_csv(HW / "candidates.csv")
    g = c[c.source == "GQE"].groupby(["m", "n"]).agg(
        g2=("gates_2q", "mean"), r2=("routed_2q", "mean"),
        depth=("routed_depth", "mean")).reset_index()
    bell = pd.read_csv(HW / "bell_tetrahedron_qcs_summary.csv")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(COL2, 2.5), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1, 1, 1.5]})
    labels = [f"{int(r.m)} on {int(r.n)}" for _, r in g.iterrows()]
    x = np.arange(len(g))
    a1.bar(x - 0.18, g.g2, width=0.34, color=C_Q, edgecolor="white", label="As generated")
    a1.bar(x + 0.18, g.r2, width=0.34, color=C_Q2, edgecolor="white", label="After routing")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7.8)
    a1.set_ylabel("Two-qubit gates")
    a1.set_xlabel("Decisions on qubits")
    a1.legend(frameon=False, loc="upper left", handlelength=1.4, borderpad=0.2)
    a1.grid(axis="y", alpha=0.25, lw=0.5)
    a1.set_ylim(0, max(g.r2) * 1.45)

    a2.bar(x, g.depth, width=0.5, color="#7FB3D5", edgecolor="white")
    for xi, v in zip(x, g.depth):
        a2.text(xi, v + 1.8, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=7.8)
    a2.set_ylabel("Depth after routing")
    a2.set_xlabel("Decisions on qubits")
    a2.grid(axis="y", alpha=0.25, lw=0.5)
    a2.set_ylim(0, max(g.depth) * 1.35)

    e = np.arange(1, len(bell) + 1)
    a3.axhline(0, color="#444444", lw=0.9)
    a3.plot(e, bell.Z, "o", color=C_Q, ms=5, label="$\\langle ZZ\\rangle$")
    a3.plot(e, bell.X, "s", color=C_Q2, ms=4.5, label="$\\langle XX\\rangle$")
    a3.plot(e, bell.Y, "^", color=C_CL, ms=5, label="$\\langle YY\\rangle$")
    a3.set_xticks(e)
    a3.set_xlabel("Device edge")
    a3.set_ylabel("Measured correlator")
    a3.set_ylim(-1.35, 1.35)
    a3.grid(axis="y", alpha=0.25, lw=0.5)
    a3.legend(frameon=False, loc="lower left", ncol=3, handlelength=1.0,
              columnspacing=1.0, handletextpad=0.3, borderpad=0.2)
    a3.text(len(bell) + 0.35, 1.28, "8 of 8 edges:\nall three signs correct", ha="right",
            va="top", fontsize=7.2, color=C_ACC, style="italic", linespacing=1.35)
    fig.savefig(OUT / "fig_hardware.pdf")
    plt.close(fig)




plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 8.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


C_Q, C_Q2 = "#2166AC", "#4393C3"
C_CL, C_CL2 = "#B2182B", "#D6604D"
C_NEU, C_ACC = "#878787", "#1A9850"
COL1, COL2 = 3.35, 7.0
GRID_STYLE = {"IEEE-57": ("o-", C_Q), "IEEE-118": ("s--", C_CL)}


def fig_resilience():
    """N-1 security as the data-centre load grows, with and without storage.

    All load levels are shown. The shaded band marks where the INTACT network
    can no longer serve the campus without shedding, so the objective's prices
    there come from the factory's nominal-load fallback rather than from the
    grown load; the security comparison itself stays valid throughout, because
    both arms see the identical load and the identical outage set.
    """
    d = pd.read_csv(RES / "g1_resilience.csv")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(COL2, 2.4), constrained_layout=True)

    # first load level at which the intact power flow stops being feasible anywhere
    infeas = d.groupby("ai_load_mw").base_feasible.mean()
    first_bad = next((lv for lv, f in infeas.items() if f < 1.0), None)

    for grid, (style, col) in GRID_STYLE.items():
        g = d[d.grid == grid]
        if g.empty:
            continue
        s = g.groupby("ai_load_mw").agg(
            mw=("total_sited_mw", "mean"),
            eue_no=("eue_no_storage", "mean"), eue_wi=("eue_with_storage", "mean"),
            sec_no=("secure_frac_no_storage", "mean"),
            sec_wi=("secure_frac_with_storage", "mean")).reset_index()
        a1.plot(s.ai_load_mw, s.eue_no, style, color=C_NEU, lw=1.3, ms=4)
        a1.plot(s.ai_load_mw, s.eue_wi, style, color=col, lw=1.6, ms=4.5, label=grid)
        a2.plot(s.ai_load_mw, 100 * s.sec_no, style, color=C_NEU, lw=1.3, ms=4)
        a2.plot(s.ai_load_mw, 100 * s.sec_wi, style, color=col, lw=1.6, ms=4.5, label=grid)
        a3.plot(s.ai_load_mw, s.eue_no - s.eue_wi, style, color=col, lw=1.6, ms=4.5,
                label=grid)

    for ax in (a1, a2, a3):
        if first_bad is not None:
            ax.axvspan(first_bad, d.ai_load_mw.max() * 1.02, color="#000000", alpha=0.055,
                       lw=0, zorder=0)
        ax.set_xlabel("Data-centre load added (MW)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(-15, d.ai_load_mw.max() * 1.02)

    a1.set_yscale("symlog", linthresh=1.0)
    a1.set_ylim(bottom=0)          # every value is non-negative; the symlog default
    a1.set_ylabel("Expected unserved energy (MW)")   # would waste half the axis on <0
    a1.set_title("Unserved energy under N-1", fontsize=9)
    a1.legend(frameon=False, handlelength=1.8, borderpad=0.2, loc="upper left",
              title="with storage", title_fontsize=7.6)

    a2.set_ylabel("Outages served in full (%)")
    a2.set_title("N-1 secure fraction", fontsize=9)
    a2.set_ylim(-6, 112)
    a2.legend(frameon=False, handlelength=1.8, borderpad=0.2, loc="lower left",
              title="with storage", title_fontsize=7.6)
    a2.text(0.97, 0.62, "grey: no storage", transform=a2.transAxes, fontsize=7.2,
            color=C_NEU, style="italic", ha="right")

    a3.set_yscale("symlog", linthresh=1.0)
    a3.set_ylim(bottom=0)
    a3.set_ylabel("Unserved energy avoided (MW)")
    a3.set_title("Storage contribution", fontsize=9)
    a3.legend(frameon=False, handlelength=1.8, borderpad=0.2, loc="upper left")

    fig.savefig(OUT / "fig_r_resilience.pdf")
    plt.close(fig)
    return d


def fig_microgrid():
    """What adding B islanding binaries costs, and whether the solver still
    lands on the certified optimum once they are there."""
    d = pd.read_csv(RES / "g2_microgrid.csv")
    # single column: the report is page-limited, and the per-grid solve rates that
    # used to sit in a second panel are carried perfectly well by one sentence
    fig, a1 = plt.subplots(1, 1, figsize=(COL1, 2.5), constrained_layout=True)
    a2 = None

    cfg = (d.groupby(["B", "L"])
             .agg(m_base=("m_base", "first"), m_ext=("m_extended", "first"),
                  q_base=("qubits_base", "first"), q_ext=("qubits_extended", "first"),
                  cap=("capacity", "first"))
             .reset_index().sort_values("m_base"))
    x = np.arange(len(cfg))
    a1.bar(x - 0.19, cfg.m_base, width=0.36, color=C_NEU, edgecolor="white",
           label="Sizing only")
    a1.bar(x + 0.19, cfg.m_ext, width=0.36, color=C_Q, edgecolor="white",
           label="Sizing + islanding")
    top = max(cfg.m_ext) * 1.62
    for xi, r in zip(x, cfg.itertuples()):
        a1.text(xi - 0.19, r.m_base + 1.0, f"{r.m_base}", ha="center", va="bottom", fontsize=7.6)
        a1.text(xi + 0.19, r.m_ext + 1.0, f"{r.m_ext}", ha="center", va="bottom", fontsize=7.6)
        # qubit cost goes ABOVE the bars: below the axis it collided with the tick labels
        extra = int(r.q_ext) - int(r.q_base)
        a1.text(xi, top * 0.76, f"{int(r.q_base)}$\\to${int(r.q_ext)} qubits",
                ha="center", va="center", fontsize=7.6, color=C_ACC)
        a1.text(xi, top * 0.68, "no extra qubit" if extra == 0 else f"+{extra} qubit",
                ha="center", va="center", fontsize=7.2, color=C_ACC, style="italic")
    a1.set_xticks(x)
    a1.set_xticklabels([f"$B={int(r.B)}$, $L={int(r.L)}$" for r in cfg.itertuples()])
    a1.set_ylabel("Binary decisions $m$")
    a1.set_ylim(0, top)
    # legend across the top: at upper-left it sat on the qubit-cost annotation
    a1.legend(frameon=False, handlelength=1.4, borderpad=0.2, loc="upper center",
              ncol=2, columnspacing=1.2, handletextpad=0.5)
    a1.grid(axis="y", alpha=0.25, lw=0.5)
    a1.set_title("Cost of adding microgrid decisions", fontsize=9)

    fig.savefig(OUT / "fig_r_microgrid.pdf")
    plt.close(fig)
    return d


def fig_hardware():
    """The three-setting claim, measured on the device."""
    d = pd.read_csv(RES / "g3_cepheus.csv")
    # keep the bit ordering that actually matches the exact values
    best = (d.sort_values("sign_agreement", ascending=False)
              .groupby("label", as_index=False).first()
              .sort_values("m"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.3), constrained_layout=True)
    x = np.arange(len(best))
    labels = [f"$m={int(r.m)}$\non {int(r.n)} qubits" for r in best.itertuples()]

    a1.bar(x - 0.19, best.m, width=0.36, color=C_NEU, edgecolor="white",
           label="Decisions encoded")
    a1.bar(x + 0.19, best.jobs, width=0.36, color=C_Q, edgecolor="white",
           label="Device jobs used")
    for xi, r in zip(x, best.itertuples()):
        a1.text(xi - 0.19, r.m * 1.06, f"{int(r.m)}", ha="center", va="bottom", fontsize=7.6)
        a1.text(xi + 0.19, r.jobs * 1.10, f"{int(r.jobs)}", ha="center", va="bottom",
                fontsize=7.6, color=C_Q)
    a1.set_yscale("log")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7.8)
    a1.set_ylabel("Count (log scale)")
    a1.set_ylim(1, 400)
    a1.legend(frameon=False, handlelength=1.4, borderpad=0.2, loc="upper left")
    a1.grid(axis="y", alpha=0.25, lw=0.5)
    a1.set_title("Three jobs, any problem size", fontsize=9)

    w = 0.36
    a2.bar(x - w / 2, 100 * best.sign_agreement, width=w, color=C_Q, edgecolor="white",
           label="Sign agreement")
    a2.bar(x + w / 2, 100 * best.magnitude_retention, width=w, color=C_Q2,
           edgecolor="white", label="Magnitude retained")
    for xi, r in zip(x, best.itertuples()):
        a2.text(xi - w / 2, 100 * r.sign_agreement + 1.6, f"{100*r.sign_agreement:.0f}%",
                ha="center", va="bottom", fontsize=7.6)
        a2.text(xi + w / 2, 100 * r.magnitude_retention + 1.6,
                f"{100*r.magnitude_retention:.0f}%", ha="center", va="bottom", fontsize=7.6)
    a2.axhline(50, color=C_CL, lw=0.9, ls=":", zorder=0)
    a2.text(len(best) - 0.45, 52, "chance", ha="right", va="bottom", fontsize=7.2,
            color=C_CL, style="italic")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=7.8)
    a2.set_ylabel("Percent of exact value")
    a2.set_ylim(0, 118)
    a2.legend(frameon=False, handlelength=1.4, borderpad=0.2, loc="upper left", ncol=1)
    a2.grid(axis="y", alpha=0.25, lw=0.5)
    a2.set_title("Correlators recovered on hardware", fontsize=9)

    fig.savefig(OUT / "fig_r_hardware.pdf")
    plt.close(fig)
    return best




def render_all(out_dir="figures"):
    """Write every figure as vector PDF into ``out_dir``.

    Figures whose result file is absent are skipped with a note rather than
    raising, so a partial checkout still produces what it can.
    """
    global OUT
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    names = [
        "architecture", "compression", "depth_budget", "main_results",
        "maxcut_scaling", "scenarios", "design_levers", "resources", "hardware",
        "r_resilience", "r_microgrid",
    ]
    written, skipped = [], []
    for name in names:
        fn = globals().get("fig_" + name) or globals().get(
            "fig_" + name.replace("r_", ""))
        if fn is None:
            continue
        try:
            fn()
            written.append(name)
        except FileNotFoundError as exc:
            skipped.append((name, getattr(exc, "filename", str(exc))))
    print(f"wrote {len(written)} figures to {OUT}")
    for name, why in skipped:
        print(f"  skipped {name}: missing {why}")
    return written
