"""E1 — solution quality versus quantum budget: GenPCE-ES vs PCE-VQA vs random circuits.

For each instance we record, as a function of *circuit executions* (each = 3 measurement settings):

* **GenPCE-ES** — elitist evolutionary search in the brickwork token space, shaped reward
  (exact cut + 0.5·relaxed loss), no gradients, no circuit parameters;
* **PCE-VQA** — Sciorilli-style brickwork ansatz trained with Adam (lr 0.01, stopping rule), booked
  at the parameter-shift cost ``2N_p + 1`` circuits per step, for several depths;
* **random circuits** — uniformly random token sequences (no-learning control).

The exact cut is reported *and* the cut decoded from finite-shot estimates (1000 and 4000 shots) of
the final circuit, because near-zero correlators do not survive real measurement.

Example::

    python -u experiments/e1_budget_curves.py --families reg3 er4 --sizes 60 252 --seeds 0 1 --budget 20000
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

from genpce.baselines import BrickworkMSAnsatz, TorchSimulator  # noqa: E402
from genpce.pce import RelaxedLossParams, decode_signs, relaxed_loss_torch  # noqa: E402
from genpce.pool import native_chain_pool  # noqa: E402
from genpce.problems import one_pass_bit_swap  # noqa: E402
from genpce.sim import ShotEvaluator, StatevectorEvaluator  # noqa: E402
from genpce.train import EvolutionConfig, EvolutionarySearch, Reward  # noqa: E402


def shot_decoded(inst, cset, circuit, shots_list=(1000, 4000), reps=20, seed=0):
    out = {}
    for shots in shots_list:
        ev = ShotEvaluator(cset, shots=shots, rng=np.random.default_rng(seed))
        cuts = [inst.cut_value(decode_signs(ev.evaluate([circuit])[0][: inst.m])) for _ in range(reps)]
        out[shots] = (float(np.mean(cuts)), float(np.min(cuts)))
    return out


def run_vqa(inst, cset, layers, seed, lr, max_steps, stop_window=50, stop_tol=0.01):
    """Adam on the relaxed loss; returns (curve rows, final params, final corr)."""
    ansatz = BrickworkMSAnsatz(cset.n, layers)
    sim = TorchSimulator(ansatz, cset)
    lp = RelaxedLossParams.sciorilli(inst, cset.n, cset.k)
    per_step = 2 * ansatz.num_params + 1
    rng = np.random.default_rng(seed)
    theta = torch.tensor(ansatz.random_params(rng), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    checkpoints = sorted(set(int(round(x)) for x in np.geomspace(1, max_steps, 40)))
    rows, history = [], []
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        loss = relaxed_loss_torch(sim.correlators(theta), inst, lp)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        stop = len(history) > stop_window and history[-1 - stop_window] - history[-1] < stop_tol
        if step in checkpoints or stop or step == max_steps:
            with torch.no_grad():
                corr = sim.correlators(theta).numpy()
            rows.append({"step": step, "evaluations": step * per_step, "cut": inst.cut_value(decode_signs(corr[: inst.m]))})
        if stop:
            break
    return rows, ansatz, theta.detach().numpy(), corr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["reg3", "er4"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[60, 252])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--runs", type=int, default=2, help="ES/random seeds per instance")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--relaxed-weight", type=float, default=0.5)
    ap.add_argument("--vqa-layers", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--vqa-inits", type=int, default=2)
    ap.add_argument("--vqa-max-steps", type=int, default=2000)
    ap.add_argument("--no-vqa", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e1")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_path, final_path = args.out_dir / f"curves_{args.tag}.csv", args.out_dir / f"final_{args.tag}.csv"
    done = set()
    if final_path.exists():
        prev = pd.read_csv(final_path)
        done = set(zip(prev.instance, prev.method, prev.run))

    def write(rows, path):
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)

    for family in args.families:
        for m in args.sizes:
            for seed in args.seeds:
                inst = benchmark_instance(family, m, seed)
                cset = cubic_cset(inst.m)
                bk = inst.best_known()
                pool = native_chain_pool(cset.n)
                evaluator = StatevectorEvaluator(cset)
                seq_len = int(round(args.seq_mult * cset.n))
                base = {"family": family, "m": inst.m, "n": cset.n, "instance": inst.name, "best_known": bk}

                for run in range(args.runs):
                    # --- GenPCE-ES
                    key = (inst.name, "genpce_es", run)
                    if key not in done:
                        es = EvolutionarySearch(inst, cset, pool, evaluator, Reward.shaped(inst, cset, args.relaxed_weight),
                                                EvolutionConfig(seq_len=seq_len, budget=args.budget, seed=run))
                        df = es.run()
                        write([{**base, "method": "genpce_es", "run": run, "evaluations": r.evaluations, "ratio": r.ratio_best} for r in df.itertuples()], curves_path)
                        cut, cut_ls = es.best_with_local_search()
                        sd = shot_decoded(inst, cset, pool.to_circuit(es.best["tokens"]))
                        counts = pool.gate_counts(es.best["tokens"])
                        write([{**base, "method": "genpce_es", "run": run, "evaluations": es.evaluations, "ratio_raw": cut / bk, "ratio_ls": cut_ls / bk,
                                "ratio_1000shots": sd[1000][0] / bk, "ratio_4000shots": sd[4000][0] / bk,
                                "median_abs_corr": float(np.median(np.abs(es.best["corr"][: inst.m]))),
                                "two_qubit": counts["two_qubit"], "non_clifford": counts["non_clifford"], "depth": pool.depth(es.best["tokens"]),
                                "seconds": df.seconds.iloc[-1]}], final_path)
                        print(f"{inst.name:16s} ES     run={run} r_raw={cut/bk:.3f} r_ls={cut_ls/bk:.3f} 1000sh={sd[1000][0]/bk:.3f} 4000sh={sd[4000][0]/bk:.3f} med|c|={np.median(np.abs(es.best['corr'][:inst.m])):.3f} {df.seconds.iloc[-1]:.0f}s", flush=True)
                    # --- random control
                    key = (inst.name, "random", run)
                    if key not in done:
                        rng = np.random.default_rng(1000 + run)
                        best, rows, evals = -np.inf, [], 0
                        best_corr = None
                        while evals < args.budget:
                            toks = rng.integers(0, pool.size, size=(50, seq_len))
                            corr = evaluator.evaluate(pool.to_circuits(toks))
                            cuts = inst.cut_values(decode_signs(corr[:, : inst.m]))
                            evals += 50
                            i = int(np.argmax(cuts))
                            if cuts[i] > best:
                                best, best_corr, best_tok = float(cuts[i]), corr[i], toks[i]
                            rows.append({**base, "method": "random", "run": run, "evaluations": evals, "ratio": best / bk})
                        write(rows, curves_path)
                        sd = shot_decoded(inst, cset, pool.to_circuit(best_tok))
                        write([{**base, "method": "random", "run": run, "evaluations": evals, "ratio_raw": best / bk,
                                "ratio_ls": inst.cut_value(one_pass_bit_swap(inst, decode_signs(best_corr[: inst.m]))) / bk,
                                "ratio_1000shots": sd[1000][0] / bk, "ratio_4000shots": sd[4000][0] / bk,
                                "median_abs_corr": float(np.median(np.abs(best_corr[: inst.m])))}], final_path)
                        print(f"{inst.name:16s} RANDOM run={run} r_raw={best/bk:.3f}", flush=True)

                # --- PCE-VQA
                if not args.no_vqa:
                    for layers in args.vqa_layers:
                        for init in range(args.vqa_inits):
                            key = (inst.name, f"pce_vqa_L{layers}", init)
                            if key in done:
                                continue
                            rows, ansatz, params, corr = run_vqa(inst, cset, layers, 1000 * init + seed, 0.01, args.vqa_max_steps)
                            write([{**base, "method": f"pce_vqa_L{layers}", "run": init, "evaluations": r["evaluations"], "ratio": r["cut"] / bk} for r in rows], curves_path)
                            x = decode_signs(corr[: inst.m])
                            cut, cut_ls = inst.cut_value(x), inst.cut_value(one_pass_bit_swap(inst, x))
                            sd = shot_decoded(inst, cset, ansatz.qiskit_circuit(params))
                            write([{**base, "method": f"pce_vqa_L{layers}", "run": init, "evaluations": rows[-1]["evaluations"], "ratio_raw": cut / bk, "ratio_ls": cut_ls / bk,
                                    "ratio_1000shots": sd[1000][0] / bk, "ratio_4000shots": sd[4000][0] / bk,
                                    "median_abs_corr": float(np.median(np.abs(corr[: inst.m]))),
                                    "two_qubit": ansatz.num_two_qubit_gates, "num_params": ansatz.num_params, "steps": rows[-1]["step"]}], final_path)
                            print(f"{inst.name:16s} VQA L={layers} init={init} steps={rows[-1]['step']} evals={rows[-1]['evaluations']} r_raw={cut/bk:.3f} r_ls={cut_ls/bk:.3f} 1000sh={sd[1000][0]/bk:.3f} med|c|={np.median(np.abs(corr[:inst.m])):.3f}", flush=True)


if __name__ == "__main__":
    main()
