"""E0-A — PCE-VQA baseline (Sciorilli et al. protocol) on the benchmark instances.

For every (family, m, instance seed) and every brickwork depth we train the relaxed-loss VQA from
several random initialisations with Adam and Sciorilli's stopping rule, and record the raw and
post-processed (one bit-swap pass) approximation ratios together with the quantum-resource proxies
(steps, parameter-shift-equivalent circuit executions, two-qubit gates).

Resumable: rows already present in the output CSV are skipped.

Example::

    python -u experiments/e0_pce_vqa.py --families reg3 er4 --sizes 60 105 168 252 --seeds 0 1 2 \
        --layers 2 4 6 8 12 --inits 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, benchmark_instance, cubic_cset  # noqa: E402

from genpce.baselines import BrickworkMSAnsatz, run_pce_vqa  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["reg3", "er4"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[60, 105, 168, 252])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--layers", nargs="+", type=int, default=[2, 4, 6, 8, 12])
    ap.add_argument("--inits", type=int, default=3)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "slsqp", "cobyla"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RESULTS / "e0" / "pce_vqa.csv")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if args.out.exists():
        prev = pd.read_csv(args.out)
        done = set(zip(prev.instance, prev.layers, prev.init, prev.optimizer))

    for family in args.families:
        for m in args.sizes:
            for seed in args.seeds:
                inst = benchmark_instance(family, m, seed)
                cset = cubic_cset(inst.m)
                bk = inst.best_known()
                for layers in args.layers:
                    ansatz = BrickworkMSAnsatz(cset.n, layers)
                    for init in range(args.inits):
                        key = (inst.name, layers, init, args.optimizer)
                        if key in done:
                            continue
                        res = run_pce_vqa(
                            inst, cset, ansatz, optimizer=args.optimizer, lr=args.lr,
                            max_steps=args.max_steps, seed=1000 * init + seed,
                        )
                        row = {
                            "family": family, "m": inst.m, "n": cset.n, "instance": inst.name,
                            "best_known": bk, "best_known_exact": inst.meta.get("best_known_exact"),
                            "layers": layers, "two_qubit": ansatz.num_two_qubit_gates,
                            "one_qubit": ansatz.num_one_qubit_gates, "num_params": ansatz.num_params,
                            "init": init, "optimizer": args.optimizer, "lr": args.lr,
                            "steps": res.steps, "function_evals": res.function_evals,
                            "circuit_equiv": res.circuit_executions_equiv,
                            "cut_raw": res.cut, "cut_ls": res.cut_ls,
                            "ratio_raw": res.cut / bk, "ratio_ls": res.cut_ls / bk,
                            "final_loss": res.loss_history[-1], "seconds": res.seconds,
                        }
                        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
                        print(
                            f"{inst.name:16s} L={layers:2d} 2q={ansatz.num_two_qubit_gates:3d} init={init} "
                            f"steps={res.steps:4d} r_raw={row['ratio_raw']:.3f} r_ls={row['ratio_ls']:.3f} "
                            f"{res.seconds:5.0f}s",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
