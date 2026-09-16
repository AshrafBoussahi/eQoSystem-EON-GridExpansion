"""E9 — powered, correctly baselined confirmation: does a learned edit policy accelerate the search?

Fixes the three defects of the first attempt (see docs/01_engineering_decisions.md, D21):

1. **Correct baseline.** ``es_1_2`` is the unrestricted evolutionary search (1–2 token mutations),
   which is what all earlier results used. ``es_single`` is the operator control (single-token
   mutations only) so the reader can separate "restricting the operator" from "learning the operator".
2. **Honest budget accounting.** The headline learned arms never use data from the instance they
   solve. ``es_learned`` starts from a randomly initialised scorer and learns only from circuits the
   search itself evaluates (no extra quantum cost). ``es_transfer_dc`` is pretrained on the complete
   mutation neighbourhood of a *different* instance, with the correlator-change head that was the
   only configuration to generalise across instances; that pretraining cost is paid once and
   amortised, and is reported.
3. **Power.** ``--instances`` × ``--runs`` paired seeds per arm (default 6 × 5 = 30 pairs), all arms
   sharing the same seeds, so every comparison is paired by (instance, seed).

Primary endpoint: circuit executions to reach an approximation ratio of 0.85 (paired, censored at the
budget). Secondary: final raw ratio, ratio decoded from 1000 shots. Reported with paired t-tests and
Wilcoxon signed-rank tests.

Example::

    python -u experiments/e9_confirmation.py --instances reg3:0 reg3:1 er4:0 --runs 5
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
from e1_budget_curves import shot_decoded  # noqa: E402

from genpce.pool import native_chain_pool  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train import EvolutionConfig, EvolutionarySearch, LearnedMutationProposal, MutationDataset, MutationScorer, Reward, ScorerConfig, train_scorer  # noqa: E402

THRESHOLDS = (0.75, 0.8, 0.85, 0.9)
ARMS = ("es_1_2", "es_single", "es_learned", "es_transfer_dc")


def parse_instance(spec: str) -> tuple[str, int]:
    fam, seed = spec.split(":")
    return fam, int(seed)


def pretrained_dc_scorer(path: Path, vocab: int, seq_len: int, m: int, seed: int) -> tuple[MutationScorer, dict]:
    """Scorer pretrained on another instance's complete neighbourhood, with the Δc head (E7 finding)."""
    data = MutationDataset.load(str(path))
    cfg = ScorerConfig(loss="mse", predict_dc=True, dc_weight=1.0, seed=seed)
    model = MutationScorer(vocab, seq_len, cfg, m=m)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.P)
    n_val = max(8, data.P // 5)
    info = train_scorer(model, data, cfg, train_idx=perm[n_val:], val_idx=perm[:n_val])
    info["pretrain_evaluations"] = int(data.P * data.N * (data.V - 1))
    return model, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--instances", nargs="+", default=["reg3:0", "reg3:1", "reg3:2", "reg3:3", "er4:0", "er4:1"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--relaxed-weight", type=float, default=0.5)
    ap.add_argument("--learned-fraction", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--update-every", type=int, default=3)
    ap.add_argument("--transfer-source", default="reg3:0", help="instance whose neighbourhood dataset pretrains the transfer arm")
    ap.add_argument("--transfer-source-alt", default="reg3:1", help="used when the target is the primary source")
    ap.add_argument("--e7-dir", type=Path, default=RESULTS / "e7")
    ap.add_argument("--e7-tag", default="m60")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e9")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = args.out_dir / f"curves_{args.tag}.csv"
    final_path = args.out_dir / f"final_{args.tag}.csv"
    done = set()
    if final_path.exists():
        prev = pd.read_csv(final_path)
        done = set(zip(prev.instance, prev.arm, prev.run))

    for spec in args.instances:
        family, iseed = parse_instance(spec)
        inst = benchmark_instance(family, args.m, iseed)
        cset = cubic_cset(inst.m)
        bk = inst.best_known()
        pool = native_chain_pool(cset.n)
        evaluator = StatevectorEvaluator(cset)
        seq_len = int(round(args.seq_mult * cset.n))
        reward = Reward.shaped(inst, cset, args.relaxed_weight)
        src_spec = args.transfer_source if spec != args.transfer_source else args.transfer_source_alt
        src_fam, src_seed = parse_instance(src_spec)
        src_inst = benchmark_instance(src_fam, args.m, src_seed)

        for arm in args.arms:
            for run in range(args.runs):
                if (inst.name, arm, run) in done:
                    continue
                prop, pre_evals, pre_val = None, 0, float("nan")
                if arm == "es_1_2":
                    cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, seed=run, max_mutations=2, single_token_mutations=False, log_every=5, record_database=False)
                elif arm == "es_single":
                    cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, seed=run, single_token_mutations=True, log_every=5, record_database=False)
                else:
                    if arm == "es_learned":
                        model = MutationScorer(pool.size, seq_len, ScorerConfig(seed=run))
                    elif arm == "es_transfer_dc":
                        model, info = pretrained_dc_scorer(args.e7_dir / f"mutations_{src_inst.name}_{args.e7_tag}.npz", pool.size, seq_len, inst.m, run)
                        pre_evals, pre_val = info["pretrain_evaluations"], info["best_val"]
                    else:
                        raise ValueError(arm)
                    prop = LearnedMutationProposal(model, epsilon=args.epsilon, seed=run)
                    cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, seed=run, learned_fraction=args.learned_fraction,
                                          single_token_mutations=True, log_every=5, record_database=False, learned_update_every=args.update_every)
                es = EvolutionarySearch(inst, cset, pool, evaluator, reward, cfg, mutation_proposal=prop)
                marks = {t: None for t in THRESHOLDS}

                def cb(rec, marks=marks):
                    for t in THRESHOLDS:
                        if marks[t] is None and rec.get("ratio_best", 0) >= t:
                            marks[t] = rec["evaluations"]

                df = es.run(callback=cb)
                df.insert(0, "run", run); df.insert(0, "arm", arm); df.insert(0, "instance", inst.name); df.insert(0, "family", family)
                df.to_csv(curves_path, mode="a", header=not curves_path.exists(), index=False)
                cut, cut_ls = es.best_with_local_search()
                sd = shot_decoded(inst, cset, pool.to_circuit(es.best["tokens"]))
                row = {"family": family, "instance": inst.name, "m": inst.m, "n": cset.n, "arm": arm, "run": run, "best_known": bk,
                       "ratio_raw": cut / bk, "ratio_ls": cut_ls / bk, "ratio_1000shots": sd[1000][0] / bk,
                       "median_abs_corr": float(np.median(np.abs(es.best["corr"][: inst.m]))),
                       **{f"evals_to_{t}": marks[t] for t in THRESHOLDS},
                       "transfer_source": src_inst.name if arm == "es_transfer_dc" else "", "pretrain_evaluations": pre_evals,
                       "pretrain_val_nll": pre_val, "seconds": df.seconds.iloc[-1]}
                pd.DataFrame([row]).to_csv(final_path, mode="a", header=not final_path.exists(), index=False)
                print(f"{inst.name:14s} {arm:15s} run={run} raw={cut/bk:.3f} 1000sh={sd[1000][0]/bk:.3f} "
                      f"to0.85={marks[0.85]} to0.9={marks[0.9]} {df.seconds.iloc[-1]:.0f}s", flush=True)

    analyse(final_path, args.budget)


def analyse(final_path: Path, budget: int) -> None:
    """Paired comparison of every arm against the unrestricted search."""
    from scipy import stats

    f = pd.read_csv(final_path)
    f["e85"] = f["evals_to_0.85"].fillna(budget)  # censored at the budget (conservative)
    f["reached85"] = f["evals_to_0.85"].notna()
    f["reached90"] = f["evals_to_0.9"].notna()
    print("\n=== per-arm summary (all runs)")
    summ = f.groupby("arm").agg(n=("run", "size"), raw_mean=("ratio_raw", "mean"), raw_sd=("ratio_raw", "std"), raw_median=("ratio_raw", "median"),
                                shots=("ratio_1000shots", "mean"), reach85=("reached85", "mean"), reach90=("reached90", "mean"),
                                e85_median=("e85", "median"), secs=("seconds", "mean"))
    print(summ.round(3).to_string())
    base = "es_1_2"
    if base not in set(f.arm):
        return
    print(f"\n=== paired against {base} (pairs = instance × seed)")
    b = f[f.arm == base].set_index(["instance", "run"])
    rows = []
    for arm in sorted(set(f.arm) - {base}):
        a = f[f.arm == arm].set_index(["instance", "run"])
        idx = a.index.intersection(b.index)
        if len(idx) < 3:
            continue
        d_raw = (a.loc[idx].ratio_raw - b.loc[idx].ratio_raw).values
        d_e85 = (np.log10(a.loc[idx].e85) - np.log10(b.loc[idx].e85)).values
        t_raw, p_raw = stats.ttest_rel(a.loc[idx].ratio_raw, b.loc[idx].ratio_raw)
        try:
            w_raw = stats.wilcoxon(d_raw).pvalue
        except ValueError:
            w_raw = float("nan")
        t_e, p_e = stats.ttest_rel(np.log10(a.loc[idx].e85), np.log10(b.loc[idx].e85))
        rows.append({"arm": arm, "pairs": len(idx), "d_raw_mean": d_raw.mean(), "d_raw_sd": d_raw.std(ddof=1),
                     "t": t_raw, "p_ttest": p_raw, "p_wilcoxon": w_raw, "wins": int((d_raw > 0).sum()), "ties": int((d_raw == 0).sum()),
                     "speedup_to_0.85": 10 ** (-d_e.mean()) if (d_e := d_e85) is not None else np.nan, "p_e85": p_e})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n(speedup > 1 means the arm reaches 0.85 in fewer circuit executions than the baseline)")
    pd.DataFrame(rows).to_csv(final_path.with_name(final_path.stem + "_paired.csv"), index=False)
    summ.to_csv(final_path.with_name(final_path.stem + "_summary.csv"))


if __name__ == "__main__":
    main()
