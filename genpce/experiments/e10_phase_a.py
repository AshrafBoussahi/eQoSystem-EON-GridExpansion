"""E10 — Phase A of target-correlation-conditioned circuit synthesis: corpus and the two feasibility tests.

Stages (each cached under ``results/e10/``; rerun with ``--stage`` to redo one):

``corpus``   Variational teacher trajectories (Sciorilli brickwork ansatz, Adam on the relaxed loss)
             for several depths and seeds; every checkpoint's correlator vector, cut and loss.
``ceiling``  Representability ceiling: for each teacher target ``c*`` and each discrete depth ``L``,
             an evolutionary search over the hardware-native token space that *minimises the
             correlation distance* to ``c*`` (no reward, no optimum). Its best circuit bounds what any
             generator can achieve at that depth. The search databases (subsampled) plus random
             circuits form the ``(c(U), U)`` pair set for the two learning tests.
``continuous`` Structural ceiling: the same CZ-brickwork template with an *arbitrary* single-qubit gate
             in every slot (a strict superset of the token family), optimised by gradient descent on the
             correlation distance. Its ``D_c`` lower-bounds what any discrete circuit of that depth can
             reach, so ``search`` (achievable) and ``continuous`` (structural) bracket the true ceiling.
``refine``   Where the search-vs-structure gap comes from: the best searched circuit's angles are made
             continuous (axes fixed) and refined by gradient descent, then snapped back to the grid.
             refined ≈ 0 and snapped ≫ 0 means the angle grid is the limit; refined ≫ 0 means one
             rotation per slot is; refined ≈ searched means the search already found the optimum.
``forward``  Learnability of ``U → c``: transformer encoder regressing all ``m`` correlators, trained
             on pairs from the training targets, tested on pairs from held-out targets.
``inverse``  Learnability of ``c* → U``: conditional decoder trained by cross-entropy on the same
             pairs (conditioning on the pair's own ``c(U)``), queried with *held-out teacher targets*
             it has never seen. Every sample is executed; we report ``D_c``, sign agreement, decoded
             cut ratio, 1000-shot ratio and margin for best-of-``K`` selection, against random
             circuits, nearest training neighbour (memorisation control) and the search ceiling.
             Also ``model+filter``: 1000 samples ranked by the forward model, only the top 100 executed.

Example::

    python -u experiments/e10_phase_a.py --instance reg3:0 --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, benchmark_instance, cubic_cset  # noqa: E402
from e1_budget_curves import shot_decoded  # noqa: E402

from genpce.pce.correlators import decode_signs  # noqa: E402
from genpce.pool import native_chain_pool  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train import EvolutionConfig, EvolutionarySearch  # noqa: E402
from genpce.train.distill import (  # noqa: E402
    ConditionalGenerator,
    CorrelationTargetReward,
    CorrelatorPredictor,
    VQACorpus,
    build_vqa_corpus,
    continuous_ceiling,
    reconstruction_metrics,
    refine_and_snap,
    train_forward,
    train_inverse,
)

OUT = RESULTS / "e10"


# ---------------------------------------------------------------------- targets
def target_table(corpus: VQACorpus, holdout_seeds: set[int], seeds: list[int], intermediate_seeds: set[int], quantiles=(0.25, 0.5)) -> list[dict]:
    """Final checkpoint of every trajectory (+ intermediate checkpoints of a few) as targets."""
    targets = []
    for rid in np.unique(corpus.run):
        idx = np.flatnonzero(corpus.run == rid)
        L, seed = int(corpus.layers[idx[0]]), seeds[int(rid) % len(seeds)]
        hold = seed in holdout_seeds
        picks = [(idx[-1], "final")]
        if seed in intermediate_seeds:
            picks += [(idx[min(len(idx) - 1, int(round(q * (len(idx) - 1))))], f"q{int(q*100)}") for q in quantiles]
        for i, tag in picks:
            targets.append({"target": f"T{L}_s{seed}_{tag}", "teacher_layers": L, "seed": seed, "stage": tag, "holdout": hold,
                            "corpus_index": int(i), "target_cut": float(corpus.cut[i]), "target_loss": float(corpus.loss[i]), "step": int(corpus.step[i])})
    return targets


# ---------------------------------------------------------------------- stages
def stage_corpus(args, inst, cset) -> VQACorpus:
    path = OUT / f"corpus_{inst.name}.npz"
    if path.exists() and args.stage != "corpus":
        return VQACorpus.load(str(path))
    t0 = perf_counter()
    corpus = build_vqa_corpus(inst, cset, layers_list=args.teacher_layers, seeds=range(args.seeds), lr=args.lr, max_steps=args.max_steps, every=args.every, verbose=True)
    corpus.save(str(path))
    bk = inst.best_known()
    fin = corpus.finals()
    summary = pd.DataFrame({"run": corpus.run[fin], "layers": corpus.layers[fin], "steps": corpus.step[fin], "cut": corpus.cut[fin], "ratio": corpus.cut[fin] / bk,
                            "loss": corpus.loss[fin], "median_abs_corr": [float(np.median(np.abs(corpus.corr[i][: inst.m]))) for i in fin]})
    summary.to_csv(OUT / f"corpus_{inst.name}_summary.csv", index=False)
    print(f"corpus: {len(corpus.cut)} checkpoints from {len(fin)} trajectories in {perf_counter()-t0:.0f}s")
    print(summary.groupby("layers")[["ratio", "median_abs_corr", "steps"]].agg(["mean", "min", "max"]).round(3).to_string())
    return corpus


def stage_ceiling(args, inst, cset, pool, evaluator, corpus, targets, L) -> pd.DataFrame:
    """Correlation-distance search per target at discrete depth ``L``; also builds the pair set."""
    ceil_path, pairs_path = OUT / f"ceiling_{inst.name}_L{L}.csv", OUT / f"pairs_{inst.name}_L{L}.npz"
    if ceil_path.exists() and pairs_path.exists() and args.stage != "ceiling":
        return pd.read_csv(ceil_path)
    seq_len = L * cset.n
    bk = inst.best_known()
    rows, tok_all, corr_all, tid_all, hold_all = [], [], [], [], []
    rng = np.random.default_rng(0)
    # random circuits: the trivial baseline for every metric, and part of the training pairs
    rand = rng.integers(0, pool.size, size=(args.random_pairs, seq_len))
    corr_rand = evaluator.evaluate(pool.to_circuits(rand))
    tok_all.append(rand); corr_all.append(corr_rand.astype(np.float32)); tid_all += ["random"] * len(rand); hold_all += [False] * len(rand)
    for t in targets:
        c_star = corpus.corr[t["corpus_index"]]
        reward = CorrelationTargetReward.for_target(inst, cset, c_star)
        cfg = EvolutionConfig(seq_len=seq_len, budget=args.budget, seed=t["seed"], max_mutations=2, log_every=10, record_database=True)
        es = EvolutionarySearch(inst, cset, pool, evaluator, reward, cfg)
        t0 = perf_counter()
        df = es.run()
        best = es.best
        met = reconstruction_metrics(best["corr"], c_star, inst)
        # random-circuit reference for the same target
        rm = [reconstruction_metrics(c, c_star, inst) for c in corr_rand[:200]]
        sd = shot_decoded(inst, cset, pool.to_circuit(best["tokens"]), shots_list=(1000,), reps=20)
        counts = pool.gate_counts(best["tokens"])
        # the search trajectory: D_c of the best circuit versus evaluations (for the figure)
        curve = [(int(r.evaluations), float(-r.best_score)) for r in df.itertuples()]
        rows.append({**t, "discrete_layers": L, "seq_len": seq_len, "budget": args.budget, "evaluations_at_best": best["evaluations"],
                     **met, "ratio_1000shots": sd[1000][0] / bk, "two_qubit": counts["two_qubit"], "non_clifford": counts["non_clifford"],
                     "random_D_c_mean": float(np.mean([r["D_c"] for r in rm])), "random_D_c_min": float(np.min([r["D_c"] for r in rm])),
                     "random_A_x_mean": float(np.mean([r["A_x"] for r in rm])), "D_c_at_1000": float(np.interp(1000, *zip(*curve))),
                     "D_c_at_5000": float(np.interp(5000, *zip(*curve))), "seconds": float(df.seconds.iloc[-1])})
        print(f"  L={L} {t['target']:16s} hold={t['holdout']!s:5s} target_r={t['target_cut']/bk:.3f} | D_c={met['D_c']:.4f} (rand {rows[-1]['random_D_c_mean']:.4f}) "
              f"A_x={met['A_x']:.3f} r={met['ratio']:.3f} 1000sh={rows[-1]['ratio_1000shots']:.3f} |c|={met['median_abs_corr']:.3f} {perf_counter()-t0:.0f}s", flush=True)
        # subsample the search database for the pair set: all archive-quality circuits + uniform sample
        db_tok, db_corr = np.stack(es.database.tokens), np.stack(es.database.corr)
        score, _ = reward(db_corr.astype(np.float64))
        top = np.argsort(score)[::-1][: args.pairs_top]
        uni = rng.choice(len(db_tok), size=min(args.pairs_uniform, len(db_tok)), replace=False)
        keep = np.unique(np.concatenate([top, uni]))
        tok_all.append(db_tok[keep]); corr_all.append(db_corr[keep].astype(np.float32)); tid_all += [t["target"]] * len(keep); hold_all += [t["holdout"]] * len(keep)
        pd.DataFrame(rows).to_csv(ceil_path, index=False)
    np.savez_compressed(pairs_path, tokens=np.concatenate(tok_all), corr=np.concatenate(corr_all), target=np.array(tid_all), holdout=np.array(hold_all))
    return pd.DataFrame(rows)


def stage_continuous(args, inst, cset, corpus, targets, L) -> pd.DataFrame:
    path = OUT / f"continuous_{inst.name}_L{L}.csv"
    if path.exists() and args.stage != "continuous":
        return pd.read_csv(path)
    bk = inst.best_known()
    rows = []
    for t in targets:
        c_star = corpus.corr[t["corpus_index"]]
        t0 = perf_counter()
        best = continuous_ceiling(cset, L, c_star, restarts=args.cont_restarts, steps=args.cont_steps, seed=t["seed"])
        met = reconstruction_metrics(best["corr"], c_star, inst)
        rows.append({**t, "discrete_layers": L, "restarts": args.cont_restarts, "steps": args.cont_steps, **met, "seconds": perf_counter() - t0})
        print(f"  continuous L={L} {t['target']:16s} target_r={t['target_cut']/bk:.3f} | D_c={met['D_c']:.4f} A_x={met['A_x']:.3f} r={met['ratio']:.3f} |c|={met['median_abs_corr']:.3f} {rows[-1]['seconds']:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def stage_refine(args, inst, cset, pool, evaluator, corpus, targets, L) -> pd.DataFrame:
    path = OUT / f"refine_{inst.name}_L{L}.csv"
    if path.exists() and args.stage != "refine":
        return pd.read_csv(path)
    tok, corr, tid, _ = load_pairs(inst, L)
    bk = inst.best_known()
    rows = []
    for t in targets:
        c_star = corpus.corr[t["corpus_index"]].astype(np.float64)
        sel = np.flatnonzero(tid == t["target"])
        if not len(sel):
            continue
        d = np.mean((corr[sel][:, : inst.m].astype(np.float64) - c_star[: inst.m]) ** 2, axis=1)
        best = tok[sel[int(np.argmin(d))]]
        t0 = perf_counter()
        out = refine_and_snap(pool, cset, best, c_star, steps=args.refine_steps)
        c_snap = evaluator.evaluate([pool.to_circuit(out["snapped_tokens"])])[0]
        met_s, met_r = reconstruction_metrics(c_snap, c_star, inst), reconstruction_metrics(out["corr_refined"], c_star, inst)
        rows.append({**t, "discrete_layers": L, "D_c_search": float(d.min()), "D_c_refined": out["D_c_refined"], "D_c_snapped": met_s["D_c"],
                     "A_x_refined": met_r["A_x"], "ratio_refined": met_r["ratio"], "A_x_snapped": met_s["A_x"], "ratio_snapped": met_s["ratio"],
                     "mean_abs_angle_move": out["mean_abs_angle_move"], "seconds": perf_counter() - t0})
        print(f"  refine L={L} {t['target']:16s} search={d.min():.4f} -> refined={out['D_c_refined']:.4f} (A_x {met_r['A_x']:.3f} r {met_r['ratio']:.3f}) "
              f"-> snapped={met_s['D_c']:.4f} (A_x {met_s['A_x']:.3f} r {met_s['ratio']:.3f}) move={out['mean_abs_angle_move']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def load_pairs(inst, L):
    z = np.load(OUT / f"pairs_{inst.name}_L{L}.npz")
    return z["tokens"], z["corr"], z["target"], z["holdout"]


def stage_forward(args, inst, cset, pool, L) -> tuple[CorrelatorPredictor, dict]:
    path = OUT / f"forward_{inst.name}_L{L}.pt"
    tok, corr, tid, hold = load_pairs(inst, L)
    model = CorrelatorPredictor(pool.size, L * cset.n, cset.m, d_model=args.d_model, n_layers=args.n_layers)
    if path.exists() and args.stage not in ("forward",):
        model.load_state_dict(torch.load(path)); model.eval()
        return model, json.loads((OUT / f"forward_{inst.name}_L{L}.json").read_text())
    rng = np.random.default_rng(0)
    is_rand = tid == "random"
    train = np.flatnonzero(~hold)
    rng.shuffle(train)
    val, train = train[: max(200, len(train) // 10)], train[max(200, len(train) // 10):]
    test_es, test_rand = np.flatnonzero(hold), np.flatnonzero(is_rand)[: 500]
    train = np.setdiff1d(train, test_rand); val = np.setdiff1d(val, test_rand)
    t0 = perf_counter()
    info = train_forward(model, tok, corr, train_idx=train, val_idx=val, max_steps=args.forward_steps, seed=0)
    with torch.no_grad():  # chunked: attention over all pairs at once would need gigabytes
        pred = np.concatenate([model(torch.as_tensor(tok[i : i + 1024], dtype=torch.long)).numpy() for i in range(0, len(tok), 1024)])
    m = inst.m
    bk = inst.best_known()

    def block(idx):
        p, c = pred[idx][:, :m], corr[idx][:, :m].astype(np.float64)
        xp, xc = decode_signs(p), decode_signs(c)
        rp, rc = inst.cut_values(xp) / bk, inst.cut_values(xc) / bk
        base = corr[train][:, :m].mean(0)
        from scipy.stats import spearmanr
        sp = np.mean([spearmanr(p[i], c[i]).correlation for i in range(len(idx))]) if len(idx) else np.nan
        return {"n": int(len(idx)), "mse": float(np.mean((p - c) ** 2)), "mse_mean_baseline": float(np.mean((base - c) ** 2)),
                "sign_acc": float(np.mean(xp == xc)), "spearman_per_circuit": float(sp), "ratio_true_mean": float(rc.mean()),
                "ratio_pred_signs_mean": float(rp.mean()), "ratio_abs_err": float(np.mean(np.abs(rp - rc))), "ratio_corr": float(np.corrcoef(rp, rc)[0, 1]) if len(idx) > 2 else np.nan}

    res = {"discrete_layers": L, "n_train": int(len(train)), "n_val": int(len(val)), "best_val_mse": info["best_val_mse"], "steps": info["steps"], "seconds": perf_counter() - t0,
           "test_holdout_search": block(test_es), "test_random": block(test_rand), "train_fit": block(train[:2000])}
    torch.save(model.state_dict(), path)
    (OUT / f"forward_{inst.name}_L{L}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"forward L={L}: train {len(train)} | held-out search pairs mse={res['test_holdout_search']['mse']:.4f} (mean-baseline {res['test_holdout_search']['mse_mean_baseline']:.4f}) "
          f"sign_acc={res['test_holdout_search']['sign_acc']:.3f} ratio_abs_err={res['test_holdout_search']['ratio_abs_err']:.3f} | random mse={res['test_random']['mse']:.4f} sign_acc={res['test_random']['sign_acc']:.3f} ({res['seconds']:.0f}s)", flush=True)
    return model, res


def best_of_k(mets: list[dict], K: int, rng, key="D_c") -> dict:
    """Expected metrics of the best-by-``key`` circuit among ``K`` samples (averaged over disjoint groups)."""
    idx = rng.permutation(len(mets))
    groups = [idx[i : i + K] for i in range(0, len(idx) - K + 1, K)] or [idx]
    picks = [min(g, key=lambda j: mets[j][key]) for g in groups]
    return {k: float(np.mean([mets[j][k] for j in picks])) for k in mets[0]}


def stage_inverse(args, inst, cset, pool, evaluator, corpus, targets, L, fwd: CorrelatorPredictor, ceiling: pd.DataFrame) -> pd.DataFrame:
    path = OUT / f"inverse_{inst.name}_L{L}.csv"
    tok, corr, tid, hold = load_pairs(inst, L)
    seq_len = L * cset.n
    gen = ConditionalGenerator(pool.size, seq_len, cset.m, d_model=args.d_model, n_layers=args.n_layers)
    gpath = OUT / f"inverse_{inst.name}_L{L}.pt"
    rng = np.random.default_rng(0)
    train = np.flatnonzero(~hold)
    rng.shuffle(train)
    val, train = train[: max(200, len(train) // 10)], train[max(200, len(train) // 10):]
    if gpath.exists() and args.stage not in ("inverse",):
        gen.load_state_dict(torch.load(gpath)); gen.eval()
        info = json.loads((OUT / f"inverse_{inst.name}_L{L}.json").read_text())
    else:
        t0 = perf_counter()
        info = train_inverse(gen, tok, corr, train_idx=train, val_idx=val, max_steps=args.inverse_steps, noise=args.cond_noise, seed=0)
        info["seconds"] = perf_counter() - t0
        info["n_train"] = int(len(train))
        torch.save(gen.state_dict(), gpath)
        (OUT / f"inverse_{inst.name}_L{L}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"inverse L={L}: train {len(train)} pairs, val NLL/token {info['best_val_nll_per_token']:.3f} (uniform {info['uniform_nll_per_token']:.3f}) {info['seconds']:.0f}s", flush=True)

    bk = inst.best_known()
    m = inst.m
    train_tok, train_corr = tok[train], corr[train][:, :m].astype(np.float64)
    rows = []
    g = torch.Generator().manual_seed(0)
    for t in targets:
        if not t["holdout"]:
            continue
        c_star = corpus.corr[t["corpus_index"]].astype(np.float64)
        ct = torch.as_tensor(c_star[: cset.m], dtype=torch.float32)
        base = {**t, "discrete_layers": L}
        ceil_row = ceiling[ceiling.target == t["target"]].iloc[0] if (ceiling.target == t["target"]).any() else None

        def emit(method, K, mets_sel, executed, extra=None):
            rows.append({**base, "method": method, "K": K, "executed": executed, **mets_sel, **(extra or {})})

        # random circuits
        rs = rng.integers(0, pool.size, size=(args.samples, seq_len))
        mets_r = [reconstruction_metrics(c, c_star, inst) for c in evaluator.evaluate(pool.to_circuits(rs))]
        keys = list(mets_r[0])
        for K in args.K:
            emit("random", K, best_of_k(mets_r, K, rng), K)
        # generator samples at each temperature
        for T in args.temperatures:
            seqs = gen.sample(ct, args.samples, temperature=T, generator=g).numpy()
            c_gen = evaluator.evaluate(pool.to_circuits(seqs))
            mets = [reconstruction_metrics(c, c_star, inst) for c in c_gen]
            # distance to the training set (memorisation check)
            ham = np.array([np.min(np.sum(train_tok != s, axis=1)) for s in seqs[:50]])
            uniq = len({s.tobytes() for s in seqs})
            for K in args.K:
                extra = {"temperature": T, "min_hamming_to_train": float(ham.mean()), "unique_fraction": uniq / len(seqs)}
                sel = best_of_k(mets, K, rng)
                if K == max(args.K):
                    j = int(np.argmin([mm["D_c"] for mm in mets]))
                    sd = shot_decoded(inst, cset, pool.to_circuit(seqs[j]), shots_list=(1000,), reps=20)
                    extra["ratio_1000shots"] = sd[1000][0] / bk
                    extra["max_ratio_any_sample"] = float(max(mm["ratio"] for mm in mets))
                emit(f"generator_T{T}", K, sel, K, extra)
            # forward-model filter: sample more, execute only the predicted-best
            if fwd is not None and T == args.temperatures[0]:
                big = gen.sample(ct, args.filter_pool, temperature=T, generator=g).numpy()
                with torch.no_grad():
                    pred = fwd(torch.as_tensor(big, dtype=torch.long)).numpy()[:, :m]
                pd_c = np.mean((pred - c_star[:m]) ** 2, axis=1)
                top = np.argsort(pd_c)[: args.samples]
                c_top = evaluator.evaluate(pool.to_circuits(big[top]))
                mets_f = [reconstruction_metrics(c, c_star, inst) for c in c_top]
                for K in args.K:
                    emit(f"generator+filter_T{T}", K, best_of_k(mets_f, K, rng), K, {"temperature": T, "sampled": args.filter_pool,
                         "filter_spearman": float(pd.Series(pd_c[top]).corr(pd.Series([mm["D_c"] for mm in mets_f]), method="spearman"))})
        # nearest training neighbour by stored correlators (no execution needed: memorisation control)
        d = np.mean((train_corr - c_star[:m]) ** 2, axis=1)
        for K in args.K:
            nn_idx = np.argsort(d)[:K]
            mets_n = [reconstruction_metrics(train_corr[i], c_star, inst) for i in nn_idx]
            emit("nearest_train", K, min(mets_n, key=lambda mm: mm["D_c"]), 0)
        # search ceiling at this depth
        if ceil_row is not None:
            emit("search_ceiling", 0, {k: float(ceil_row[k]) for k in keys}, int(ceil_row.budget), {"ratio_1000shots": float(ceil_row.ratio_1000shots)})
        pd.DataFrame(rows).to_csv(path, index=False)
        summ = pd.DataFrame(rows)
        summ = summ[(summ.target == t["target"]) & summ.K.isin([1, max(args.K), 0])]
        print(f"  L={L} {t['target']:16s} target_r={t['target_cut']/bk:.3f}", flush=True)
        for r in summ.itertuples():
            print(f"      {r.method:22s} K={r.K:3d} D_c={r.D_c:.4f} A_x={r.A_x:.3f} r={r.ratio:.3f} |c|={r.median_abs_corr:.3f}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main
def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="reg3:0")
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--stage", default="all", choices=["all", "corpus", "ceiling", "continuous", "refine", "forward", "inverse"])
    ap.add_argument("--teacher-layers", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--holdout", type=int, default=3, help="last seeds of every teacher depth are held out")
    ap.add_argument("--intermediate-seeds", type=int, nargs="*", default=[0, 9])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8], help="discrete circuit depths (brickwork layers)")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--cont-restarts", type=int, default=4)
    ap.add_argument("--cont-steps", type=int, default=400)
    ap.add_argument("--refine-steps", type=int, default=300)
    ap.add_argument("--random-pairs", type=int, default=5000)
    ap.add_argument("--pairs-top", type=int, default=300)
    ap.add_argument("--pairs-uniform", type=int, default=2000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--forward-steps", type=int, default=4000)
    ap.add_argument("--inverse-steps", type=int, default=8000)
    ap.add_argument("--cond-noise", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--filter-pool", type=int, default=1000)
    ap.add_argument("--K", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.7])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)

    fam, iseed = args.instance.split(":")
    inst = benchmark_instance(fam, args.m, int(iseed))
    cset = cubic_cset(inst.m)
    pool = native_chain_pool(cset.n)
    evaluator = StatevectorEvaluator(cset)
    print(f"instance {inst.name}: m={inst.m} n={cset.n} best_known={inst.best_known()} pool={pool.size} tokens", flush=True)

    corpus = stage_corpus(args, inst, cset)
    if args.stage == "corpus":
        return
    seeds = list(range(args.seeds))
    targets = target_table(corpus, set(seeds[-args.holdout:]), seeds, set(args.intermediate_seeds))
    pd.DataFrame(targets).to_csv(OUT / f"targets_{inst.name}.csv", index=False)
    print(f"{len(targets)} targets ({sum(t['holdout'] for t in targets)} held out)", flush=True)

    for L in args.depths:
        if args.stage == "continuous":
            stage_continuous(args, inst, cset, corpus, targets, L)
            continue
        ceiling = stage_ceiling(args, inst, cset, pool, evaluator, corpus, targets, L)
        if args.stage == "ceiling":
            continue
        if args.stage == "refine":
            stage_refine(args, inst, cset, pool, evaluator, corpus, targets, L)
            continue
        fwd, _ = stage_forward(args, inst, cset, pool, L)
        if args.stage == "forward":
            continue
        stage_inverse(args, inst, cset, pool, evaluator, corpus, targets, L, fwd, ceiling)


if __name__ == "__main__":
    main()
