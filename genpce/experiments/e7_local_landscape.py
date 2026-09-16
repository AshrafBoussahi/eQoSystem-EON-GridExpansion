"""E7 — Stage 1: is the local circuit landscape learnable? (oracle mutation-learning test)

1. Collect parents of three quality classes (elite / mid / random) from an ES run on a source
   instance; enumerate and evaluate **every** single-token child of every parent (exact ΔR, Δcut, Δc).
2. Landscape statistics: improving-edit fraction, best-edit gain, per-position sensitivity map
   (Experiment 5), and correlator-space geometry (Experiment 10).
3. Learner matrix (Stage 1 B–D): random / MLP / transformer scorers with regression, listwise
   ranking and correlator-prediction targets; held-out parents of the same instance and parents of a
   second, unseen instance. Metrics: Spearman over the 1152 edits, hit@k of the predicted best edit,
   mean true gain of the predicted top-k vs random, improving fraction of the predicted top-k.

Example::

    python -u experiments/e7_local_landscape.py --m 60 --parents 60 60 40 --budget 20000
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
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, benchmark_instance, cubic_cset  # noqa: E402

from genpce.pool import native_chain_pool  # noqa: E402
from genpce.sim import StatevectorEvaluator  # noqa: E402
from genpce.train import EvolutionConfig, EvolutionarySearch, Reward  # noqa: E402
from genpce.train.mutation import MLPScorer, MutationDataset, MutationScorer, ScorerConfig, build_mutation_dataset, evaluate_scorer, rank_metrics, train_scorer  # noqa: E402


def collect_parents(inst, cset, pool, evaluator, reward, *, n_elite, n_mid, n_random, budget, seed, seq_len):
    """Parents from an ES run's database: top (elite), middle quantiles (mid), and uniform random."""
    es = EvolutionarySearch(inst, cset, pool, evaluator, reward, EvolutionConfig(seq_len=seq_len, budget=budget, seed=seed, record_database=True))
    es.run()
    toks = np.stack(es.database.tokens)
    scores, _ = reward(np.stack(es.database.corr).astype(np.float64))
    uniq, first = np.unique(toks, axis=0, return_index=True)
    toks, scores = toks[first], scores[first]
    order = np.argsort(scores)[::-1]
    rng = np.random.default_rng(seed)
    elite = toks[order[:n_elite]]
    lo, hi = int(0.4 * len(order)), int(0.6 * len(order))
    mid = toks[rng.choice(order[lo:hi], size=n_mid, replace=False)]
    rand = rng.integers(0, pool.size, size=(n_random, seq_len))
    parents = np.concatenate([elite, mid, rand])
    cls = np.array(["elite"] * n_elite + ["mid"] * n_mid + ["random"] * n_random)
    return parents, cls, float(scores[order[0]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--family", default="reg3")
    ap.add_argument("--source-seed", type=int, default=0)
    ap.add_argument("--target-seed", type=int, default=1)
    ap.add_argument("--parents", nargs=3, type=int, default=[60, 60, 40], help="elite mid random")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--seq-mult", type=float, default=8.0)
    ap.add_argument("--relaxed-weight", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "e7")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cset = cubic_cset(args.m)
    pool = native_chain_pool(cset.n)
    evaluator = StatevectorEvaluator(cset)
    seq_len = int(round(args.seq_mult * cset.n))
    datasets, classes = {}, {}
    for role, seed in (("source", args.source_seed), ("target", args.target_seed)):
        inst = benchmark_instance(args.family, args.m, seed)
        reward = Reward.shaped(inst, cset, args.relaxed_weight)
        cache = args.out_dir / f"mutations_{inst.name}_{args.tag}.npz"
        if cache.exists():
            data = MutationDataset.load(str(cache))
            cls = np.load(args.out_dir / f"classes_{inst.name}_{args.tag}.npy")
        else:
            parents, cls, es_best = collect_parents(inst, cset, pool, evaluator, reward, n_elite=args.parents[0], n_mid=args.parents[1], n_random=args.parents[2],
                                                    budget=args.budget, seed=seed, seq_len=seq_len)
            print(f"[{role}] {inst.name}: ES best score {es_best:.3f}; enumerating {len(parents)} × {seq_len * (pool.size - 1)} children ...", flush=True)
            data = build_mutation_dataset(parents, pool, evaluator, reward, keep_dc=True)
            data.save(str(cache))
            np.save(args.out_dir / f"classes_{inst.name}_{args.tag}.npy", cls)
        datasets[role], classes[role] = data, cls
        bk = inst.best_known()
        # landscape statistics per class
        for c in ("elite", "mid", "random"):
            sel = np.flatnonzero(cls == c)
            best_gain = np.nanmax(data.dcut[sel].reshape(len(sel), -1), axis=1)
            print(f"[{role}] {c:6s}: parent ratio {np.mean(data.parent_cut[sel])/bk:.3f} | improving edits {np.mean(data.improving_fraction()[sel]):.3f} | "
                  f"best single edit gain {np.mean(best_gain)/bk:+.3f} ratio | edits with Δcut=0: {np.nanmean(data.dcut[sel] == 0):.2f}", flush=True)

    src, tgt = datasets["source"], datasets["target"]
    cls_src, cls_tgt = classes["source"], classes["target"]
    src_inst = benchmark_instance(args.family, args.m, args.source_seed)
    bk = src_inst.best_known()

    # ---------------- sensitivity map (Experiment 5)
    S = src.sensitivity()  # (P, N)
    elite = np.flatnonzero(cls_src == "elite")
    S_el = S[elite].mean(0).reshape(-1, cset.n)  # (layers, qubits)
    Smax = np.nanmax(src.dR, axis=2)[elite].mean(0).reshape(-1, cset.n)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, mat, title in zip(axes, (S_el, Smax), ("mean ΔR over replacements", "best ΔR at position")):
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
        ax.set_xlabel("qubit slot"); ax.set_ylabel("layer"); ax.set_title(f"{title} (elite parents)")
        plt.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(args.out_dir / f"sensitivity_map_{args.tag}.png", dpi=150); plt.close(fig)
    layer_profile = S_el.mean(1)
    print(f"sensitivity by layer (elite parents, mean ΔR): {np.round(layer_profile, 4).tolist()}", flush=True)

    # ---------------- geometry (Experiment 10)
    dc = src.dc.astype(np.float32)
    dcn = np.linalg.norm(dc, axis=-1)
    valid = ~np.isnan(src.dR)
    from scipy.stats import spearmanr
    geo = {"spearman_dcnorm_vs_absdR": float(spearmanr(dcn[valid], np.abs(src.dR[valid])).correlation),
           "spearman_dcnorm_vs_absdcut": float(spearmanr(dcn[valid], np.abs(src.dcut[valid])).correlation),
           "frac_edits_dcut_zero": float(np.nanmean(src.dcut == 0)), "median_dcnorm_single_edit": float(np.median(dcn[valid]))}
    # multi-edit distances around elite parents
    reward = Reward.shaped(src_inst, cset, args.relaxed_weight)
    rng = np.random.default_rng(0)
    rows = []
    for h in (1, 2, 4, 8, 16, 32):
        P = src.parents[elite[rng.integers(0, len(elite), size=150)]].copy()
        for r in range(len(P)):
            pos = rng.choice(seq_len, size=h, replace=False)
            P[r, pos] = (P[r, pos] + rng.integers(1, pool.size, size=h)) % pool.size
        corr = evaluator.evaluate(pool.to_circuits(P))
        sc, cut = reward(corr)
        rows.append({"hamming": h, "mean_abs_dscore": float(np.mean(np.abs(sc - src.parent_score[elite].mean()))), "mean_cut_ratio": float(cut.mean() / bk)})
    geo_h = pd.DataFrame(rows)
    print("geometry:", geo, flush=True)
    print(geo_h.to_string(index=False), flush=True)
    geo_h.to_csv(args.out_dir / f"geometry_hamming_{args.tag}.csv", index=False)

    # ---------------- learner matrix
    rng = np.random.default_rng(0)
    P = src.P
    perm = rng.permutation(P)
    n_val = max(8, P // 5)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    tgt_idx = np.arange(tgt.P)
    results = []

    def report(name, model):
        for split, data, idx, cls in (("held-out parents (source)", src, val_idx, cls_src), ("unseen instance (target)", tgt, tgt_idx, cls_tgt)):
            for c in ("elite", "mid", "random", "all"):
                sel = idx if c == "all" else idx[cls[idx] == c]
                if len(sel) == 0:
                    continue
                met = evaluate_scorer(model, data, sel) if model is not None else random_metrics(data, sel)
                results.append({"model": name, "split": split, "class": c, "n_parents": len(sel), **met})
        r = [x for x in results if x["model"] == name and x["class"] == "all"]
        print(f"  {name:28s} | " + " || ".join(f"{x['split'][:9]}: spearman={x['spearman']:.3f} hit@10={x['hit@10']:.2f} rank={x['rank_of_pred_best']:.0f} top5-gain={x['mean_true_top5']*100:+.2f} vs rand {x['mean_true_random']*100:+.2f} (×100) improving@5={x['improving_top5']:.2f}/{x['improving_random']:.2f}" for x in r), flush=True)

    def random_metrics(data, sel):
        rows = []
        for p in sel:
            true = data.dR[p].reshape(-1)
            pred = rng.random(true.shape)
            rows.append(rank_metrics(pred, true))
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    print("\nLearner matrix (Spearman over all single edits; hit@10 = predicted best edit is a true top-10 edit; gains in reward units ×100):", flush=True)
    report("random", None)
    mlp = MLPScorer(pool.size, seq_len)
    train_scorer(mlp, src, ScorerConfig(loss="mse", lr=3e-4), train_idx=train_idx, val_idx=val_idx)
    report("MLP one-hot, MSE", mlp)
    tr = MutationScorer(pool.size, seq_len, ScorerConfig(loss="mse"))
    train_scorer(tr, src, ScorerConfig(loss="mse"), train_idx=train_idx, val_idx=val_idx)
    report("Transformer, MSE", tr)
    tr_lw = MutationScorer(pool.size, seq_len, ScorerConfig(loss="listwise"))
    train_scorer(tr_lw, src, ScorerConfig(loss="listwise"), train_idx=train_idx, val_idx=val_idx)
    report("Transformer, listwise", tr_lw)
    cfg_dc = ScorerConfig(loss="mse", predict_dc=True, dc_weight=1.0)
    tr_dc = MutationScorer(pool.size, seq_len, cfg_dc, m=src_inst.m)
    train_scorer(tr_dc, src, cfg_dc, train_idx=train_idx, val_idx=val_idx)
    report("Transformer, MSE + Δc head", tr_dc)
    torch.save({"mse": tr.state_dict(), "listwise": tr_lw.state_dict(), "dc": tr_dc.state_dict()}, args.out_dir / f"scorers_{args.tag}.pt")

    df = pd.DataFrame(results)
    df.to_csv(args.out_dir / f"learner_matrix_{args.tag}.csv", index=False)
    cols = ["model", "split", "class", "spearman", "hit@1", "hit@10", "hit@50", "rank_of_pred_best", "mean_true_top5", "mean_true_random", "improving_top5", "improving_random", "regret", "oracle_gain"]
    (args.out_dir / f"learner_matrix_{args.tag}.md").write_text(df[cols].round(4).to_markdown(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
