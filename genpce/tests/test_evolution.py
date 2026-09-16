"""Evolutionary engine, reward and prior: smoke tests on a tiny instance."""

import numpy as np

from genpce.pce import CorrelatorSet
from genpce.pool import native_chain_pool
from genpce.problems import random_regular
from genpce.sim import StatevectorEvaluator
from genpce.train import EvolutionConfig, EvolutionarySearch, ModelProposal, PriorConfig, Reward, train_prior


def _setup():
    inst = random_regular(12, 3, seed=3)
    cset = CorrelatorSet.build(4, 2, m=12)
    pool = native_chain_pool(4)
    return inst, cset, pool, StatevectorEvaluator(cset)


def test_reward_never_uses_optimum_and_is_rank_consistent():
    inst, cset, pool, ev = _setup()
    r_exact, r_shaped = Reward.exact(inst, cset), Reward.shaped(inst, cset)
    corr = ev.evaluate(pool.to_circuits(np.random.default_rng(0).integers(0, pool.size, size=(8, 16))))
    s, cut = r_exact(corr)
    assert np.allclose(s * r_exact.params.nu, cut)  # exact reward = cut / nu, nu is graph-intrinsic
    assert inst.best_known() is None  # nothing in this test knows the optimum
    s2, cut2 = r_shaped(corr)
    assert np.allclose(cut, cut2) and s2.shape == (8,)


def test_evolution_improves_and_archives():
    inst, cset, pool, ev = _setup()
    es = EvolutionarySearch(inst, cset, pool, ev, Reward.shaped(inst, cset), EvolutionConfig(seq_len=16, budget=600, mu=5, lam=20, seed=0))
    df = es.run()
    assert es.evaluations >= 600 and len(es.database) == es.evaluations
    assert df.best_cut.iloc[-1] >= df.best_cut.iloc[0]
    assert es.elites(3).shape == (3, 16)
    cut, cut_ls = es.best_with_local_search()
    assert cut_ls >= cut


def test_prior_trains_and_proposes():
    inst, cset, pool, ev = _setup()
    elites = np.random.default_rng(0).integers(0, pool.size, size=(40, 16))
    model, info = train_prior(elites, pool.size, PriorConfig(d_model=32, n_layers=2, n_heads=2, max_steps=60, patience=40))
    assert info["best_val_nll_per_token"] > 0
    prop = ModelProposal(model, 16, beta=1.0)
    assert prop(7).shape == (7, 16)
    es = EvolutionarySearch(inst, cset, pool, ev, Reward.shaped(inst, cset), EvolutionConfig(seq_len=16, budget=200, mu=5, lam=20, model_fraction=0.5), proposal=prop)
    es.run()
    assert es.evaluations >= 200


def test_online_proposals_run_and_archive_is_maintained():
    from genpce.train import PBILProposal, TransformerProposal
    inst, cset, pool, ev = _setup()
    for prop in (PBILProposal(16, pool.size), TransformerProposal(16, pool.size, d_model=32, n_layers=2, n_heads=2, steps_per_update=2)):
        es = EvolutionarySearch(inst, cset, pool, ev, Reward.shaped(inst, cset), EvolutionConfig(seq_len=16, budget=300, mu=5, lam=20, model_fraction=0.5, archive_size=30), proposal=prop)
        es.run()
        assert 0 < len(es.archive_tokens) <= 30 and np.all(np.diff(es.archive_scores) <= 1e-12)
        assert len(np.unique(es.archive_tokens, axis=0)) == len(es.archive_tokens)
    assert PBILProposal(16, pool.size).entropy() > 3.0


def test_cpo_trainer_runs_and_learns_something():
    from genpce.train import CPOConfig, CPOTrainer
    inst, cset, pool, ev = _setup()
    tr = CPOTrainer(inst, cset, pool, ev, Reward.shaped(inst, cset), CPOConfig(seq_len=16, budget=640, samples_per_step=64, d_model=32, n_layers=2, n_heads=2, lr=1e-3))
    df = tr.run()
    assert tr.evaluations == 640 and len(df) == 10
    assert 0 <= df.pref_acc.iloc[-1] <= 1 and np.isfinite(df.loss).all()
    out = tr.evaluate(num=32)
    assert out["cut_ls"] >= out["cut"] and tr.evaluations == 672


def test_mutation_dataset_and_scorer():
    from genpce.train import LearnedMutationProposal, MutationScorer, ScorerConfig, build_mutation_dataset, enumerate_single_mutations, evaluate_scorer, train_scorer
    inst, cset, pool, ev = _setup()
    parents = np.random.default_rng(0).integers(0, pool.size, size=(6, 8))
    children, pos, tok = enumerate_single_mutations(parents[0], pool.size)
    assert children.shape == (8 * (pool.size - 1), 8) and np.all(children[np.arange(len(pos)), pos] == tok)
    data = build_mutation_dataset(parents, pool, ev, Reward.shaped(inst, cset), keep_dc=True)
    assert data.dR.shape == (6, 8, pool.size) and np.isnan(data.dR[0, 0, parents[0, 0]]) and data.dc.shape[-1] == inst.m
    cfg = ScorerConfig(d_model=32, n_layers=1, n_heads=2, max_steps=40, patience=40, predict_dc=True)
    model = MutationScorer(pool.size, 8, cfg, m=inst.m)
    train_scorer(model, data, cfg, train_idx=np.arange(4), val_idx=np.arange(4, 6))
    met = evaluate_scorer(model, data, np.arange(4, 6))
    assert -1 <= met["spearman"] <= 1 and 0 <= met["hit@10"] <= 1
    prop = LearnedMutationProposal(model, epsilon=0.5)
    kids, p, t = prop.propose(parents[:3])
    assert kids.shape == (3, 8) and np.all(kids[np.arange(3), p] == t) and np.all(parents[np.arange(3), p] != t)
    prop.record(parents[:3], p, t, np.zeros(3)); prop.record(np.repeat(parents[:1], 40, axis=0), np.zeros(40, int), np.ones(40, int), np.zeros(40))
    assert np.isfinite(prop.update())


def test_es_with_learned_mutation_proposal():
    from genpce.train import LearnedMutationProposal, MutationScorer, ScorerConfig
    inst, cset, pool, ev = _setup()
    model = MutationScorer(pool.size, 16, ScorerConfig(d_model=32, n_layers=1, n_heads=2))
    prop = LearnedMutationProposal(model, epsilon=0.3, steps_per_update=1)
    es = EvolutionarySearch(inst, cset, pool, ev, Reward.shaped(inst, cset), EvolutionConfig(seq_len=16, budget=400, mu=5, lam=20, learned_fraction=0.5), mutation_proposal=prop)
    df = es.run()
    assert es.evaluations >= 400 and len(prop.buffer) >= 380 and np.isfinite(df.scorer_loss.iloc[-1])
