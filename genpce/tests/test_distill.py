"""Target-correlation distillation: corpus, matching reward, forward and inverse models (tiny scale)."""

import numpy as np
import torch

from genpce.pce import CorrelatorSet
from genpce.pool import native_chain_pool
from genpce.problems import random_regular
from genpce.sim import StatevectorEvaluator
from genpce.train import (
    ConditionalGenerator,
    CorrelationTargetReward,
    CorrelatorPredictor,
    EvolutionConfig,
    EvolutionarySearch,
    build_vqa_corpus,
    reconstruction_metrics,
    train_forward,
    train_inverse,
)


def _setup():
    inst = random_regular(12, 3, seed=3)
    cset = CorrelatorSet.build(4, 2, m=12)
    pool = native_chain_pool(4)
    return inst, cset, pool, StatevectorEvaluator(cset)


def test_corpus_records_every_checkpoint_consistently():
    inst, cset, _, _ = _setup()
    corpus = build_vqa_corpus(inst, cset, layers_list=(2,), seeds=range(2), max_steps=12, every=4)
    assert corpus.corr.shape[1] == cset.m and len(corpus.finals()) == 2
    assert np.all(np.diff(corpus.step[corpus.run == 0]) > 0)
    assert corpus.loss[corpus.run == 0][-1] <= corpus.loss[corpus.run == 0][0] + 1e-6  # Adam does not go uphill over the run


def test_target_reward_is_maximal_at_the_target_and_search_reduces_distance():
    inst, cset, pool, ev = _setup()
    rng = np.random.default_rng(0)
    target = ev.evaluate(pool.to_circuits(rng.integers(0, pool.size, size=(1, 16))))[0]
    reward = CorrelationTargetReward.for_target(inst, cset, target, lam_sign=0.5, lam_margin=0.1)
    s_t, _ = reward(target[None])
    s_r, _ = reward(ev.evaluate(pool.to_circuits(rng.integers(0, pool.size, size=(5, 16)))))
    assert np.all(s_t >= s_r)
    es = EvolutionarySearch(inst, cset, pool, ev, CorrelationTargetReward.for_target(inst, cset, target), EvolutionConfig(seq_len=16, budget=400, mu=5, lam=20, seed=0))
    df = es.run()
    assert -df.best_score.iloc[-1] <= -df.best_score.iloc[0]
    met = reconstruction_metrics(es.best["corr"], target, inst)
    assert 0 <= met["A_x"] <= 1 and met["D_c"] >= 0 and met["A_x_flip"] >= 0.5


def test_forward_and_inverse_models_train_and_sample():
    inst, cset, pool, ev = _setup()
    rng = np.random.default_rng(0)
    tok = rng.integers(0, pool.size, size=(64, 16))
    corr = ev.evaluate(pool.to_circuits(tok)).astype(np.float32)
    fwd = CorrelatorPredictor(pool.size, 16, cset.m, d_model=32, n_layers=1, n_heads=2)
    info = train_forward(fwd, tok, corr, train_idx=np.arange(48), val_idx=np.arange(48, 64), max_steps=60, batch=16)
    assert np.isfinite(info["best_val_mse"]) and fwd(torch.as_tensor(tok[:3])).shape == (3, cset.m)
    gen = ConditionalGenerator(pool.size, 16, cset.m, d_model=32, n_layers=1, n_heads=2)
    info = train_inverse(gen, tok, corr, train_idx=np.arange(48), val_idx=np.arange(48, 64), max_steps=60, batch=16)
    assert info["best_val_nll_per_token"] < info["uniform_nll_per_token"] + 0.5
    seqs = gen.sample(torch.as_tensor(corr[0]), 5, temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert seqs.shape == (5, 16) and int(seqs.max()) < pool.size
    lp = gen.log_probs(torch.as_tensor(corr[:5]), seqs)
    assert lp.shape == (5, 16) and torch.all(lp <= 0)
