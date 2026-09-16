"""(mu+lambda)-ES with margin-aware rewards: does a margin term buy robust correlators without losing cut?"""
import sys, numpy as np, time
sys.path.insert(0, r"C:\Users\Ashra\Desktop\GQE+PCE Project\experiments")
from common import benchmark_instance, cubic_cset
from genpce.pool import native_chain_pool
from genpce.sim import StatevectorEvaluator, ShotEvaluator
from genpce.pce import decode_signs, RelaxedLossParams, relaxed_loss_np
from genpce.problems import one_pass_bit_swap
inst = benchmark_instance("reg3", 60, 0); cset = cubic_cset(60); bk = inst.best_known()
lp = RelaxedLossParams.sciorilli(inst, cset.n, cset.k)
pool = native_chain_pool(cset.n); ev = StatevectorEvaluator(cset); N = 48
def reward(corr, mode, lam, tau=0.2):
    cut = inst.cut_values(decode_signs(corr))/bk
    if mode == "cut": return cut, cut
    if mode == "margin":
        marg = np.mean(np.minimum(np.abs(corr), tau), axis=-1)/tau
        return cut + lam*marg, cut
    if mode == "relaxed_mix":
        rel = -relaxed_loss_np(corr, inst, lp)/lp.nu
        return cut + lam*rel, cut
def es(mode, lam, budget=20000, mu=10, lamb=50, seed=0):
    rng = np.random.default_rng(seed); pop = rng.integers(0, pool.size, size=(mu, N))
    def fit(seqs):
        c = ev.evaluate(pool.to_circuits(seqs)); f, cut = reward(c, mode, lam); return f, cut, c
    f, cut, c = fit(pop); evals = mu
    while evals < budget:
        kids = pop[rng.integers(0, mu, size=lamb)].copy(); k = rng.integers(1, 3, size=lamb)
        for i in range(lamb):
            pos = rng.choice(N, size=k[i], replace=False); kids[i, pos] = rng.integers(0, pool.size, size=k[i])
        fk, ck, cc = fit(kids); evals += lamb
        allp, allf, allcut, allc = np.concatenate([pop, kids]), np.concatenate([f, fk]), np.concatenate([cut, ck]), np.concatenate([c, cc])
        o = np.argsort(allf)[::-1][:mu]; pop, f, cut, c = allp[o], allf[o], allcut[o], allc[o]
    best = pop[0]; cb = c[0]
    # robustness: decode from 1000-shot and 200-shot estimates (20 repetitions)
    rob = {}
    for shots in (200, 1000):
        se = ShotEvaluator(cset, shots=shots, rng=np.random.default_rng(0))
        cuts = [inst.cut_value(decode_signs(se.evaluate([pool.to_circuit(best)])[0]))/bk for _ in range(20)]
        rob[shots] = (np.mean(cuts), np.min(cuts))
    ls = inst.cut_value(one_pass_bit_swap(inst, decode_signs(cb)))/bk
    return cut[0], ls, np.median(np.abs(cb)), np.min(np.abs(cb)), rob
for mode, lam in (("cut", 0.0), ("margin", 0.1), ("margin", 0.3), ("margin", 1.0), ("relaxed_mix", 0.5), ("relaxed_mix", 2.0)):
    t = time.perf_counter(); cut, ls, medc, minc, rob = es(mode, lam)
    print(f"ES reward={mode:12s} lam={lam:<4} | exact cut={cut:.3f} +LS={ls:.3f} | med|c|={medc:.3f} min|c|={minc:.3f} | shot-decoded cut: 200 shots mean={rob[200][0]:.3f} min={rob[200][1]:.3f}; 1000 shots mean={rob[1000][0]:.3f} min={rob[1000][1]:.3f} | {time.perf_counter()-t:.0f}s", flush=True)
