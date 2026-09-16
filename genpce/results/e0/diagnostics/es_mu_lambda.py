"""(mu+lambda)-ES reference in the slot/template token space, 100k evaluations."""
import sys, numpy as np, time
sys.path.insert(0, r"C:\Users\Ashra\Desktop\GQE+PCE Project\experiments")
from common import benchmark_instance, cubic_cset
from genpce.pool import native_chain_pool
from genpce.sim import StatevectorEvaluator
from genpce.pce import decode_signs
from genpce.problems import one_pass_bit_swap
inst = benchmark_instance("reg3", 60, 0); cset = cubic_cset(60); bk = inst.best_known()
pool = native_chain_pool(cset.n); ev = StatevectorEvaluator(cset); N = 48
rng = np.random.default_rng(0); mu, lam = 10, 50
pop = rng.integers(0, pool.size, size=(mu, N))
def fit(seqs):
    c = ev.evaluate(pool.to_circuits(seqs)); return inst.cut_values(decode_signs(c))/bk, c
f, _ = fit(pop); evals = mu; t0 = time.perf_counter()
for g in range(2000):
    parents = pop[rng.integers(0, mu, size=lam)]
    kids = parents.copy(); k = rng.integers(1, 3, size=lam)
    for i in range(lam):
        pos = rng.choice(N, size=k[i], replace=False); kids[i, pos] = rng.integers(0, pool.size, size=k[i])
    fk, ck = fit(kids); evals += lam
    allp = np.concatenate([pop, kids]); allf = np.concatenate([f, fk]); order = np.argsort(allf)[::-1][:mu]
    pop, f = allp[order], allf[order]
    if g % 100 == 0 or g == 1999:
        print(f"  gen={g:4d} evals={evals:6d} best={f[0]:.3f} mu-mean={f.mean():.3f} {time.perf_counter()-t0:4.0f}s", flush=True)
c = ev.evaluate([pool.to_circuit(pop[0])])[0]
print(f"DONE (mu+lambda)-ES best raw={f[0]:.3f} +LS={inst.cut_value(one_pass_bit_swap(inst, decode_signs(c)))/bk:.3f} median|c|={np.median(np.abs(c)):.3f}", flush=True)
