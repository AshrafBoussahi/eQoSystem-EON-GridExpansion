"""Diagnostic: which knob makes GenPCE learn on reg3_m60_s0? One config per process."""
import argparse, sys, time
from math import pi
import numpy as np, torch
sys.path.insert(0, r"C:\Users\Ashra\Desktop\GQE+PCE Project\experiments")
from common import benchmark_instance, cubic_cset
from genpce.pool import native_chain_pool
from genpce.sim import StatevectorEvaluator
from genpce.train import GenPCEConfig, GenPCETrainer
from genpce.model import sequence_log_probs

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--n-iter", type=int, default=5)
ap.add_argument("--reward", default="cut")
ap.add_argument("--pool", default="fine")  # fine | coarse
ap.add_argument("--seq-len", type=int, default=48)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--n-sample", type=int, default=50)
ap.add_argument("--loss", default="grpo")
ap.add_argument("--buffer", type=int, default=1000)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--template", type=int, default=1)
ap.add_argument("--elite", type=float, default=0.1)
ap.add_argument("--entropy", type=float, default=0.0)
ap.add_argument("--mutation", type=float, default=0.0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
inst = benchmark_instance("reg3", 60, 0); cset = cubic_cset(60); bk = inst.best_known()
angles = (pi/2, pi/4, pi/8, pi/16) if a.pool == "fine" else (pi/2, pi/4)
pool = native_chain_pool(cset.n, angles=angles, template=("brickwork_cz" if a.template else None), slot=bool(a.template), entangler_tokens=not a.template)
ev = StatevectorEvaluator(cset)
cfg = GenPCEConfig(seq_len=a.seq_len, epochs=a.epochs, n_sample=a.n_sample, n_batch=50, n_iter=a.n_iter,
                   buffer_size=a.buffer, lr=a.lr, reward=a.reward, loss=a.loss, elite_fraction=a.elite, entropy_weight=a.entropy, mutation_rate=a.mutation, seed=0, d_model=128, n_layers=4, n_heads=4)
tr = GenPCETrainer(inst, cset, pool, ev, cfg)
tag = f"H={a.entropy} mut={a.mutation} elite={a.elite} template={a.template} lr={a.lr} iter={a.n_iter} reward={a.reward} pool={a.pool} N={a.seq_len} ns={a.n_sample} loss={a.loss} buf={a.buffer}"
print(tag, "| vocab", pool.size, flush=True)
t0 = time.perf_counter()
def cb(r):
    if r["epoch"] % 20 == 0 or r["epoch"] == a.epochs - 1:
        # policy entropy per token (nats) on a fresh sample
        with torch.no_grad():
            toks = torch.as_tensor(np.stack(tr.buffer.tokens[-20:]))
            bos = torch.full((toks.shape[0], 1), tr.model.bos, dtype=torch.long)
            w = tr.model(torch.cat([bos, toks[:, :-1]], 1))
            p = torch.softmax(-tr.beta * w, -1); ent = float(-(p * torch.log(p + 1e-12)).sum(-1).mean())
        print(f"  ep={r['epoch']:3d} evals={r['evaluations']:5d} beta={r['beta']:.2f} mean={r['ratio_epoch_mean']:.3f} max_ep={r['cut_max']/bk:.3f} best={r['ratio_best']:.3f} ls={r['ratio_best_ls']:.3f} uniq={r['unique_frac']:.2f} 2q={r['two_qubit_best']} H={ent:.2f}/{np.log(pool.size):.2f} min|c|={r['min_abs_corr_best']:.3f} {time.perf_counter()-t0:4.0f}s", flush=True)
tr.run(callback=cb)
print(f"DONE {tag} | best raw={tr.best['cut']/bk:.3f} ls={tr.best['cut_ls']/bk:.3f} epoch={tr.best['epoch']}", flush=True)
