"""Project 2, Arm C: Blind control (protocol Section 1.5). Random
correlation values (no circuit, no training, no learned structure of any
kind) fed through the IDENTICAL sign-readout + 1-bit/2-bit local-search
pipeline used by Arms A and B. Purpose: isolate how much of Arms A/B's
advantage comes from the decoder alone vs. the learned/optimized circuit.

pi sampled Uniform(-1,1) per decision -- the protocol text's own first-
listed option ("uniform or Gaussian, matching the marginal distribution of
Arm A/B correlations if known"); a fixed prior choice was needed to avoid a
circular dependency on Arm A/B's own empirical output, so Uniform(-1,1)
(the natural non-informative default over the correlator's valid range) is
used and disclosed here rather than fit post-hoc to Arm A/B's marginals.
"""
import numpy as np

from qgridx.maxcut.local_search import cut_value, local_search_1bit_2bit
from qgridx.decoder.repair import sign_readout


def run_blind(W: np.ndarray, m: int, E_ref: float, n_restarts: int = 2, seed: int = 0):
    edges = list(zip(*[a.tolist() for a in np.nonzero(np.triu(W, k=1))]))
    rng = np.random.default_rng(seed)
    results = []
    for s in range(n_restarts):
        pi = rng.uniform(-1, 1, size=m)
        x_raw = sign_readout(pi.copy())
        E_raw = cut_value(x_raw, W)
        x_full, E_full, n_rounds = local_search_1bit_2bit(x_raw, W, edges=edges)
        results.append(dict(seed=s, E_raw=E_raw, E_full=E_full, r_circuit=E_raw / E_ref, r_full=E_full / E_ref,
                             local_search_rounds=n_rounds, x_raw=x_raw.tolist(), x_full=x_full.tolist()))
    return dict(per_restart=results, mean_r_circuit=float(np.mean([r["r_circuit"] for r in results])),
                mean_r_full=float(np.mean([r["r_full"] for r in results])),
                std_r_full=float(np.std([r["r_full"] for r in results])))
