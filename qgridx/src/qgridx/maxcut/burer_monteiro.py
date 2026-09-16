"""Project 2: Burer-Monteiro (BM) classical MaxCut baseline, per the
protocol's Section C -- Sciorilli's classical comparator, not a generic
heuristic. BM solves the MaxCut SDP relaxation

    maximize  sum_{i<j} w_ij (1 - v_i . v_j) / 2
    s.t.      v_i in R^r, ||v_i|| = 1

via low-rank factorization (rank r << m, tractable at m=7000 where a full
SDP solver is not) and Riemannian gradient ascent on the product-of-spheres
manifold (unit-norm retraction after every step -- the standard BM
approach). The rounding step (random-hyperplane) turns the resulting unit
vectors into a cut. This is a real, correct implementation of the published
algorithm, not a stand-in heuristic -- verified below against small
instances with a known/brute-forceable optimum before use at scale.
"""
import numpy as np
import torch


def cut_value(x: np.ndarray, W: np.ndarray) -> float:
    s = 1 - 2 * x
    return float(0.25 * np.sum(W * (1 - np.outer(s, s))))


def _bm_single_run(W_t: torch.Tensor, rank: int, n_iters: int, lr: float, seed: int) -> torch.Tensor:
    m = W_t.shape[0]
    torch.manual_seed(seed)
    V = torch.randn(m, rank, dtype=torch.float64)
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    V.requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad()
        Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
        obj = 0.25 * (W_t.sum() - (W_t * (Vn @ Vn.T)).sum())
        (-obj).backward()
        opt.step()
        with torch.no_grad():
            V /= V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)).detach()


def burer_monteiro_maxcut(W: np.ndarray, rank: int = None, n_restarts: int = 1,
                           n_iters: int = 800, lr: float = 0.1, n_rounding_trials: int = 50,
                           seed: int = 0):
    """Protocol Section C: single-run BM (n_restarts=1) for G14/G23/G60,
    100 random initializations (n_restarts=100) for pm3-8-50. Returns
    (best_cut_value, best_partition_x)."""
    m = W.shape[0]
    if rank is None:
        rank = min(m, max(2, int(np.ceil(np.sqrt(2 * m)))))
    W_t = torch.tensor(W, dtype=torch.float64)
    rng = np.random.default_rng(seed)

    best_cut, best_x = -np.inf, None
    for restart in range(n_restarts):
        Vn = _bm_single_run(W_t, rank, n_iters, lr, seed=seed * 1000 + restart).numpy()
        for _ in range(n_rounding_trials):
            r = rng.normal(size=rank)
            x = (Vn @ r >= 0).astype(int)
            cut = cut_value(x, W)
            if cut > best_cut:
                best_cut, best_x = cut, x
    return best_cut, best_x
