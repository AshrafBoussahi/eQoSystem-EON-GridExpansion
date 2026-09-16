"""Project 2: the exact MaxCut post-processor from Sciorilli's own source
(src/utilities.py: local_search/_one_flip/_two_flip), NOT the best-
improvement/all-pairs version the protocol document's prose paraphrase
suggested -- read directly from their code (Project 2/MarcoSciorilli/...
/src/utilities.py lines 79-156) and confirmed materially different in two
ways:

1. `_one_flip` is a single GREEDY SEQUENTIAL sweep over nodes in index
   order: for each node i, flip immediately if the flip strictly improves
   the cut, then move on -- not "evaluate every node, take the single best
   flip" (best-improvement), which is what an initial reading of the
   protocol text would suggest and what this file's first draft
   implemented before the actual source was available.
2. `_two_flip` only tries pairs that are actual graph EDGES (i,j) -- not
   every C(m,2) pair. This is also why it stays tractable at m=7000 (G60
   has 17,148 edges, versus ~24.5 million node pairs).

Overall structure (`local_search`, utilities.py lines 79-85): alternate one
full one_flip sweep + one full two_flip sweep, repeat while the result
keeps changing, then apply one FINAL [one_flip, two_flip] pass after the
loop exits (an extra pass beyond the fixed point, matching their code
exactly, not simplified away).

Delta formulas below are algebraically equivalent to what their
_update_cut/_update_cut_double compute (verified independently by direct
brute-force comparison, not assumed), just vectorized/O(1)-per-move via a
maintained local-field vector instead of their per-move edge-dict rebuild
-- same algorithm, faster bookkeeping.
"""
import numpy as np


def cut_value(x: np.ndarray, W: np.ndarray) -> float:
    s = 1 - 2 * x
    return float(0.25 * np.sum(W * (1 - np.outer(s, s))))


def _one_flip_sweep(s: np.ndarray, h: np.ndarray, W: np.ndarray) -> tuple:
    """Greedy sequential sweep, node order 0..m-1: flip immediately if
    delta_1(i) = s_i * h_i > 0 (strictly improves), updating h incrementally
    before moving to the next node (so later nodes in the same sweep see
    the effect of earlier flips within the sweep -- matches the reference
    algorithm's in-place `best_solution` mutation)."""
    m = len(s)
    changed = False
    for i in range(m):
        delta = s[i] * h[i]
        if delta > 1e-9:
            h -= 2 * s[i] * W[:, i]  # uses OLD s[i]; h_k -= 2*W_ki*s_i_old (derivation in module docstring)
            s[i] = -s[i]
            changed = True
    return s, h, changed


def _two_flip_sweep(s: np.ndarray, h: np.ndarray, W: np.ndarray, edges: list) -> tuple:
    """Greedy sequential sweep over EDGES only (i,j): flip both endpoints
    immediately if delta_2(i,j) = s_i*h_i + s_j*h_j - 2*W_ij*s_i*s_j > 0.
    h_new = h_old - 2*s_i_old*W[:,i] - 2*s_j_old*W[:,j] (derived directly:
    h_k = sum_l W_kl s_l, only the l=i and l=j terms change when both flip,
    each by -2*W_ki*s_i_old / -2*W_kj*s_j_old respectively -- verified
    against brute-force cut recomputation, see test suite)."""
    changed = False
    for i, j in edges:
        delta = s[i] * h[i] + s[j] * h[j] - 2.0 * W[i, j] * s[i] * s[j]
        if delta > 1e-9:
            si_old, sj_old = s[i], s[j]
            h -= 2 * si_old * W[:, i] + 2 * sj_old * W[:, j]
            s[i], s[j] = -si_old, -sj_old
            changed = True
    return s, h, changed


def local_search_1bit_2bit(x0: np.ndarray, W: np.ndarray, edges: list = None,
                            max_outer_rounds: int = 500) -> tuple:
    """Sciorilli's exact local_search(): alternate one_flip+two_flip sweeps
    until the combined result stops changing, then one final extra pass.
    `edges`: list of (i,j) with i<j; derived from W's nonzero entries if not
    given. Returns (x_final, cut_value_final, n_outer_rounds_used)."""
    m = W.shape[0]
    if edges is None:
        iu, ju = np.nonzero(np.triu(W, k=1))
        edges = list(zip(iu.tolist(), ju.tolist()))

    x = x0.copy().astype(int)
    s = (1 - 2 * x).astype(np.float64)
    h = W @ s

    def one_pass(s, h):
        s, h, c1 = _one_flip_sweep(s, h, W)
        s, h, c2 = _two_flip_sweep(s, h, W, edges)
        return s, h, (c1 or c2)

    for round_i in range(max_outer_rounds):
        s_before = s.copy()
        s, h, _ = one_pass(s, h)
        if np.array_equal(s, s_before):
            break
    # one final extra pass, matching the reference implementation exactly
    s, h, _ = one_pass(s, h)

    x_final = ((1 - s) / 2).astype(int)
    return x_final, cut_value(x_final, W), round_i
