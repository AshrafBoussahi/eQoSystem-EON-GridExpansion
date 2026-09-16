"""Classical MaxCut solvers used to certify or estimate the best-known cut value.

* :func:`exact_milp` — exact optimum via mixed-integer programming (HiGHS through SciPy); practical
  for sparse graphs up to ~100–150 vertices.
* :func:`simulated_annealing` — multi-restart simulated annealing with incremental gain updates,
  followed by 1-flip local search; the "best-known" estimator for larger instances (Sciorilli et al.
  used Burer–Monteiro and Breakout Local Search for the same purpose).
* :func:`best_known` — convenience wrapper with on-disk caching keyed by instance name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from genpce.problems.maxcut import MaxCutInstance, local_search_to_convergence

__all__ = ["exact_milp", "simulated_annealing", "best_known"]


def exact_milp(inst: MaxCutInstance, time_limit: float = 600.0) -> tuple[np.ndarray, float, bool]:
    """Solve MaxCut exactly as a MILP (HiGHS).

    Formulation: binary ``b_i``; edge variables ``y_e ∈ [0, 1]`` with ``y_e ≤ b_i + b_j`` and
    ``y_e ≤ 2 − b_i − b_j`` for positive weights (so ``y_e ≤ [b_i ≠ b_j]``) and ``y_e ≥ |b_i − b_j|``
    for negative weights; maximise ``Σ w_e y_e``. ``b_0 = 0`` breaks the global spin-flip symmetry.

    Returns ``(x, value, proven_optimal)``; ``x`` is in ``{-1, +1}^m``.
    """
    m, E = inst.m, inst.num_edges
    nvar = m + E
    c = np.zeros(nvar)
    c[m:] = -inst.weights  # milp minimises
    lower: list[float] = []
    upper: list[float] = []
    A = lil_matrix((2 * E + 1, nvar))
    r = 0
    for e, ((i, j), w) in enumerate(zip(inst.edges, inst.weights)):
        ye = m + e
        if w >= 0:
            # y - b_i - b_j <= 0 ; y + b_i + b_j <= 2
            A[r, ye] = 1; A[r, i] = -1; A[r, j] = -1; lower.append(-np.inf); upper.append(0.0); r += 1
            A[r, ye] = 1; A[r, i] = 1; A[r, j] = 1; lower.append(-np.inf); upper.append(2.0); r += 1
        else:
            # y - b_i + b_j >= 0 ; y + b_i - b_j >= 0
            A[r, ye] = 1; A[r, i] = -1; A[r, j] = 1; lower.append(0.0); upper.append(np.inf); r += 1
            A[r, ye] = 1; A[r, i] = 1; A[r, j] = -1; lower.append(0.0); upper.append(np.inf); r += 1
    A[r, 0] = 1; lower.append(0.0); upper.append(0.0); r += 1  # symmetry breaking b_0 = 0
    constraints = LinearConstraint(A.tocsr(), np.array(lower), np.array(upper))
    integrality = np.zeros(nvar)
    integrality[:m] = 1
    bounds = Bounds(np.zeros(nvar), np.ones(nvar))
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": time_limit})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    b = np.round(res.x[:m]).astype(int)
    x = np.where(b == 1, 1, -1).astype(np.int8)
    value = inst.cut_value(x)
    proven = bool(res.status == 0)
    return x, value, proven


def simulated_annealing(
    inst: MaxCutInstance,
    *,
    restarts: int = 10,
    sweeps: int = 300,
    t_start: float | None = None,
    t_end: float = 0.02,
    seed: int = 0,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Multi-restart simulated annealing with incremental gain updates + local search polish.

    Args:
        inst: Instance.
        restarts: Independent runs (random initialisation unless ``x0`` is given for the first).
        sweeps: Metropolis sweeps (``m`` attempted flips each) per run, geometric cooling.
        t_start: Initial temperature (default: mean absolute edge weight × 2).
        t_end: Final temperature.
        seed: RNG seed.
        x0: Optional warm start for the first restart.

    Returns ``(best_x, best_value)``.
    """
    rng = np.random.default_rng(seed)
    adj = inst.adjacency_lists()
    m = inst.m
    if t_start is None:
        t_start = 2.0 * float(np.mean(np.abs(inst.weights))) if inst.num_edges else 1.0
    best_x = None
    best_val = -math.inf
    cooling = (t_end / t_start) ** (1.0 / max(1, sweeps - 1))
    for run in range(restarts):
        if run == 0 and x0 is not None:
            x = np.array(x0, dtype=np.int8, copy=True)
        else:
            x = rng.choice(np.array([-1, 1], dtype=np.int8), size=m)
        xl = x.tolist()
        # gains[i] = increase in cut if spin i is flipped = x_i * sum_j w_ij x_j
        gains = [0.0] * m
        for i in range(m):
            gains[i] = xl[i] * sum(w * xl[j] for j, w in adj[i])
        value = inst.cut_value(x)
        t = t_start
        cur_best_val, cur_best_x = value, list(xl)
        for _ in range(sweeps):
            order = rng.permutation(m)
            thresholds = rng.random(m)
            for idx in range(m):
                i = int(order[idx])
                g = gains[i]
                if g > 0 or thresholds[idx] < math.exp(g / t):
                    xl[i] = -xl[i]
                    value += g
                    gains[i] = -g
                    xi = xl[i]
                    for j, w in adj[i]:
                        gains[j] += 2.0 * w * xi * xl[j]
                    if value > cur_best_val:
                        cur_best_val, cur_best_x = value, list(xl)
            t *= cooling
        polished = local_search_to_convergence(inst, np.array(cur_best_x, dtype=np.int8))
        val = inst.cut_value(polished)
        if val > best_val:
            best_val, best_x = val, polished
    return best_x, float(best_val)


def best_known(
    inst: MaxCutInstance,
    *,
    cache_dir: str | Path | None = None,
    exact_max_m: int = 120,
    milp_time_limit: float = 600.0,
    sa_restarts: int = 10,
    sa_sweeps: int = 300,
    seed: int = 0,
) -> MaxCutInstance:
    """Attach a best-known (exact when affordable) cut value to ``inst.meta``.

    Uses the MILP for ``m ≤ exact_max_m`` (falls back to SA if the time limit is hit) and
    multi-restart SA otherwise. Results are cached as JSON in ``cache_dir`` keyed by instance name.
    """
    if inst.best_known() is not None:
        return inst
    cache_path = Path(cache_dir) / f"{inst.name}.bestknown.json" if cache_dir else None
    if cache_path is not None and cache_path.exists():
        rec = json.loads(cache_path.read_text())
        return inst.with_best_known(rec["value"], exact=rec["exact"], source=rec["source"])

    t0 = perf_counter()
    x_sa, v_sa = simulated_annealing(inst, restarts=sa_restarts, sweeps=sa_sweeps, seed=seed)
    value, exact, source = v_sa, False, f"SA(restarts={sa_restarts},sweeps={sa_sweeps})"
    if inst.m <= exact_max_m:
        try:
            _, v_milp, proven = exact_milp(inst, time_limit=milp_time_limit)
            if proven:
                value, exact, source = max(v_milp, v_sa), True, "MILP(HiGHS)"
                if v_sa > v_milp + 1e-9:
                    exact = False  # should not happen; be conservative
            else:
                value = max(value, v_milp)
                source += "+MILP(timeout)"
        except RuntimeError:
            pass
    rec = {"value": float(value), "exact": bool(exact), "source": source, "seconds": perf_counter() - t0}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rec))
    return inst.with_best_known(value, exact=exact, source=source)
