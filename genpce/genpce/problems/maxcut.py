"""(Weighted) MaxCut instances, objective, generators and light post-processing.

Conventions: spins ``x ∈ {-1, +1}^m``; the cut value is ``V(x) = Σ_{(i,j)∈E} W_ij (1 - x_i x_j)/2``
(Sciorilli et al., Eq. 4). Instances are immutable numpy containers so they can be shared across
processes and cached to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

__all__ = [
    "MaxCutInstance",
    "random_regular",
    "erdos_renyi",
    "with_pm1_weights",
    "load_gset",
    "one_pass_bit_swap",
    "local_search_to_convergence",
]


@dataclass(frozen=True)
class MaxCutInstance:
    """A weighted MaxCut instance on ``m`` vertices.

    Attributes:
        m: Number of vertices (= binary variables).
        edges: ``(E, 2)`` int array of vertex pairs (``i < j``).
        weights: ``(E,)`` float array of edge weights.
        name: Human-readable identifier (used for caching).
        meta: Free-form provenance information (generator, seed, best-known value, ...).
    """

    m: int
    edges: np.ndarray
    weights: np.ndarray
    name: str = "maxcut"
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ objective
    @property
    def num_edges(self) -> int:
        return int(len(self.edges))

    @property
    def total_weight(self) -> float:
        return float(self.weights.sum())

    def cut_value(self, x: np.ndarray) -> float:
        """Cut value of a single spin configuration ``x ∈ {-1,+1}^m``."""
        x = np.asarray(x)
        return float(0.5 * np.sum(self.weights * (1.0 - x[self.edges[:, 0]] * x[self.edges[:, 1]])))

    def cut_values(self, X: np.ndarray) -> np.ndarray:
        """Cut values for a batch ``X`` of shape ``(B, m)``."""
        X = np.asarray(X)
        prod = X[..., self.edges[:, 0]] * X[..., self.edges[:, 1]]
        return 0.5 * np.sum(self.weights * (1.0 - prod), axis=-1)

    def best_known(self) -> float | None:
        """Best-known (or exact) cut value if recorded in ``meta``."""
        value = self.meta.get("best_known")
        return None if value is None else float(value)

    def with_best_known(self, value: float, *, exact: bool, source: str) -> "MaxCutInstance":
        meta = dict(self.meta)
        meta.update({"best_known": float(value), "best_known_exact": bool(exact), "best_known_source": source})
        return MaxCutInstance(self.m, self.edges, self.weights, self.name, meta)

    # ------------------------------------------------------------------ graph helpers
    def graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.m))
        g.add_weighted_edges_from((int(i), int(j), float(w)) for (i, j), w in zip(self.edges, self.weights))
        return g

    def adjacency_lists(self) -> list[list[tuple[int, float]]]:
        adj: list[list[tuple[int, float]]] = [[] for _ in range(self.m)]
        for (i, j), w in zip(self.edges, self.weights):
            adj[int(i)].append((int(j), float(w)))
            adj[int(j)].append((int(i), float(w)))
        return adj

    def average_degree(self) -> float:
        return 2.0 * self.num_edges / self.m

    def regularisation_scale(self) -> float:
        """Sciorilli et al.'s ``ν``: an a-priori lower bound on the maximum cut.

        Unweighted graphs: Edwards–Erdős bound ``|E|/2 + (m-1)/4``. Weighted graphs: the
        Poljak–Turzík bound ``w(G)/2 + w(T_min)/4`` with ``T_min`` a minimum spanning tree.
        """
        if np.allclose(self.weights, 1.0):
            return self.num_edges / 2.0 + (self.m - 1) / 4.0
        tree = nx.minimum_spanning_tree(self.graph(), weight="weight")
        w_tree = float(sum(d["weight"] for _, _, d in tree.edges(data=True)))
        return self.total_weight / 2.0 + w_tree / 4.0

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "m": self.m,
            "edges": self.edges.tolist(),
            "weights": self.weights.tolist(),
            "name": self.name,
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "MaxCutInstance":
        payload = json.loads(Path(path).read_text())
        return cls(
            m=int(payload["m"]),
            edges=np.asarray(payload["edges"], dtype=np.int64).reshape(-1, 2),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            name=payload.get("name", "maxcut"),
            meta=payload.get("meta", {}),
        )

    @classmethod
    def from_graph(cls, g: nx.Graph, name: str = "maxcut", meta: dict | None = None) -> "MaxCutInstance":
        nodes = sorted(g.nodes())
        index = {v: i for i, v in enumerate(nodes)}
        edges, weights = [], []
        for u, v, d in g.edges(data=True):
            i, j = index[u], index[v]
            if i == j:
                continue
            edges.append((min(i, j), max(i, j)))
            weights.append(float(d.get("weight", 1.0)))
        order = np.lexsort((np.array(edges)[:, 1], np.array(edges)[:, 0])) if edges else np.array([], int)
        return cls(
            m=len(nodes),
            edges=np.asarray(edges, dtype=np.int64)[order].reshape(-1, 2),
            weights=np.asarray(weights, dtype=np.float64)[order],
            name=name,
            meta=meta or {},
        )


# ---------------------------------------------------------------------- generators
def random_regular(m: int, d: int = 3, seed: int = 0) -> MaxCutInstance:
    """Random ``d``-regular graph on ``m`` vertices (unit weights)."""
    g = nx.random_regular_graph(d, m, seed=seed)
    return MaxCutInstance.from_graph(
        g, name=f"reg{d}_m{m}_s{seed}", meta={"generator": "random_regular", "d": d, "seed": seed}
    )


def erdos_renyi(m: int, avg_degree: float = 4.0, seed: int = 0) -> MaxCutInstance:
    """``G(m, E)`` random graph with ``E = round(avg_degree · m / 2)`` edges (unit weights)."""
    num_edges = int(round(avg_degree * m / 2))
    g = nx.gnm_random_graph(m, num_edges, seed=seed)
    return MaxCutInstance.from_graph(
        g,
        name=f"er{avg_degree:g}_m{m}_s{seed}",
        meta={"generator": "erdos_renyi", "avg_degree": avg_degree, "seed": seed},
    )


def with_pm1_weights(inst: MaxCutInstance, seed: int = 0) -> MaxCutInstance:
    """Assign i.i.d. ``±1`` weights (G11/G14-style weighted MaxCut)."""
    rng = np.random.default_rng(seed)
    w = rng.choice(np.array([-1.0, 1.0]), size=inst.num_edges)
    meta = dict(inst.meta)
    meta.update({"weights": "pm1", "weight_seed": seed})
    return MaxCutInstance(inst.m, inst.edges, w, name=inst.name + "_pm1", meta=meta)


def load_gset(path: str | Path, name: str | None = None, best_known: float | None = None) -> MaxCutInstance:
    """Load a G-set file (``m |E|`` header, then ``i j w`` lines, 1-indexed)."""
    path = Path(path)
    lines = path.read_text().split("\n")
    m, num_edges = (int(v) for v in lines[0].split()[:2])
    edges, weights = [], []
    for line in lines[1 : 1 + num_edges]:
        parts = line.split()
        if len(parts) < 3:
            continue
        i, j, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
        edges.append((min(i, j), max(i, j)))
        weights.append(w)
    meta: dict[str, Any] = {"generator": "gset", "file": str(path)}
    if best_known is not None:
        meta.update({"best_known": float(best_known), "best_known_exact": False, "best_known_source": "gset table"})
    return MaxCutInstance(
        m=m,
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        weights=np.asarray(weights, dtype=np.float64),
        name=name or path.stem,
        meta=meta,
    )


# ---------------------------------------------------------------------- post-processing
def _gains(inst: MaxCutInstance, x: np.ndarray) -> np.ndarray:
    """Gain in cut value from flipping each spin: ``g_i = x_i Σ_j W_ij x_j``."""
    i, j = inst.edges[:, 0], inst.edges[:, 1]
    field_ = np.zeros(inst.m)
    np.add.at(field_, i, inst.weights * x[j])
    np.add.at(field_, j, inst.weights * x[i])
    return x * field_


def one_pass_bit_swap(inst: MaxCutInstance, x: np.ndarray) -> np.ndarray:
    """Sciorilli et al.'s post-processing: one sequential pass of improving single-bit flips.

    Cost ``Θ(|E|)``. Returns a new spin vector.
    """
    x = np.array(x, dtype=np.int8, copy=True)
    adj = inst.adjacency_lists()
    gains = _gains(inst, x.astype(np.float64))
    for i in range(inst.m):
        if gains[i] > 1e-12:
            x[i] = -x[i]
            gains[i] = -gains[i]
            for j, w in adj[i]:
                gains[j] += 2.0 * w * x[i] * x[j]
    return x


def local_search_to_convergence(inst: MaxCutInstance, x: np.ndarray, max_passes: int = 100) -> np.ndarray:
    """Repeat :func:`one_pass_bit_swap` until no improving flip remains (1-flip local optimum)."""
    x = np.array(x, dtype=np.int8, copy=True)
    for _ in range(max_passes):
        new = one_pass_bit_swap(inst, x)
        if np.array_equal(new, x):
            break
        x = new
    return x
