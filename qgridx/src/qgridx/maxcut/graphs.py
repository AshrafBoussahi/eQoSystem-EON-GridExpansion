"""Project 2: load the 4 named Sciorilli/G-set benchmark instances, and
generate Phase 1/2/3 random-graph instances from Sciorilli's own density
formula (src/newgraph.py's RandomGraphs.create_graph, read directly --
`p = uniform(6/(n-1), dens(n))`, `dens(n) = exp(-0.8541-0.0055n+4.003e-6n^2
-1.158e-9n^3)` for n<=1700), substituting networkx.gnp_random_graph for
their missing `./rudy` binary (disclosed substitution, protocol Section B's
own allowance: "you may generate random graphs, but you must disclose the
ensemble parameters explicitly").
"""
import networkx as nx
import numpy as np


def load_gset(path: str) -> np.ndarray:
    """G-set format: header 'n_vertices n_edges', then 'u v w' per edge,
    1-indexed."""
    with open(path) as f:
        lines = f.read().splitlines()
    m, _ = map(int, lines[0].split())
    W = np.zeros((m, m))
    for line in lines[1:]:
        if not line.strip():
            continue
        u, v, w = line.split()
        u, v, w = int(u) - 1, int(v) - 1, float(w)
        W[u, v] = w
        W[v, u] = w
    return W


def load_pm3(path: str) -> np.ndarray:
    """pm3-8-50 format: no header, 'u v w' per edge, 0-indexed, w in {-1,+1},
    512 vertices."""
    edges = []
    max_idx = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            u, v, w = line.split()
            u, v, w = int(u), int(v), float(w)
            edges.append((u, v, w))
            max_idx = max(max_idx, u, v)
    m = max_idx + 1
    W = np.zeros((m, m))
    for u, v, w in edges:
        W[u, v] = w
        W[v, u] = w
    return W


def sciorilli_density(n: int) -> float:
    """dens(n) from RandomGraphs.create_graph, src/newgraph.py line 112-116."""
    if n > 1700:
        return 0.01
    return np.exp(-0.8541 - 0.0055 * n + 4.003e-06 * n ** 2 - 1.158e-09 * n ** 3)


def generate_random_graph(m: int, seed: int, weight_mode: str = "pm1") -> np.ndarray:
    """Sciorilli's own graph ensemble (create_graph): p ~ Uniform(6/(m-1),
    dens(m)), edge existence ~ G(m,p), enforced connected (their
    gnp_random_connected_graph retries with p+=0.001 until min degree > 1;
    we do the same). `rudy` substituted by networkx.gnp_random_graph
    (disclosed, protocol-permitted substitution -- same target density,
    different sampler implementation). weight_mode: 'pm1' (+-1, matching
    pm3-8-50) or 'unit' (unweighted, matching G14/G23/G60)."""
    rng = np.random.default_rng(seed)
    p_lo = 6 / (m - 1)
    p_hi = sciorilli_density(m)
    p = rng.uniform(p_lo, max(p_hi, p_lo * 1.001))
    min_degree = 0
    attempt = 0
    g = None
    while min_degree <= 1:
        g = nx.gnp_random_graph(m, p, seed=seed * 100003 + attempt)
        degrees = [d for _, d in g.degree()]
        min_degree = min(degrees) if degrees else 0
        p = min(p + 0.001, 0.999)
        attempt += 1
        if attempt > 200:
            # extremely small/degenerate m: fall back to a connected ring + p-random extra edges
            g = nx.connected_watts_strogatz_graph(m, k=4, p=0.3, seed=seed)
            break
    W = nx.to_numpy_array(g)
    if weight_mode == "pm1":
        signs = rng.choice([-1.0, 1.0], size=W.shape)
        W = W * signs
        W = np.triu(W, 1)
        W = W + W.T
    return W


def is_nontrivial(W: np.ndarray) -> bool:
    """Exclude bipartite-complete or all-zero-weight instances (protocol
    1.1's non-triviality filter)."""
    if not np.any(W):
        return False
    g = nx.from_numpy_array(W)
    if nx.is_bipartite(g):
        complement_edges = g.number_of_nodes() * (g.number_of_nodes() - 1) // 2
        parts = nx.bipartite.sets(g) if nx.is_connected(g) else (set(), set())
        if parts[0] and parts[1] and g.number_of_edges() == len(parts[0]) * len(parts[1]):
            return False  # complete bipartite
    return True
