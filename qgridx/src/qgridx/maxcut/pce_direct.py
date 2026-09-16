"""Project 2, Arm A: PCE-direct, Sciorilli's own training loop (protocol
Section 1.3) -- circuit parameters trained by gradient descent directly
against the exact surrogate loss (Section A / src/loss.py's
_loss_numpy_PCE, read directly and reproduced in torch below), decoder is
NEVER touched during training, sign-readout + 1-bit/2-bit local search
applied exactly once after training converges.

Optimizer: torch.optim.Adam in place of their hand-rolled `adam()`
(src/optimizers.py) -- same algorithm (identical beta1=0.9, beta2=0.999,
eps=1e-8 defaults; their alpha=0.1 default matches lr=0.1 here), a
disclosed substitution of a battle-tested library implementation for a
hand-rolled one, not a different optimizer. Their early-stopping
(wait=10 iters without >0.1 total improvement) is replaced by: run a
fixed, generously-sized iteration budget and track the best-loss
parameters seen throughout -- strictly at least as thorough as early
stopping, just without the variable-length-loop complexity of doing it
per-restart inside one vectorized batch.

Initialization: Uniform(-pi, pi) per parameter, matching the angle range
their own `identity_start` also samples from (np.random.uniform(-1,1,...)
* pi) -- NOT their identity_start's specific "half-depth circuit + its own
inverse" construction, since it is not confirmed (from the source read so
far) whether headline runs actually use identity_start vs plain random
init. Disclosed as an un-replicated initialization-strategy detail.
"""
import networkx as nx
import numpy as np
import torch

from qgridx.maxcut.local_search import cut_value, local_search_1bit_2bit
from qgridx.decoder.repair import sign_readout
from qgridx.encoding.correlators import correlators_fast as correlators
from qgridx.encoding.families import random_assignment
from qgridx.maxcut.reference_ansatz import build_sciorilli_gate_sequence, run_circuit_pytorch


def _edge_list(W: np.ndarray):
    iu, ju = np.nonzero(np.triu(W, k=1))
    return [(int(i), int(j), float(W[i, j])) for i, j in zip(iu, ju)]


def compute_nu(W: np.ndarray, edges: list) -> float:
    """Sciorilli's exact nu (src/loss.py lines 194-199): unweighted case
    nu=(nedges/2+(m-1)/4) [algebraically ==(2|E|+m-1)/4, protocol Section A];
    weighted case nu=(total_weight/2+mst_weight/4), computed on a copy with
    negative weights zeroed (their `disposable_graph`, lines 48-54) --
    critically, their code zeroes the WEIGHT attribute but leaves the EDGE
    itself in the graph, so a formerly-negative edge still participates in
    the MST as a (now free, weight=0) connection. Building the graph from
    `edges` (every originally-nonzero entry, weight clamped to >=0) instead
    of via nx.from_numpy_array(Wpos) preserves that -- the dense zeroed
    matrix alone can't distinguish "zeroed negative edge" from "no edge",
    both read as 0, so from_numpy_array would silently drop those edges
    from the MST graph instead of keeping them at weight 0 (caught by
    direct re-derivation before trusting any result off it, not by a test
    case -- the smoke tests happened to only exercise the unweighted
    branch)."""
    m = W.shape[0]
    n_edges = len(edges)
    weighted = any(w != 1 for (_, _, w) in edges)
    if not weighted:
        return n_edges / 2 + (m - 1) / 4
    g = nx.Graph()
    g.add_nodes_from(range(m))
    for i, j, w in edges:
        g.add_edge(i, j, weight=max(w, 0.0))
    total_weight = float(sum(d["weight"] for _, _, d in g.edges(data=True)))
    mst = nx.minimum_spanning_tree(g, weight="weight")
    mst_weight = float(sum(d["weight"] for _, _, d in mst.edges(data=True)))
    return total_weight / 2 + mst_weight / 4


def sciorilli_loss(pi: torch.Tensor, edges_i: torch.Tensor, edges_j: torch.Tensor,
                    edges_w: torch.Tensor, alpha: float, beta: float, nu: float, m: int) -> torch.Tensor:
    """pi: (batch, m). Returns (batch,) loss, Section A's exact formula."""
    t = torch.tanh(alpha * pi)
    surrogate = (edges_w * t[:, edges_i] * t[:, edges_j]).sum(dim=1)
    penalization = beta * nu * (t.pow(2).sum(dim=1) / m) ** 2
    return surrogate + penalization


def run_pce_direct(W: np.ndarray, n: int, k: int, depth: int, alpha: float, beta: float,
                    E_ref: float, n_restarts: int = 2, n_iter: int = 1500, lr: float = 0.1,
                    assignment_seed: int = 0, param_seed: int = 0):
    """Returns dict with per-restart and mean r_circuit/r_full, matching
    protocol 1.3's evaluation protocol exactly (train -> measure pi -> sign
    readout -> E_raw -> local search -> E_full -> ratios)."""
    m = W.shape[0]
    edges = _edge_list(W)
    nu = compute_nu(W, edges)
    assignment = random_assignment(n, k, m, seed=assignment_seed)
    gate_seq = build_sciorilli_gate_sequence(n, depth)
    n_params = len(gate_seq)

    edges_i = torch.tensor([e[0] for e in edges], dtype=torch.long)
    edges_j = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edges_w = torch.tensor([e[2] for e in edges], dtype=torch.float64)

    torch.manual_seed(param_seed)
    params = (torch.rand(n_restarts, n_params, dtype=torch.float64) * 2 - 1) * np.pi
    params.requires_grad_(True)
    opt = torch.optim.Adam([params], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    best_loss = torch.full((n_restarts,), float("inf"), dtype=torch.float64)
    best_params = params.detach().clone()

    loss_history = []
    for it in range(n_iter):
        opt.zero_grad()
        state = run_circuit_pytorch(params, gate_seq, n)
        pi = correlators(state, assignment, n)
        loss = sciorilli_loss(pi, edges_i, edges_j, edges_w, alpha, beta, nu, m)
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_params[improved] = params[improved]
        loss_history.append(loss.detach().mean().item())

    with torch.no_grad():
        state = run_circuit_pytorch(best_params, gate_seq, n)
        pi_final = correlators(state, assignment, n).numpy()

    results = []
    for s in range(n_restarts):
        x_raw = sign_readout(pi_final[s].copy())
        E_raw = cut_value(x_raw, W)
        x_full, E_full, n_rounds = local_search_1bit_2bit(x_raw, W, edges=[(e[0], e[1]) for e in edges])
        results.append(dict(seed=s, E_raw=E_raw, E_full=E_full, r_circuit=E_raw / E_ref, r_full=E_full / E_ref,
                             final_loss=float(best_loss[s]), local_search_rounds=n_rounds,
                             x_raw=x_raw.tolist(), x_full=x_full.tolist()))

    return dict(per_restart=results, mean_r_circuit=float(np.mean([r["r_circuit"] for r in results])),
                mean_r_full=float(np.mean([r["r_full"] for r in results])),
                std_r_full=float(np.std([r["r_full"] for r in results])),
                nu=nu, n_params=n_params, assignment=assignment, loss_history=loss_history,
                pi_final=pi_final)
