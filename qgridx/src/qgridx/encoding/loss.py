"""PCE loss (paper Eq. 2 & 5), generalized from pure MaxCut to a general QUBO
via the standard x=(1+s)/2 QUBO->Ising map, and evaluated batched (one
(h_i, J_ij) system per training instance) over a shared Pauli-string
assignment for the whole batch.
"""
import torch

from qgridx.encoding.simulator import pauli_expectation


def qubo_to_ising(c: torch.Tensor, Q: torch.Tensor):
    """c: (batch, m), Q: (batch, m, m) symmetric zero-diagonal. Returns (h, J)
    with H(s) = const + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j, exactly
    equivalent to cost(x) = c^T x + x^T Q x under x_i = (1+s_i)/2."""
    h = c / 2 + 0.5 * Q.sum(dim=-1)
    J = Q / 2  # full symmetric matrix; pairwise sum_{i<j} J_ij s_i s_j = 0.5 * s^T J s
    return h, J


def correlators(state: torch.Tensor, assignment, n: int) -> torch.Tensor:
    """assignment: list of (qubits, axis), len m. Returns (batch, m) real tensor
    of <Pi> for each i."""
    cols = [pauli_expectation(state, qubits, axis, n) for (qubits, axis) in assignment]
    return torch.stack(cols, dim=1)


def pce_loss(pi_vals: torch.Tensor, h: torch.Tensor, J: torch.Tensor, alpha: float,
             beta: float = 0.5, nu: torch.Tensor | None = None,
             margin_weight: torch.Tensor | None = None) -> torch.Tensor:
    """pi_vals: (batch, m) correlator expectations. h: (batch,m). J: (batch,m,m).
    nu: (batch,) scale estimate for the regularizer (defaults to sum|h|+sum|J|).
    margin_weight: optional (batch, m) per-decision regularization-target weight
    (Phase-2 margin-weighted regularizer, P5); defaults to uniform (all ones).
    """
    t = torch.tanh(alpha * pi_vals)  # (batch, m)
    linear_term = (h * t).sum(dim=1)
    pairwise_term = 0.5 * torch.einsum('bi,bij,bj->b', t, J, t)
    surrogate = linear_term + pairwise_term

    m = pi_vals.shape[1]
    if nu is None:
        nu = h.abs().sum(dim=1) + J.abs().sum(dim=(1, 2)) + 1e-9
    if margin_weight is None:
        margin_weight = torch.ones_like(t)
    reg = beta * (nu / m) * (margin_weight * t.pow(2)).sum(dim=1)
    return surrogate + reg, surrogate, reg
