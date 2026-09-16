"""The relaxed PCE loss of Sciorilli et al. (Eq. 2 and Eq. 5).

    L(c) = Σ_{(i,j)∈E} W_ij tanh(α c_i) tanh(α c_j) + β ν [ (1/m) Σ_i tanh(α c_i) ]²

with ``c_i = <Π_i>``, ``α ≈ n^{⌊k/2⌋}`` (restores the non-linear regime of ``tanh`` because k-body
correlators are polynomially small), ``β = 1/2`` and ``ν`` the a-priori cut lower bound
(Edwards–Erdős / Poljak–Turzík; see :meth:`MaxCutInstance.regularisation_scale`). Minimising
``L`` pushes ``sgn(c)`` towards large cuts. Provided in NumPy (for gradient-free optimisers and
reward shaping) and in PyTorch (for autodiff-based PCE-VQA baselines).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genpce.problems.maxcut import MaxCutInstance

__all__ = ["RelaxedLossParams", "relaxed_loss_np", "relaxed_loss_torch", "default_alpha"]


def default_alpha(n: int, k: int) -> float:
    """Sciorilli et al.'s rescaling ``α ≈ n^{⌊k/2⌋}``."""
    return float(n ** (k // 2))


@dataclass(frozen=True)
class RelaxedLossParams:
    alpha: float
    beta: float
    nu: float

    @classmethod
    def sciorilli(cls, inst: MaxCutInstance, n: int, k: int, beta: float = 0.5) -> "RelaxedLossParams":
        return cls(alpha=default_alpha(n, k), beta=beta, nu=inst.regularisation_scale())


def relaxed_loss_np(corr: np.ndarray, inst: MaxCutInstance, params: RelaxedLossParams) -> np.ndarray:
    """Relaxed loss for a (batch of) correlator vector(s) ``(..., m)``. Lower is better."""
    t = np.tanh(params.alpha * np.asarray(corr))
    i, j = inst.edges[:, 0], inst.edges[:, 1]
    edge_term = np.sum(inst.weights * t[..., i] * t[..., j], axis=-1)
    reg = params.beta * params.nu * np.mean(t, axis=-1) ** 2
    return edge_term + reg


def relaxed_loss_torch(corr, inst: MaxCutInstance, params: RelaxedLossParams):
    """PyTorch version of :func:`relaxed_loss_np` (differentiable in ``corr``)."""
    import torch

    t = torch.tanh(params.alpha * corr)
    edges = torch.as_tensor(inst.edges, dtype=torch.long, device=t.device)
    w = torch.as_tensor(inst.weights, dtype=t.dtype, device=t.device)
    edge_term = torch.sum(w * t[..., edges[:, 0]] * t[..., edges[:, 1]], dim=-1)
    reg = params.beta * params.nu * torch.mean(t, dim=-1) ** 2
    return edge_term + reg
