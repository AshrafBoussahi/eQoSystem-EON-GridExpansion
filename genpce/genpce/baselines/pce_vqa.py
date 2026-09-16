"""PCE-VQA baseline: the variational solver of Sciorilli et al. (Nat. Commun. 2025).

Ansatz (their Fig. 1 / Methods): a brickwork circuit whose layers alternate a single-qubit rotation
layer — all qubits rotated about the *same* axis, cycling X → Y → Z from layer to layer, one angle
per qubit — and a layer of partially-entangling Mølmer–Sørensen gates on brickwork pairs, each with
three variational parameters ``MS(θ, φ₁, φ₂) = exp(-i θ/2 · σ_{φ₁} ⊗ σ_{φ₂})`` with
``σ_φ = cos φ X + sin φ Y``.

Two numerically identical execution paths are provided:

* :meth:`BrickworkMSAnsatz.qiskit_circuit` — the canonical Qiskit circuit (used for validation,
  shot/noise simulation and hardware);
* :class:`TorchSimulator` — a differentiable statevector simulator so that the relaxed loss can be
  minimised with Adam via autograd, as in the original work (they used Qibo/TensorFlow). A unit
  test asserts both paths agree to machine precision.

Quantum-resource accounting: autodiff has no hardware analogue, so for every Adam step we also
record the parameter-shift-equivalent cost ``2·N_p + 1`` circuit executions (× 3 settings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from qiskit import QuantumCircuit

from genpce.pce.correlators import CorrelatorSet, decode_signs
from genpce.pce.loss import RelaxedLossParams, relaxed_loss_torch
from genpce.problems.maxcut import MaxCutInstance, one_pass_bit_swap

__all__ = ["BrickworkMSAnsatz", "TorchSimulator", "PCEVQAResult", "run_pce_vqa"]

_AXES = ("x", "y", "z")


# ---------------------------------------------------------------------- ansatz
@dataclass(frozen=True)
class BrickworkMSAnsatz:
    """Sciorilli-style brickwork ansatz on ``n`` qubits with ``layers`` (rotation + MS) layers."""

    n: int
    layers: int

    def pairs(self, layer: int) -> list[tuple[int, int]]:
        start = layer % 2
        return [(q, q + 1) for q in range(start, self.n - 1, 2)]

    @property
    def num_params(self) -> int:
        return sum(self.n + 3 * len(self.pairs(layer)) for layer in range(self.layers))

    @property
    def num_two_qubit_gates(self) -> int:
        return sum(len(self.pairs(layer)) for layer in range(self.layers))

    @property
    def num_one_qubit_gates(self) -> int:
        return self.n * self.layers

    def _iter_layers(self, params: np.ndarray | torch.Tensor):
        """Yield ``(axis, rot_angles, [(q0, q1, θ, φ1, φ2), ...])`` per layer."""
        p = 0
        for layer in range(self.layers):
            axis = _AXES[layer % 3]
            rot = params[p : p + self.n]
            p += self.n
            ms = []
            for q0, q1 in self.pairs(layer):
                ms.append((q0, q1, params[p], params[p + 1], params[p + 2]))
                p += 3
            yield axis, rot, ms

    def qiskit_circuit(self, params: np.ndarray, name: str = "pce_vqa") -> QuantumCircuit:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.num_params,):
            raise ValueError(f"expected {self.num_params} parameters, got {params.shape}")
        qc = QuantumCircuit(self.n, name=name)
        for axis, rot, ms in self._iter_layers(params):
            for q in range(self.n):
                getattr(qc, f"r{axis}")(float(rot[q]), q)
            for q0, q1, th, ph1, ph2 in ms:
                # MS(θ, φ1, φ2) = [Rz(φ1) ⊗ Rz(φ2)] · RXX(θ) · [Rz(-φ1) ⊗ Rz(-φ2)]
                qc.rz(-float(ph1), q0)
                qc.rz(-float(ph2), q1)
                qc.rxx(float(th), q0, q1)
                qc.rz(float(ph1), q0)
                qc.rz(float(ph2), q1)
        return qc

    def random_params(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        return rng.uniform(-pi, pi, size=self.num_params) * scale


# ---------------------------------------------------------------------- torch simulator
def _fwht_torch(a: torch.Tensor) -> torch.Tensor:
    """Unnormalised WHT along the last axis (differentiable)."""
    length = a.shape[-1]
    lead = a.shape[:-1]
    h = 1
    while h < length:
        a = a.reshape(lead + (length // (2 * h), 2, h))
        x, y = a[..., 0, :], a[..., 1, :]
        a = torch.stack([x + y, x - y], dim=-2).reshape(lead + (length,))
        h *= 2
    return a


class TorchSimulator:
    """Differentiable statevector simulation of :class:`BrickworkMSAnsatz` + PCE readout.

    The state tensor has shape ``(2,)*n`` with axis ``a`` ↔ qubit ``n-1-a``, so that flattening
    reproduces Qiskit's little-endian amplitude ordering.
    """

    def __init__(self, ansatz: BrickworkMSAnsatz, cset: CorrelatorSet, dtype=torch.complex128):
        self.ansatz = ansatz
        self.cset = cset
        self.n = ansatz.n
        self.dtype = dtype
        idx = torch.arange(1 << self.n)
        pop = torch.tensor([bin(int(i)).count("1") for i in idx])
        self._y_phase = (-1j) ** pop.to(dtype)
        self._family = torch.as_tensor(cset.family, dtype=torch.long)
        self._mask = torch.as_tensor(cset.mask, dtype=torch.long)

    # -- gates -----------------------------------------------------------------
    def _axis(self, q: int) -> int:
        return self.n - 1 - q

    def _apply_1q(self, psi: torch.Tensor, mat: torch.Tensor, q: int) -> torch.Tensor:
        a = self._axis(q)
        psi = torch.tensordot(mat, psi, dims=([1], [a]))
        return torch.movedim(psi, 0, a)

    @staticmethod
    def _rot(axis: str, theta: torch.Tensor, dtype) -> torch.Tensor:
        c = torch.cos(theta / 2).to(dtype)
        s = torch.sin(theta / 2).to(dtype)
        if axis == "x":
            return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])
        if axis == "y":
            return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        e = torch.exp(-1j * theta / 2).to(dtype)
        zero = torch.zeros((), dtype=dtype)
        return torch.stack([torch.stack([e, zero]), torch.stack([zero, e.conj()])])

    def _apply_rxx(self, psi: torch.Tensor, theta: torch.Tensor, q0: int, q1: int) -> torch.Tensor:
        a0, a1 = self._axis(q0), self._axis(q1)
        flipped = torch.flip(psi, dims=(a0, a1))  # (X ⊗ X) |psi>
        return torch.cos(theta / 2).to(self.dtype) * psi - 1j * torch.sin(theta / 2).to(self.dtype) * flipped

    # -- forward ---------------------------------------------------------------
    def statevector(self, params: torch.Tensor) -> torch.Tensor:
        psi = torch.zeros((2,) * self.n, dtype=self.dtype)
        psi[(0,) * self.n] = 1.0
        for axis, rot, ms in self.ansatz._iter_layers(params):
            for q in range(self.n):
                psi = self._apply_1q(psi, self._rot(axis, rot[q], self.dtype), q)
            for q0, q1, th, ph1, ph2 in ms:
                psi = self._apply_1q(psi, self._rot("z", -ph1, self.dtype), q0)
                psi = self._apply_1q(psi, self._rot("z", -ph2, self.dtype), q1)
                psi = self._apply_rxx(psi, th, q0, q1)
                psi = self._apply_1q(psi, self._rot("z", ph1, self.dtype), q0)
                psi = self._apply_1q(psi, self._rot("z", ph2, self.dtype), q1)
        return psi.reshape(-1)

    def correlators(self, params: torch.Tensor) -> torch.Tensor:
        psi = self.statevector(params)
        length = psi.shape[-1]
        p_z = psi.abs() ** 2
        p_x = _fwht_torch(psi).abs() ** 2 / length
        p_y = _fwht_torch(psi * self._y_phase).abs() ** 2 / length
        parities = _fwht_torch(torch.stack([p_x, p_y, p_z], dim=0))  # (3, 2**n)
        return parities[self._family, self._mask]


# ---------------------------------------------------------------------- optimisation
@dataclass
class PCEVQAResult:
    params: np.ndarray
    corr: np.ndarray
    x: np.ndarray
    cut: float
    x_ls: np.ndarray
    cut_ls: float
    loss_history: list[float]
    steps: int
    function_evals: int
    circuit_executions_equiv: int  # parameter-shift-equivalent circuit executions (per setting)
    seconds: float
    extra: dict = field(default_factory=dict)


def run_pce_vqa(
    inst: MaxCutInstance,
    cset: CorrelatorSet,
    ansatz: BrickworkMSAnsatz,
    *,
    optimizer: str = "adam",
    lr: float = 0.05,
    max_steps: int = 5000,
    stop_window: int = 50,
    stop_tol: float = 0.01,
    seed: int = 0,
    loss_params: RelaxedLossParams | None = None,
    params0: np.ndarray | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> PCEVQAResult:
    """Train the PCE-VQA baseline on one instance from one random initialisation.

    Args:
        optimizer: ``"adam"`` (autodiff; Sciorilli's choice for large systems), ``"slsqp"`` or
            ``"cobyla"`` (SciPy, gradient-free / finite differences).
        stop_window, stop_tol: Sciorilli's rule — stop after ``stop_window`` steps whose cumulative
            loss improvement is below ``stop_tol`` (Adam only).
    """
    rng = np.random.default_rng(seed)
    if loss_params is None:
        loss_params = RelaxedLossParams.sciorilli(inst, cset.n, cset.k)
    sim = TorchSimulator(ansatz, cset)
    p0 = ansatz.random_params(rng) if params0 is None else np.asarray(params0, dtype=np.float64)
    t0 = perf_counter()
    history: list[float] = []
    evals = 0

    def loss_np(p: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        with torch.no_grad():
            c = sim.correlators(torch.as_tensor(p, dtype=torch.float64))
            return float(relaxed_loss_torch(c, inst, loss_params))

    if optimizer == "adam":
        theta = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        window_start = None
        steps = 0
        for step in range(max_steps):
            opt.zero_grad()
            loss = relaxed_loss_torch(sim.correlators(theta), inst, loss_params)
            loss.backward()
            opt.step()
            value = float(loss.detach())
            history.append(value)
            steps = step + 1
            evals += 1
            if callback:
                callback(step, value)
            if len(history) > stop_window:
                if history[-1 - stop_window] - history[-1] < stop_tol:
                    break
        params = theta.detach().numpy().copy()
        circuit_equiv = steps * (2 * ansatz.num_params + 1)
    else:
        from scipy.optimize import minimize

        method = {"slsqp": "SLSQP", "cobyla": "COBYLA"}[optimizer]
        res = minimize(loss_np, p0, method=method, options={"maxiter": max_steps})
        params = np.asarray(res.x, dtype=np.float64)
        steps = int(getattr(res, "nit", 0))
        history = [float(res.fun)]
        circuit_equiv = evals

    with torch.no_grad():
        corr = sim.correlators(torch.as_tensor(params)).numpy()
    x = decode_signs(corr)
    cut = inst.cut_value(x)
    x_ls = one_pass_bit_swap(inst, x)
    cut_ls = inst.cut_value(x_ls)
    return PCEVQAResult(
        params=params,
        corr=corr,
        x=x,
        cut=cut,
        x_ls=x_ls,
        cut_ls=cut_ls,
        loss_history=history,
        steps=steps,
        function_evals=evals,
        circuit_executions_equiv=circuit_equiv,
        seconds=perf_counter() - t0,
        extra={"optimizer": optimizer, "lr": lr, "alpha": loss_params.alpha, "beta": loss_params.beta, "nu": loss_params.nu},
    )
