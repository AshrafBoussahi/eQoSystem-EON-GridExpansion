"""B1: shot-noise simulator for the 3 commuting Pauli families. Samples N
measurement outcomes per family from the *exact* statevector (basis-rotate,
then sample computational-basis outcomes), and estimates every correlator in
that family from the *same* N shots -- so correlators sharing qubit support
are correctly correlated (not simulated as independent Gaussians).
"""
from dataclasses import dataclass

import numpy as np
import torch

from qgridx.encoding.families import all_strings
from qgridx.encoding.simulator import CDTYPE, apply_1q

_H = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE) / np.sqrt(2)
_SDAG = torch.tensor([[1, 0], [0, -1j]], dtype=CDTYPE)
_HSDAG = _H @ _SDAG  # maps Y-eigenbasis -> computational (Z) basis


def _rotate_to_z_basis(state: torch.Tensor, axis: str, n: int) -> torch.Tensor:
    if axis == "Z":
        return state
    gate = _H if axis == "X" else _HSDAG
    for q in range(n):
        state = apply_1q(state, gate, q, n)
    return state


def sample_spins(state: torch.Tensor, axis: str, n: int, N: int,
                  generator: torch.Generator) -> torch.Tensor:
    """Returns (batch, N, n) spins in {+1,-1}, sampled from N shots of a
    computational-basis measurement after rotating to `axis`."""
    rotated = _rotate_to_z_basis(state, axis, n)
    probs = (rotated.conj() * rotated).real.clamp(min=0)
    probs = probs / probs.sum(dim=1, keepdim=True)
    batch = probs.shape[0]
    idx = torch.multinomial(probs, N, replacement=True, generator=generator)  # (batch,N)
    bit_positions = torch.arange(n - 1, -1, -1)  # qubit 0 = MSB (matches apply_1q's reshape convention)
    bits = (idx.unsqueeze(-1) >> bit_positions) & 1  # (batch,N,n)
    spins = (1 - 2 * bits).to(torch.float64)
    return spins


def all_string_shot_estimates(state: torch.Tensor, n: int, k: int, N: int, seed: int):
    """Returns (mu, sigma_emp, strings): mu/sigma_emp shape (batch, capacity),
    strings = all_strings(n,k) in matching column order. One shot-sampling
    draw per axis-family (X,Y,Z), reused for every string in that family."""
    generator = torch.Generator().manual_seed(seed)
    spins_by_axis = {axis: sample_spins(state, axis, n, N, generator) for axis in ("X", "Y", "Z")}
    strings = all_strings(n, k)
    mus, sigmas = [], []
    for (qubits, axis) in strings:
        spins = spins_by_axis[axis]  # (batch, N, n)
        prod = spins[:, :, list(qubits)].prod(dim=-1)  # (batch, N)
        mus.append(prod.mean(dim=1))
        sigmas.append(prod.std(dim=1, unbiased=True))
    return torch.stack(mus, dim=1), torch.stack(sigmas, dim=1), strings


def gather_assignment_estimates(mu_all, sigma_all, strings, assignment):
    """assignment: shared list (len m) or per-instance list-of-lists.
    Returns (mu, sigma) each (batch, m), matching Sprint-1's gather trick."""
    str_index = {s: idx for idx, s in enumerate(strings)}
    if isinstance(assignment[0], list):
        idx = np.array([[str_index[qa] for qa in assignment[b]] for b in range(len(assignment))])
    else:
        idx = np.array([[str_index[qa] for qa in assignment]] * mu_all.shape[0])
    idx_t = torch.tensor(idx, dtype=torch.long)
    return torch.gather(mu_all, 1, idx_t), torch.gather(sigma_all, 1, idx_t)


@dataclass
class ShotEstimate:
    mu: np.ndarray        # (batch, m) empirical mean correlator
    sigma_emp: np.ndarray  # (batch, m) empirical std of the per-shot estimator
    sigma_theory: np.ndarray  # (batch, m) sqrt((1-mu_exact^2)/N), using the *exact* pi as reference


def estimate_shots(state: torch.Tensor, assignment, n: int, k: int, N: int, seed: int,
                    exact_pi: np.ndarray) -> ShotEstimate:
    mu_all, sigma_all, strings = all_string_shot_estimates(state, n, k, N, seed)
    mu, sigma_emp = gather_assignment_estimates(mu_all, sigma_all, strings, assignment)
    mu_np, sigma_np = mu.numpy(), sigma_emp.numpy()
    sigma_theory = np.sqrt(np.clip(1 - exact_pi ** 2, 0, 1) / N)
    return ShotEstimate(mu=mu_np, sigma_emp=sigma_np, sigma_theory=sigma_theory)


def sample_spins_depolarized(state: torch.Tensor, axis: str, n: int, N: int,
                              generator: torch.Generator, p: float) -> torch.Tensor:
    """Sprint 8 C3's global-depolarizing shot sampler, promoted here (Sprint
    15 PR-54) so it's reusable machinery, not experiment-script-local code.
    Shrinks the Born distribution to (1-p)*P_ideal + p*uniform before sampling
    -- exact for computational-basis statistics in any rotated axis, not an
    approximation (a global depolarizing channel's effect on any measurement
    is exactly this convex mixture with the maximally-mixed state)."""
    rotated = _rotate_to_z_basis(state, axis, n)
    probs = (rotated.conj() * rotated).real.clamp(min=0)
    probs = probs / probs.sum(dim=1, keepdim=True)
    if p > 0:
        probs = (1 - p) * probs + p / probs.shape[1]
    idx = torch.multinomial(probs, N, replacement=True, generator=generator)
    bit_positions = torch.arange(n - 1, -1, -1)
    bits = (idx.unsqueeze(-1) >> bit_positions) & 1
    spins = (1 - 2 * bits).to(torch.float64)
    return spins


def all_string_shot_estimates_depolarized(state: torch.Tensor, n: int, k: int, N: int,
                                           seed: int, p: float):
    """Depolarized counterpart to all_string_shot_estimates (Sprint 15 PR-54)."""
    generator = torch.Generator().manual_seed(seed)
    spins_by_axis = {axis: sample_spins_depolarized(state, axis, n, N, generator, p) for axis in ("X", "Y", "Z")}
    strings = all_strings(n, k)
    mus = []
    for (qubits, axis) in strings:
        spins = spins_by_axis[axis]
        prod = spins[:, :, list(qubits)].prod(dim=-1)
        mus.append(prod.mean(dim=1))
    return torch.stack(mus, dim=1), strings


def sample_spins_device_noise(state: torch.Tensor, axis: str, n: int, N: int,
                               generator: torch.Generator, p_gate_effective: float,
                               p_spam: float) -> torch.Tensor:
    """Sprint 1 (paper de-risking), S1.C3: device-calibrated noise sampler.
    Two independent error sources, composed in the two places they actually
    act: (1) gate infidelity, applied the same way PR-54's global-depolarizing
    sampler does (convex mixture of the ideal Born distribution with uniform,
    exact for a single end-of-circuit depolarizing channel; `p_gate_effective`
    is the caller's responsibility to compose from separate 1q/2q per-gate
    rates via p_eff = 1-(1-p_1q)^n_1q*(1-p_2q)^n_2q, the same multiplicative-
    composition convention PR-54 already established for 2q-only); (2) SPAM
    (readout) error, applied AFTER sampling as an independent per-qubit
    bit-flip at rate p_spam -- the standard way vendors report and the
    literature models measurement infidelity, distinct in kind from gate
    error (it acts once, on the classical outcome, not on the quantum state
    evolution)."""
    rotated = _rotate_to_z_basis(state, axis, n)
    probs = (rotated.conj() * rotated).real.clamp(min=0)
    probs = probs / probs.sum(dim=1, keepdim=True)
    if p_gate_effective > 0:
        probs = (1 - p_gate_effective) * probs + p_gate_effective / probs.shape[1]
    idx = torch.multinomial(probs, N, replacement=True, generator=generator)
    bit_positions = torch.arange(n - 1, -1, -1)
    bits = (idx.unsqueeze(-1) >> bit_positions) & 1  # (batch, N, n), 0/1 long
    if p_spam > 0:
        flip = torch.rand(bits.shape, generator=generator) < p_spam
        bits = bits ^ flip.to(bits.dtype)
    spins = (1 - 2 * bits).to(torch.float64)
    return spins


def all_string_shot_estimates_device_noise(state: torch.Tensor, n: int, k: int, N: int, seed: int,
                                            p_gate_effective: float, p_spam: float):
    """Device-calibrated counterpart to all_string_shot_estimates (S1.C3)."""
    generator = torch.Generator().manual_seed(seed)
    spins_by_axis = {axis: sample_spins_device_noise(state, axis, n, N, generator, p_gate_effective, p_spam)
                      for axis in ("X", "Y", "Z")}
    strings = all_strings(n, k)
    mus = []
    for (qubits, axis) in strings:
        spins = spins_by_axis[axis]
        prod = spins[:, :, list(qubits)].prod(dim=-1)
        mus.append(prod.mean(dim=1))
    return torch.stack(mus, dim=1), strings
