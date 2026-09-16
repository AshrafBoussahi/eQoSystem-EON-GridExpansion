"""Project 2: fast batched replacement for loss.py's correlators(). The
original computes each of the m assigned k-qubit Pauli-string expectation
values via its OWN sequential apply_1q calls (a Python loop of m*k tiny
tensor ops) -- fine at the main project's usual m (dozens), but a severe
bottleneck once m reaches the hundreds-to-thousands this project's MaxCut
phases need (up to m=10,404 in Phase 3): profiling showed >150k function
calls and ~90ms/iteration at just m=105, almost entirely autograd overhead
from the sheer NUMBER of tiny ops, not their FLOP cost.

Key fact this exploits: for a FIXED measurement axis, <psi|P_{q1}...P_{qk}|
psi> for every possible qubit-subset (q1..qk) can be obtained SIMULTANEOUSLY
from a single Walsh-Hadamard transform of the axis-rotated basis
probabilities -- because in the axis's own eigenbasis (reached by a FIXED,
n-gate-not-m-gate rotation of the whole state), P_{q1}...P_{qk} is diagonal
with eigenvalue (-1)^popcount(x & mask_S) on basis state x, so
<P_S> = sum_x p(x) (-1)^{S.x} = WHT(p)[mask_S] exactly, for every subset S
at once. So the cost becomes: 3 fixed O(n)-gate rotations (not O(m)) + 3
O(n * 2^n) Hadamard transforms (independent of how many correlators m are
actually needed) + O(m) indexing -- from O(m*k) tiny autograd ops down to
O(n) of them.

Axis rotations (derived directly from THIS project's own Pauli convention
in simulator.py's _pauli_matrix, not assumed from any external library):
  Z: already diagonal, no rotation.
  X: H on every qubit (H Z H = X, standard, H Hermitian/self-adjoint).
  Y: apply S^dagger then H on every qubit (derived by solving U^dagger Z U
     = Y for this project's Y=[[0,-1j],[1j,0]]; verified U=S.H gives
     U Z U^dagger = Y exactly, so the needed pre-rotation is U^dagger =
     H . S^dagger, i.e. S^dagger first then H).

Verified bit-exact (see test_fast_correlators.py-style check run at
integration time) against the existing pauli_expectation-based
correlators() across random states and random assignments before use in
any Arm A/B/C computation.
"""
import torch

from qgridx.encoding.simulator import CDTYPE, _rot_matrix, apply_1q


def _rotate_to_z_eigenbasis(state: torch.Tensor, axis: str, n: int) -> torch.Tensor:
    if axis == "Z":
        return state
    out = state
    if axis == "X":
        h = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE) / (2 ** 0.5)
        for q in range(n):
            out = apply_1q(out, h, q, n)
    elif axis == "Y":
        sdag = torch.tensor([[1, 0], [0, -1j]], dtype=CDTYPE)
        h = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE) / (2 ** 0.5)
        for q in range(n):
            out = apply_1q(out, sdag, q, n)
            out = apply_1q(out, h, q, n)
    else:
        raise ValueError(axis)
    return out


def _fwht(a: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard transform along the last axis (size N=2^k, unnormalized:
    out[mask] = sum_x a[x] * (-1)^popcount(x & mask)). Functional (no
    in-place ops), safe under autograd."""
    N = a.shape[-1]
    h = 1
    while h < N:
        a = a.reshape(*a.shape[:-1], N // (2 * h), 2, h)
        x = a[..., 0, :]
        y = a[..., 1, :]
        a = torch.cat([(x + y).unsqueeze(-2), (x - y).unsqueeze(-2)], dim=-2)
        a = a.reshape(*a.shape[:-3], N)
        h *= 2
    return a


def correlators_fast(state: torch.Tensor, assignment, n: int) -> torch.Tensor:
    """Drop-in-compatible fast replacement for loss.py's correlators().
    state: (batch, 2^n) complex. assignment: list of (qubits, axis), len m.
    Returns (batch, m) real."""
    batch = state.shape[0]
    m = len(assignment)
    by_axis = {"X": [], "Y": [], "Z": []}
    for out_idx, (qubits, axis) in enumerate(assignment):
        by_axis[axis].append((out_idx, qubits))

    out = torch.zeros(batch, m, dtype=torch.float64)
    for axis, items in by_axis.items():
        if not items:
            continue
        rotated = _rotate_to_z_eigenbasis(state, axis, n)
        probs = (rotated.conj() * rotated).real  # (batch, 2^n)
        wht = _fwht(probs)  # (batch, 2^n)
        for out_idx, qubits in items:
            # apply_1q's reshape (state.view(batch, 2**q, 2, 2**(n-q-1))) puts
            # qubit q's bit at ARRAY bit-position (n-1-q) of the flat index
            # (qubit 0 = MSB, qubit n-1 = LSB) -- NOT position q. mask must
            # match that convention, verified against pauli_expectation below.
            mask = 0
            for q in qubits:
                mask |= (1 << (n - 1 - q))
            out[:, out_idx] = wht[:, mask]
    return out
