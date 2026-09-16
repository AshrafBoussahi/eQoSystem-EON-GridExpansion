"""Batched statevector simulator (torch, complex128). Every instance in a
batch gets independent circuit parameters (per-instance variational training,
vanilla PCE), but the same circuit *structure* -- so a whole batch of
per-instance PCE solves runs as one vectorized torch computation instead of a
Python loop, which is what makes a 500-instance library and multiple ablation
sweeps tractable at n <= 8 qubits.
"""
import torch

CDTYPE = torch.complex128


def zero_state(batch: int, n: int, device="cpu") -> torch.Tensor:
    dim = 2 ** n
    state = torch.zeros((batch, dim), dtype=CDTYPE, device=device)
    state[:, 0] = 1.0
    return state


def _rot_matrix(axis: str, theta: torch.Tensor) -> torch.Tensor:
    """theta: (batch,) real. Returns (batch, 2, 2) complex rotation matrices."""
    c = torch.cos(theta / 2).to(CDTYPE)
    s = torch.sin(theta / 2).to(CDTYPE)
    i = torch.complex(torch.zeros_like(theta), torch.ones_like(theta))
    zero = torch.zeros_like(c)
    if axis == "X":
        return torch.stack([torch.stack([c, -i * s], -1),
                             torch.stack([-i * s, c], -1)], -2)
    if axis == "Y":
        return torch.stack([torch.stack([c, -s], -1),
                             torch.stack([s, c], -1)], -2)
    if axis == "Z":
        eneg = torch.exp(-i * theta / 2)
        epos = torch.exp(i * theta / 2)
        return torch.stack([torch.stack([eneg, zero.to(CDTYPE)], -1),
                             torch.stack([zero.to(CDTYPE), epos], -1)], -2)
    raise ValueError(axis)


def _pauli_matrix(axis: str, device="cpu") -> torch.Tensor:
    if axis == "X":
        return torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=device)
    if axis == "Y":
        return torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE, device=device)
    if axis == "Z":
        return torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE, device=device)
    raise ValueError(axis)


def _rxx_yy_zz_matrix(axis: str, theta: torch.Tensor) -> torch.Tensor:
    """exp(-i theta/2 * P⊗P) for P in {X,Y,Z}. theta: (batch,). Returns (batch,4,4)."""
    batch = theta.shape[0]
    device = theta.device
    P = _pauli_matrix(axis, device)
    PP = torch.kron(P, P)  # (4,4)
    eye = torch.eye(4, dtype=CDTYPE, device=device)
    c = torch.cos(theta / 2).to(CDTYPE).view(batch, 1, 1)
    s = torch.sin(theta / 2).to(CDTYPE).view(batch, 1, 1)
    i = 1j
    return c * eye.unsqueeze(0) - i * s * PP.unsqueeze(0)


def _broadcast_gate(gate: torch.Tensor, batch: int) -> torch.Tensor:
    if gate.dim() == 2:
        gate = gate.unsqueeze(0)
    if gate.shape[0] == 1 and batch > 1:
        gate = gate.expand(batch, *gate.shape[1:])
    return gate


def apply_1q(state: torch.Tensor, gate: torch.Tensor, qubit: int, n: int) -> torch.Tensor:
    """state: (batch, 2^n). gate: (batch, 2, 2), (1, 2, 2), or (2,2). qubit: 0-indexed."""
    batch, dim = state.shape
    pre = 2 ** qubit
    post = 2 ** (n - qubit - 1)
    s = state.view(batch, pre, 2, post)
    gate = _broadcast_gate(gate, batch)
    out = torch.einsum('bij,bxjy->bxiy', gate, s)
    return out.reshape(batch, dim)


def apply_2q(state: torch.Tensor, gate: torch.Tensor, q1: int, q2: int, n: int) -> torch.Tensor:
    """q1 < q2, 0-indexed. gate: (batch, 4, 4) or (4,4)."""
    assert q1 < q2
    batch, dim = state.shape
    pre = 2 ** q1
    mid = 2 ** (q2 - q1 - 1)
    post = 2 ** (n - q2 - 1)
    s = state.view(batch, pre, 2, mid, 2, post)
    gate = _broadcast_gate(gate, batch)
    g = gate.reshape(batch, 2, 2, 2, 2)  # (b, i1, i2, j1, j2): rows i1,i2 <- cols j1,j2
    out = torch.einsum('bpqrs,bxrmsy->bxpmqy', g, s)
    return out.reshape(batch, dim)


def pauli_expectation(state: torch.Tensor, qubits, axis: str, n: int) -> torch.Tensor:
    """<psi| P_{q1} P_{q2} ... P_{qk} |psi>, same axis on each qubit in `qubits`.
    Returns (batch,) real tensor."""
    P = _pauli_matrix(axis, state.device)
    ps = state
    for q in qubits:
        ps = apply_1q(ps, P, q, n)
    val = torch.einsum('bi,bi->b', state.conj(), ps)
    return val.real
