"""Operator pools ("vocabularies") for generative circuit synthesis.

A pool is an ordered list of :class:`Token` objects; a circuit is a sequence of token ids. Tokens
are *fixed* unitaries (no free parameters), exactly as in the generative quantum eigensolver
(Nakaji et al., 2024): every trainable parameter lives in the classical generative model.

Pools may carry a **template** that fixes the entangling structure by hardware connectivity — a
brickwork of ``CZ`` layers on a linear chain inserted after every ``n`` tokens — so that the
generator writes the single-qubit content (axis, angle, identity) while the two-qubit structure is
the one the device executes natively. Without a template, ``CZ`` gates are ordinary tokens.

Rotations by multiples of ``π/2`` are Clifford; all other angles are non-Clifford, which lets us
count the "non-Clifford content" of a generated circuit directly from its token ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit

__all__ = ["Token", "Pool", "native_chain_pool", "pauli_rotation_pool"]

_CLIFFORD_TOL = 1e-9


@dataclass(frozen=True)
class Token:
    """One vocabulary entry: a fixed gate on fixed qubits.

    ``qubits`` may be empty for *slot* tokens, whose target qubit is determined by the position in
    the sequence (see :attr:`Pool.slot`).
    """

    kind: str  # "id", "rx", "ry", "rz", "cz", "rzz", "rxx", "ryy", "h", "sx", "x"
    qubits: tuple[int, ...]
    theta: float = 0.0

    @property
    def label(self) -> str:
        q = ",".join(map(str, self.qubits)) if self.qubits else "slot"
        if self.kind in ("id", "cz", "h", "sx", "x"):
            return f"{self.kind}({q})"
        return f"{self.kind}({q};{self.theta / pi:+.4g}π)"

    @property
    def is_two_qubit(self) -> bool:
        return self.kind in ("cz", "rzz", "rxx", "ryy")

    @property
    def is_clifford(self) -> bool:
        if self.kind in ("id", "cz", "h", "sx", "x"):
            return True
        ratio = self.theta / (pi / 2)
        return abs(ratio - round(ratio)) < _CLIFFORD_TOL

    def apply(self, qc: QuantumCircuit, qubits: tuple[int, ...] | None = None) -> None:
        """Append this token's gate to ``qc`` (``qubits`` overrides the token's own targets)."""
        k = self.kind
        q = self.qubits if qubits is None else qubits
        if k == "id":
            return
        if k == "rx":
            qc.rx(self.theta, q[0])
        elif k == "ry":
            qc.ry(self.theta, q[0])
        elif k == "rz":
            qc.rz(self.theta, q[0])
        elif k == "h":
            qc.h(q[0])
        elif k == "sx":
            qc.sx(q[0])
        elif k == "x":
            qc.x(q[0])
        elif k == "cz":
            qc.cz(q[0], q[1])
        elif k == "rzz":
            qc.rzz(self.theta, q[0], q[1])
        elif k == "rxx":
            qc.rxx(self.theta, q[0], q[1])
        elif k == "ryy":
            qc.ryy(self.theta, q[0], q[1])
        else:
            raise ValueError(f"unknown token kind {k!r}")


@dataclass(frozen=True)
class Pool:
    """An ordered vocabulary of tokens on ``n`` qubits, with an optional entangling template.

    Attributes:
        n: Number of qubits.
        tokens: The vocabulary.
        name: Human-readable name.
        template: ``None`` (no fixed entanglers) or ``"brickwork_cz"`` — a layer of ``CZ`` gates on
            alternating nearest-neighbour pairs of the chain is appended after every
            ``template_period`` tokens.
        template_period: Number of tokens between entangling layers (default ``n``).
        slot: If True, tokens carry no qubit index and position ``t`` in the sequence acts on
            qubit ``t mod n`` (one rotation "slot" per qubit per layer, as in a brickwork ansatz).
        prefix: Optional gates prepended to every circuit (a reference-state preparation), given
            as ``(kind, qubit, theta)`` triples.
    """

    n: int
    tokens: tuple[Token, ...]
    name: str = "pool"
    template: str | None = None
    template_period: int | None = None
    slot: bool = False
    prefix: tuple[tuple[str, int, float], ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def period(self) -> int:
        return self.template_period or self.n

    def _entangling_layer(self, qc: QuantumCircuit, layer: int) -> int:
        if self.template == "brickwork_cz":
            pairs = [(q, q + 1) for q in range(layer % 2, self.n - 1, 2)]
            for a, b in pairs:
                qc.cz(a, b)
            return len(pairs)
        raise ValueError(f"unknown template {self.template!r}")

    def to_circuit(self, ids: Sequence[int], name: str | None = None) -> QuantumCircuit:
        """Build the Qiskit circuit for a token-id sequence (prefix, tokens, template layers)."""
        qc = QuantumCircuit(self.n, name=name or "genpce")
        for kind, q, theta in self.prefix:
            Token(kind, (q,), theta).apply(qc)
        layer = 0
        for t, j in enumerate(ids):
            tok = self.tokens[int(j)]
            tok.apply(qc, (t % self.n,) if self.slot and tok.kind != "id" else None)
            if self.template and (t + 1) % self.period == 0:
                self._entangling_layer(qc, layer)
                layer += 1
        return qc

    def to_circuits(self, id_batch: np.ndarray | Sequence[Sequence[int]]) -> list[QuantumCircuit]:
        return [self.to_circuit(row) for row in id_batch]

    # ------------------------------------------------------------------ metrics
    def template_two_qubit_gates(self, length: int) -> int:
        """Number of two-qubit gates contributed by the template for a sequence of ``length``."""
        if not self.template:
            return 0
        layers = length // self.period
        return sum(len(range(layer % 2, self.n - 1, 2)) for layer in range(layers))

    def gate_counts(self, ids: Sequence[int]) -> dict[str, int]:
        """Gate-count summary of a token sequence (identities excluded from ``gates``)."""
        toks = [self.tokens[int(j)] for j in ids]
        real = [t for t in toks if t.kind != "id"]
        two_q = sum(t.is_two_qubit for t in real) + self.template_two_qubit_gates(len(toks))
        return {
            "tokens": len(toks),
            "gates": len(real) + self.template_two_qubit_gates(len(toks)),
            "two_qubit": two_q,
            "non_clifford": sum(not t.is_clifford for t in real),
            "identity": len(toks) - len(real),
        }

    def depth(self, ids: Sequence[int]) -> int:
        return int(self.to_circuit(ids).depth())


def native_chain_pool(
    n: int,
    angles: Sequence[float] = (pi / 2, pi / 4, pi / 8, pi / 16),
    *,
    axes: Sequence[str] = ("rx", "ry", "rz"),
    template: str | None = "brickwork_cz",
    slot: bool = True,
    entangler_tokens: bool = False,
    include_identity: bool = True,
    prefix: Sequence[tuple[str, int, float]] = (),
) -> Pool:
    """IBM-native pool on a linear chain of ``n`` qubits.

    Default (``template="brickwork_cz", slot=True``): the vocabulary is ``R_a(±θ)`` for each axis
    and angle plus identity (``2·|angles|·|axes| + 1`` tokens, e.g. 25); position ``t`` acts on
    qubit ``t mod n``; a brickwork ``CZ`` layer follows every ``n`` tokens. With ``slot=False`` each
    rotation token carries its qubit index (``2·|angles|·|axes|·n + 1`` tokens). With
    ``entangler_tokens=True`` (and typically ``template=None``) ``CZ`` on each chain edge is added to
    the vocabulary as ordinary tokens.
    """
    toks: list[Token] = []
    if include_identity:
        toks.append(Token("id", ()))
    targets = [()] if slot else [(q,) for q in range(n)]
    for tq in targets:
        for a in axes:
            for th in angles:
                toks.append(Token(a, tq, float(th)))
                toks.append(Token(a, tq, -float(th)))
    if entangler_tokens:
        for q in range(n - 1):
            toks.append(Token("cz", (q, q + 1)))
    kind = "slot" if slot else "free"
    tmpl = template or "none"
    return Pool(
        n=n, tokens=tuple(toks), name=f"native_chain[n={n},{kind},{tmpl}]",
        template=template, slot=slot, prefix=tuple(prefix),
    )


def pauli_rotation_pool(
    n: int,
    angles: Sequence[float] = (pi / 2, pi / 4, pi / 8),
    *,
    two_qubit: Sequence[str] = ("rzz", "rxx", "ryy"),
    all_to_all: bool = False,
    include_identity: bool = True,
) -> Pool:
    """GQE-style pool of Pauli rotations ``exp(-iθP/2)`` for 1-local and 2-local ``P`` (no template)."""
    toks: list[Token] = []
    if include_identity:
        toks.append(Token("id", ()))
    for q in range(n):
        for a in ("rx", "ry", "rz"):
            for th in angles:
                toks.append(Token(a, (q,), float(th)))
                toks.append(Token(a, (q,), -float(th)))
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)] if all_to_all else [
        (q, q + 1) for q in range(n - 1)
    ]
    for a, b in pairs:
        for kind in two_qubit:
            for th in angles:
                toks.append(Token(kind, (a, b), float(th)))
                toks.append(Token(kind, (a, b), -float(th)))
    return Pool(n=n, tokens=tuple(toks), name=f"pauli_rot[n={n},a2a={all_to_all}]")
