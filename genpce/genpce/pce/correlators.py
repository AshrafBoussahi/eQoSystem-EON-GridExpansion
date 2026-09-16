"""Pauli-correlation encoding (PCE): correlator sets, readout and sign decoding.

PCE (Sciorilli et al., Nat. Commun. 16, 476 (2025)) encodes ``m`` binary variables into the signs
of ``m`` Pauli expectation values of an ``n``-qubit state, ``x_i = sgn(<Π_i>)``. We use the
"same-letter" k-body families

    Π^(k) = { permutations of X^{⊗k} ⊗ 1^{⊗(n-k)} } ∪ { ... Y ... } ∪ { ... Z ... },

which are three mutually commuting sets, so that *all* ``m ≤ 3·C(n, k)`` correlators are obtained
from just three measurement settings (all qubits in the X, Y or Z basis). Within a basis, every
``k``-body correlator is a parity of the measured bits, which we compute for all subsets at once with
a Walsh–Hadamard transform (see :mod:`genpce.pce.wht`).

This module is deliberately backend-agnostic: it consumes either an exact statevector (numpy
array, Qiskit little-endian ordering) or per-basis outcome counts (as returned by Qiskit
samplers / Aer). Nothing here depends on how the state was produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable, Mapping, Sequence

import numpy as np

from genpce.pce.wht import fwht

__all__ = [
    "FAMILIES",
    "CorrelatorSet",
    "basis_probabilities",
    "correlators_from_statevector",
    "correlators_from_probabilities",
    "correlators_from_counts",
    "sample_counts",
    "decode_signs",
]

FAMILIES: tuple[str, str, str] = ("X", "Y", "Z")


def _popcount(a: np.ndarray) -> np.ndarray:
    """Number of set bits per element (numpy >= 2.0 has ``bitwise_count``)."""
    return np.bitwise_count(a.astype(np.uint64)).astype(np.int64)


@dataclass(frozen=True)
class CorrelatorSet:
    """The ordered list of Pauli strings ``Π_1 ... Π_m`` used to encode ``m`` variables.

    Attributes:
        n: Number of qubits.
        orders: Correlator weights used (e.g. ``(3,)`` for a pure cubic encoding, ``(1, 2, 3)`` for
            all weights up to three).
        family: ``(m,)`` int array; 0 = X, 1 = Y, 2 = Z family of ``Π_i``.
        mask: ``(m,)`` int array; bitmask of the qubit subset supporting ``Π_i``.
    """

    n: int
    orders: tuple[int, ...]
    family: np.ndarray
    mask: np.ndarray

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(
        cls,
        n: int,
        k: int,
        m: int | None = None,
        *,
        all_orders_up_to_k: bool = False,
        assignment: str = "ordered",
        seed: int | None = None,
    ) -> "CorrelatorSet":
        """Build the correlator set for ``n`` qubits and order ``k``.

        Args:
            n: Number of qubits.
            k: Correlator order (weight of the Pauli strings).
            m: Number of variables to encode. Defaults to the full capacity.
            all_orders_up_to_k: If True, use all weights ``1..k`` (capacity ``3·Σ_w C(n, w)``);
                otherwise only weight ``k`` (capacity ``3·C(n, k)``).
            assignment: ``"ordered"`` (family-major, lexicographic subsets — reproducible and
                measurement-friendly) or ``"random"`` (seeded random permutation of the ordered list).
            seed: Seed for the random assignment.
        """
        if not (1 <= k <= n):
            raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
        orders = tuple(range(1, k + 1)) if all_orders_up_to_k else (k,)
        masks_by_order: list[int] = []
        for w in orders:
            for subset in combinations(range(n), w):
                masks_by_order.append(sum(1 << q for q in subset))
        masks_one_family = np.array(masks_by_order, dtype=np.int64)
        family = np.repeat(np.arange(3, dtype=np.int64), len(masks_one_family))
        mask = np.tile(masks_one_family, 3)
        capacity = len(mask)
        if assignment == "random":
            rng = np.random.default_rng(seed)
            perm = rng.permutation(capacity)
            family, mask = family[perm], mask[perm]
        elif assignment != "ordered":
            raise ValueError(f"unknown assignment {assignment!r}")
        if m is None:
            m = capacity
        if m > capacity:
            raise ValueError(f"m={m} exceeds capacity {capacity} for n={n}, orders={orders}")
        return cls(n=n, orders=orders, family=family[:m], mask=mask[:m])

    @staticmethod
    def capacity(n: int, k: int, *, all_orders_up_to_k: bool = False) -> int:
        """Maximum number of variables encodable with ``n`` qubits and order ``k``."""
        if all_orders_up_to_k:
            return 3 * sum(comb(n, w) for w in range(1, k + 1))
        return 3 * comb(n, k)

    @staticmethod
    def min_qubits(m: int, k: int, *, all_orders_up_to_k: bool = False) -> int:
        """Smallest ``n`` such that ``m`` variables fit."""
        n = k
        while CorrelatorSet.capacity(n, k, all_orders_up_to_k=all_orders_up_to_k) < m:
            n += 1
        return n

    # ------------------------------------------------------------------ properties
    @property
    def m(self) -> int:
        return int(len(self.mask))

    @property
    def k(self) -> int:
        return max(self.orders)

    def pauli_labels(self) -> list[str]:
        """Qiskit-style Pauli labels (qubit ``n-1`` leftmost) for each ``Π_i``."""
        labels = []
        for fam, mask in zip(self.family, self.mask):
            letter = FAMILIES[int(fam)]
            chars = [letter if (int(mask) >> q) & 1 else "I" for q in range(self.n)]
            labels.append("".join(reversed(chars)))
        return labels

    def support_weights(self) -> np.ndarray:
        """Weight (number of non-identity factors) of each ``Π_i``."""
        return _popcount(self.mask)


# ---------------------------------------------------------------------- readout
def _y_phase(n: int) -> np.ndarray:
    """Per-amplitude phase ``(-i)^{popcount(z)}`` implementing ``S†^{⊗n}`` before ``H^{⊗n}``."""
    idx = np.arange(1 << n, dtype=np.int64)
    return (-1j) ** _popcount(idx)


def basis_probabilities(psi: np.ndarray) -> np.ndarray:
    """Outcome probabilities in the X, Y and Z bases for a (batch of) statevector(s).

    Args:
        psi: ``(..., 2**n)`` complex array in Qiskit little-endian ordering.

    Returns:
        ``(..., 3, 2**n)`` real array ``p[..., b, z]`` for ``b`` = X, Y, Z.
    """
    psi = np.asarray(psi)
    length = psi.shape[-1]
    n = length.bit_length() - 1
    norm = 1.0 / length  # |H^{⊗n} psi|^2 = |WHT(psi)|^2 / 2^n
    p_z = np.abs(psi) ** 2
    p_x = np.abs(fwht(psi)) ** 2 * norm
    p_y = np.abs(fwht(psi * _y_phase(n))) ** 2 * norm
    return np.stack([p_x, p_y, p_z], axis=-2)


def correlators_from_probabilities(p_xyz: np.ndarray, cset: CorrelatorSet) -> np.ndarray:
    """Correlators ``<Π_i>`` from per-basis outcome probabilities.

    Args:
        p_xyz: ``(..., 3, 2**n)`` probabilities (exact or empirical) for the X, Y, Z bases.
        cset: Correlator set.

    Returns:
        ``(..., m)`` array of correlators in ``[-1, 1]``.
    """
    parities = fwht(p_xyz, axis=-1)  # (..., 3, 2**n): all parity correlators of each basis
    return parities[..., cset.family, cset.mask]


def correlators_from_statevector(psi: np.ndarray, cset: CorrelatorSet) -> np.ndarray:
    """Exact correlators ``<Π_i>`` for a (batch of) statevector(s)."""
    return correlators_from_probabilities(basis_probabilities(psi), cset)


def sample_counts(
    p_xyz: np.ndarray, shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw multinomial outcome counts for each basis from exact probabilities.

    Returns ``(..., 3, 2**n)`` integer counts. Dividing by ``shots`` gives empirical probabilities
    that can be fed to :func:`correlators_from_probabilities` to emulate finite-shot readout.
    """
    p = np.asarray(p_xyz, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p = p / p.sum(axis=-1, keepdims=True)
    flat = p.reshape(-1, p.shape[-1])
    counts = np.stack([rng.multinomial(shots, row) for row in flat], axis=0)
    return counts.reshape(p.shape)


def correlators_from_counts(
    counts_xyz: Sequence[Mapping[str, int]] | Sequence[Mapping[int, int]],
    cset: CorrelatorSet,
) -> np.ndarray:
    """Correlators from Qiskit-style count dictionaries for the X, Y and Z settings.

    Args:
        counts_xyz: Three mappings (X, Y, Z order) from outcome bitstring (Qiskit convention,
            qubit 0 rightmost) or integer outcome to count.
        cset: Correlator set.
    """
    length = 1 << cset.n
    p = np.zeros((3, length), dtype=np.float64)
    for b, counts in enumerate(counts_xyz):
        total = 0.0
        for key, c in counts.items():
            z = int(key, 2) if isinstance(key, str) else int(key)
            p[b, z] += c
            total += c
        if total > 0:
            p[b] /= total
    return correlators_from_probabilities(p, cset)


# ---------------------------------------------------------------------- decoding
def decode_signs(
    corr: np.ndarray,
    *,
    zero_rule: str = "plus",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Decode correlators into spins ``x ∈ {-1, +1}^m`` via ``x_i = sgn(<Π_i>)``.

    Args:
        corr: ``(..., m)`` correlators.
        zero_rule: What to do with exact zeros: ``"plus"`` (deterministic +1) or ``"random"``.
        rng: Generator used when ``zero_rule == "random"``.
    """
    x = np.where(corr >= 0, 1, -1).astype(np.int8)
    zeros = corr == 0
    if zero_rule == "random" and zeros.any():
        rng = np.random.default_rng() if rng is None else rng
        x[zeros] = rng.choice(np.array([-1, 1], dtype=np.int8), size=int(zeros.sum()))
    elif zero_rule not in ("plus", "random"):
        raise ValueError(f"unknown zero_rule {zero_rule!r}")
    return x


def iter_labels(cset: CorrelatorSet) -> Iterable[str]:
    """Convenience iterator over Pauli labels (see :meth:`CorrelatorSet.pauli_labels`)."""
    yield from cset.pauli_labels()
