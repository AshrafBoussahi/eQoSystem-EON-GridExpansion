"""Circuit evaluators: turn Qiskit circuits into PCE correlator vectors.

Three execution paths share one interface (``evaluate(circuits) -> (B, m) correlators``):

* :class:`StatevectorEvaluator` — exact, via ``qiskit.quantum_info.Statevector``; the reference
  path used inside training loops (cheap for ``n ≤ ~17``).
* :class:`ShotEvaluator` — exact probabilities + multinomial sampling; the cheapest way to study
  finite-shot readout without a noise model.
* :class:`AerEvaluator` — ``qiskit_aer.AerSimulator`` with optional noise model and shots; the
  same three measurement circuits (X, Y, Z settings) are what we submit to real hardware, so this
  is the dress rehearsal for the device run.

All evaluators return correlators in the *variable order* of the :class:`CorrelatorSet`, so the
rest of the pipeline (sign decoding, objectives, rewards) never needs to know how the state was
produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from genpce.pce.correlators import (
    CorrelatorSet,
    basis_probabilities,
    correlators_from_counts,
    correlators_from_probabilities,
    sample_counts,
)

__all__ = [
    "Evaluator",
    "StatevectorEvaluator",
    "ShotEvaluator",
    "AerEvaluator",
    "measurement_circuits",
]


class Evaluator(Protocol):
    """Anything that maps a batch of circuits to a ``(B, m)`` correlator array."""

    cset: CorrelatorSet

    def evaluate(self, circuits: Sequence[QuantumCircuit]) -> np.ndarray: ...


def measurement_circuits(circuit: QuantumCircuit) -> list[QuantumCircuit]:
    """The three measurement circuits (X, Y, Z settings) for a state-preparation circuit.

    X basis: ``H`` on every qubit; Y basis: ``S† H``; Z basis: nothing. All qubits are measured.
    """
    n = circuit.num_qubits
    out = []
    for basis in ("X", "Y", "Z"):
        qc = circuit.copy()
        qc.barrier()
        if basis == "X":
            qc.h(range(n))
        elif basis == "Y":
            qc.sdg(range(n))
            qc.h(range(n))
        qc.measure_all()
        qc.name = f"{circuit.name}_{basis}"
        out.append(qc)
    return out


def statevectors(circuits: Sequence[QuantumCircuit]) -> np.ndarray:
    """Exact statevectors ``(B, 2**n)`` (Qiskit little-endian) for measurement-free circuits."""
    return np.stack([Statevector(qc).data for qc in circuits], axis=0)


@dataclass
class StatevectorEvaluator:
    """Exact correlators via ``qiskit.quantum_info.Statevector``."""

    cset: CorrelatorSet

    def probabilities(self, circuits: Sequence[QuantumCircuit]) -> np.ndarray:
        return basis_probabilities(statevectors(circuits))

    def evaluate(self, circuits: Sequence[QuantumCircuit]) -> np.ndarray:
        return correlators_from_probabilities(self.probabilities(circuits), self.cset)


@dataclass
class ShotEvaluator:
    """Finite-shot correlators sampled from the exact outcome distributions (no gate noise)."""

    cset: CorrelatorSet
    shots: int
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def evaluate(self, circuits: Sequence[QuantumCircuit]) -> np.ndarray:
        p = basis_probabilities(statevectors(circuits))
        counts = sample_counts(p, self.shots, self.rng)
        return correlators_from_probabilities(counts / self.shots, self.cset)


@dataclass
class AerEvaluator:
    """Correlators from ``AerSimulator`` runs of the three measurement circuits per input.

    Args:
        cset: Correlator set.
        shots: Shots per measurement setting.
        noise_model: Optional ``qiskit_aer.noise.NoiseModel`` (e.g. from a backend).
        seed: Simulator seed.
        backend_kwargs: Extra keyword arguments passed to ``AerSimulator``.
    """

    cset: CorrelatorSet
    shots: int = 4000
    noise_model: object | None = None
    seed: int | None = None
    backend_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from qiskit_aer import AerSimulator

        kwargs = dict(self.backend_kwargs)
        if self.noise_model is not None:
            kwargs["noise_model"] = self.noise_model
        self._sim = AerSimulator(**kwargs)

    def evaluate(self, circuits: Sequence[QuantumCircuit]) -> np.ndarray:
        from qiskit import transpile

        meas = [mc for qc in circuits for mc in measurement_circuits(qc)]
        meas = transpile(meas, self._sim, optimization_level=0)
        result = self._sim.run(meas, shots=self.shots, seed_simulator=self.seed).result()
        out = np.empty((len(circuits), self.cset.m), dtype=np.float64)
        for b in range(len(circuits)):
            counts = [result.get_counts(3 * b + s) for s in range(3)]
            out[b] = correlators_from_counts(counts, self.cset)
        return out
