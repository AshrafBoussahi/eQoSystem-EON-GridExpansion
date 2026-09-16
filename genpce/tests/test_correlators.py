"""Readout correctness: WHT-based correlators must match Qiskit's own Pauli expectation values."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from genpce.pce import (
    CorrelatorSet,
    basis_probabilities,
    correlators_from_counts,
    correlators_from_statevector,
    decode_signs,
    fwht,
    sample_counts,
)
from genpce.sim import AerEvaluator, ShotEvaluator, StatevectorEvaluator, measurement_circuits


def _random_circuit(n: int, depth: int, rng: np.random.Generator) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice(["rx", "ry", "rz"])
            getattr(qc, gate)(float(rng.uniform(-np.pi, np.pi)), q)
        for q in range(rng.integers(0, 2), n - 1, 2):
            qc.cz(q, q + 1)
    return qc


def test_fwht_matches_kron_hadamard():
    n = 5
    rng = np.random.default_rng(0)
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    H = np.array([[1, 1], [1, -1]], dtype=float)
    Hn = np.array([[1.0]])
    for _ in range(n):
        Hn = np.kron(Hn, H)
    np.testing.assert_allclose(fwht(v), Hn @ v, atol=1e-12)
    np.testing.assert_allclose(fwht(fwht(v)) / (1 << n), v, atol=1e-12)


@pytest.mark.parametrize("n,k", [(4, 2), (5, 3), (6, 3), (6, 2)])
def test_correlators_match_qiskit_expectation_values(n, k):
    rng = np.random.default_rng(n * 10 + k)
    cset = CorrelatorSet.build(n, k)
    qc = _random_circuit(n, depth=3, rng=rng)
    sv = Statevector(qc)
    ours = correlators_from_statevector(sv.data, cset)
    ref = np.array([sv.expectation_value(Pauli(lbl)).real for lbl in cset.pauli_labels()])
    np.testing.assert_allclose(ours, ref, atol=1e-10)
    assert ours.shape == (cset.m,)
    assert cset.m == 3 * __import__("math").comb(n, k)


def test_all_orders_and_random_assignment():
    cset = CorrelatorSet.build(5, 3, all_orders_up_to_k=True, assignment="random", seed=1)
    assert cset.m == CorrelatorSet.capacity(5, 3, all_orders_up_to_k=True)
    assert set(cset.support_weights().tolist()) == {1, 2, 3}
    sv = Statevector(_random_circuit(5, 2, np.random.default_rng(3)))
    ours = correlators_from_statevector(sv.data, cset)
    ref = np.array([sv.expectation_value(Pauli(lbl)).real for lbl in cset.pauli_labels()])
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_min_qubits():
    assert CorrelatorSet.min_qubits(250, 3) == 9  # Soloviev & Krompiec: n = 9 for m = 250
    assert CorrelatorSet.min_qubits(800, 3) == 13  # Usuki et al.: n = 13 for G-set 800 vertices
    assert CorrelatorSet.min_qubits(2000, 3) == 17  # Sciorilli et al.: n = 17 for m = 2000


def test_measurement_circuits_reproduce_basis_probabilities():
    """The X/Y/Z measurement circuits (as sent to hardware) must sample the same distributions."""
    n = 4
    qc = _random_circuit(n, 2, np.random.default_rng(7))
    p = basis_probabilities(Statevector(qc).data)
    for b, mc in enumerate(measurement_circuits(qc)):
        mc_no_meas = mc.remove_final_measurements(inplace=False)
        np.testing.assert_allclose(Statevector(mc_no_meas).probabilities(), p[b], atol=1e-12)


def test_counts_path_equals_exact_in_the_limit():
    n, k = 5, 2
    cset = CorrelatorSet.build(n, k)
    qc = _random_circuit(n, 2, np.random.default_rng(11))
    exact = StatevectorEvaluator(cset).evaluate([qc])[0]
    p = basis_probabilities(Statevector(qc).data)
    counts = sample_counts(p, shots=200_000, rng=np.random.default_rng(0))
    dicts = [{int(z): int(c) for z, c in enumerate(counts[b]) if c} for b in range(3)]
    est = correlators_from_counts(dicts, cset)
    assert np.max(np.abs(est - exact)) < 0.02
    shot = ShotEvaluator(cset, shots=200_000, rng=np.random.default_rng(1)).evaluate([qc])[0]
    assert np.max(np.abs(shot - exact)) < 0.02


def test_aer_evaluator_agrees_with_exact():
    n, k = 4, 2
    cset = CorrelatorSet.build(n, k)
    qc = _random_circuit(n, 2, np.random.default_rng(5))
    exact = StatevectorEvaluator(cset).evaluate([qc, qc])
    aer = AerEvaluator(cset, shots=100_000, seed=0).evaluate([qc, qc])
    assert aer.shape == exact.shape
    assert np.max(np.abs(aer - exact)) < 0.03


def test_decode_signs():
    c = np.array([0.3, -0.1, 0.0, 1e-9])
    x = decode_signs(c)
    assert x.tolist() == [1, -1, 1, 1]
    x_rand = decode_signs(c, zero_rule="random", rng=np.random.default_rng(0))
    assert x_rand[0] == 1 and x_rand[1] == -1 and x_rand[2] in (-1, 1)
