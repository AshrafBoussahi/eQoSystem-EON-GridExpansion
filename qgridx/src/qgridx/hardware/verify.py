"""Offline checks that run before any device time is spent.

Device disagreement is only informative if the program you sent is provably the
circuit you meant. These checks establish that, entirely offline and at zero
cost, so that a later shortfall on hardware can be attributed to the device
rather than to the translation.

Two things are checked.

**The emitted program reproduces the trainer's own state.** The generated token
sequence is executed twice, once through the package's unitary-table executor
and once by interpreting the emitted QASM, and the two statevectors must agree
to numerical precision. In the reported campaign this gives unit fidelity and
correlator agreement to 4e-12.

**Each basis rotation reads the family it claims to.** For every measurement
setting, the rotated state's computational-basis parities must reproduce the
exact correlators of that family. This is what caught the sign question on the
single-gate Y rotation: ``rx(pi/2)`` conjugates Z to ``+Y``, not ``-Y``, and the
check confirms it numerically rather than trusting the algebra.
"""
import numpy as np
import torch

from qgridx.encoding.loss import correlators
from qgridx.generator.executor import build_unitary_table, execute_batch
from qgridx.generator.vocab import GQEVocab
from qgridx.hardware.qasm import simulate_qasm, tokens_to_qasm


def validate_emission(tokens, assignment, n, pi_exact, max_len=None, tol=1e-9):
    """Check that the emitted QASM is the circuit the trainer produced.

    Parameters
    ----------
    tokens : array_like
        The generated token sequence.
    assignment : list[tuple]
        ``(qubit_tuple, axis)`` per decision, as produced by
        :func:`qgridx.encoding.random_assignment`.
    n : int
        Register width.
    pi_exact : array_like
        The exact correlators of the state the trainer prepared.
    tol : float
        Agreement tolerance. Anything above this means the emission changed the
        circuit and no device result from it would be interpretable.

    Returns
    -------
    dict
        ``fidelity``, ``max_correlator_deviation``,
        ``max_basis_rotation_error`` and a boolean ``passed``.
    """
    tokens = np.asarray(tokens)
    pi_exact = np.asarray(pi_exact, dtype=float)
    vocab = GQEVocab(n=n, max_len=max_len or (3 * n + 2) * 3)

    qasm = tokens_to_qasm(tokens, vocab, n, "Z", measure=False)
    state_qasm = simulate_qasm(qasm, n)
    state_pkg = execute_batch(
        torch.tensor(tokens[None], dtype=torch.long), build_unitary_table(vocab), n)

    fidelity = abs(torch.vdot(state_qasm[0], state_pkg[0]).item())
    dev = float(np.max(np.abs(
        correlators(state_qasm, assignment, n).numpy()[0] - pi_exact)))

    # each basis rotation must turn its family's correlators into Z parities,
    # read with this package's convention that qubit 0 is the most significant
    # bit of the basis index
    idx = np.arange(2 ** n)
    worst = 0.0
    for axis in ("X", "Y", "Z"):
        probs = np.abs(simulate_qasm(
            tokens_to_qasm(tokens, vocab, n, axis, measure=False), n)[0].numpy()) ** 2
        for i, (qubits, ax) in enumerate(assignment):
            if ax != axis:
                continue
            par = np.zeros(2 ** n, dtype=np.int8)
            for q in qubits:
                par ^= ((idx >> (n - 1 - q)) & 1).astype(np.int8)
            worst = max(worst, abs(float(np.dot(probs, 1.0 - 2.0 * par)) - pi_exact[i]))

    return {
        "fidelity": fidelity,
        "max_correlator_deviation": dev,
        "max_basis_rotation_error": worst,
        "passed": bool(fidelity > 1 - tol and dev < tol and worst < tol),
    }


def determine_bit_order(counts, assignment, axis, n, pi_exact, shots):
    """Decide empirically which way a backend orders the bits it returns.

    Run the same program on a noiseless simulator, then score both conventions
    against the exactly known correlators. The right one reproduces them to the
    finite-shot resolution ``1/sqrt(N)``; the wrong one is several times worse.
    In the reported campaign this gave 0.030 against 0.140 at 1,024 shots, which
    is unambiguous, and it is checked before any device time is spent rather
    than assumed from documentation.

    Returns
    -------
    dict
        ``msb_first`` (bool), the root-mean-square deviation under each
        convention, and the shot-noise floor for comparison.
    """
    from qgridx.hardware.analysis import counts_to_correlators

    pi_exact = np.asarray(pi_exact, dtype=float)
    out = {}
    for msb in (True, False):
        v = counts_to_correlators(counts, assignment, axis, n, msb)
        got = ~np.isnan(v)
        out["msb" if msb else "lsb"] = float(
            np.sqrt(np.mean((v[got] - pi_exact[got]) ** 2)))
    return {
        "msb_first": out["msb"] <= out["lsb"],
        "rmse_msb_first": out["msb"],
        "rmse_lsb_first": out["lsb"],
        "shot_noise_floor": float(1.0 / np.sqrt(shots)),
    }


__all__ = ["validate_emission", "determine_bit_order"]
