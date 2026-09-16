"""Running correlation-encoded circuits on real quantum hardware.

The claim this subpackage exists to test is operational: the number of distinct
circuits a device must execute is three, whatever the problem size, because the
Pauli strings partition into three mutually commuting families. Measured on a
108-qubit superconducting processor, 45 decisions and then 105 decisions were
both read from exactly three jobs and 3,072 shots.

Modules
-------
qasm      Emits device-ready OpenQASM 3 from a generated token sequence, and
          documents the two compiler traps that make QASM 3 mandatory here.
runner    Submits jobs to a provider and retrieves counts, tolerating the
          different result shapes providers return. Finished jobs can be
          collected by id so an interrupted campaign is never re-billed.
analysis  Reconstructs every correlator from the three measurement records and
          scores sign agreement and magnitude retention against simulation.
verify    Offline checks that must pass before any device time is spent.

Nothing here submits a job unless you call it. The verification path in
:mod:`qgridx.hardware.verify` runs entirely offline and is the recommended
first step: it proves the emitted program reproduces the trainer's own
statevector, which is what makes a later disagreement attributable to the
device rather than to the translation.
"""
from qgridx.hardware.qasm import (
    tokens_to_qasm, to_qasm3, calibration_qasm, simulate_qasm, basis_change_qasm,
)
from qgridx.hardware.analysis import counts_to_correlators, tensored_mitigate

__all__ = [
    "tokens_to_qasm", "to_qasm3", "calibration_qasm", "simulate_qasm",
    "basis_change_qasm", "counts_to_correlators", "tensored_mitigate",
]
