"""Pauli correlation encoding: decisions live in correlation signs.

Decision ``i`` is read from the sign of one k-body Pauli string's expectation
value, so an n-qubit register carries ``3 * C(n, k)`` decisions. The strings
split into three mutually commuting families, which is why three measurement
settings suffice at any problem size.

Modules
-------
families     Enumerates the three commuting families and assigns decisions to
             strings, randomly or with a coupling-aware heuristic.
simulator    Exact statevector primitives. Qubit 0 is the most significant bit
             of the basis index; see :mod:`qgridx.hardware.qasm` for why that
             matters when talking to a device.
loss         Correlator extraction from a state, and the surrogate loss used by
             the directly trained reference arm.
correlators  Walsh-Hadamard extraction of every correlator in a family from one
             measurement record. This is what makes m = 9,009 tractable.
shot_noise   Finite-shot sampling model and the sign-flip law it implies.
"""
from qgridx.encoding.families import (
    all_strings, random_assignment, graph_aware_assignment, max_capacity,
)
from qgridx.encoding.loss import correlators, pce_loss
from qgridx.encoding.correlators import correlators_fast
from qgridx.encoding.simulator import (
    zero_state, apply_1q, apply_2q, pauli_expectation,
)

__all__ = [
    "all_strings", "random_assignment", "graph_aware_assignment", "max_capacity",
    "correlators", "correlators_fast", "pce_loss",
    "zero_state", "apply_1q", "apply_2q", "pauli_expectation",
]
