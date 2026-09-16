"""qGridX: qubit-efficient grid planning with correlation-encoded optimization.

qGridX puts a transmission-planning decision problem onto a handful of qubits
and solves it end to end. It packages three things that normally live apart:

  1. a power-system layer that builds planning instances from real DC optimal
     power flow and screens the resulting plans against the full N-1 outage set
     (:mod:`qgridx.grid`, :mod:`qgridx.problems`);
  2. a quantum layer that stores each binary decision in the sign of one
     multi-qubit Pauli correlation and lets a transformer write the circuit
     (:mod:`qgridx.encoding`, :mod:`qgridx.generator`, :mod:`qgridx.decoder`);
  3. the classical references, statistics, and device-execution paths needed to
     say honestly how well it worked (:mod:`qgridx.baselines`,
     :mod:`qgridx.analysis`, :mod:`qgridx.hardware`).

The headline property is a resource profile that does not grow the way the
problem does. An n-qubit register carries ``3 * C(n, k)`` decisions, and because
the Pauli strings partition into three mutually commuting families, the device
runs exactly three circuits no matter how large the problem becomes.

Quick start
-----------
>>> from qgridx.problems import build_instance
>>> from qgridx.pipeline import solve
>>> inst = build_instance(seed=0)
>>> plan = solve(inst, evals=2000)               # doctest: +SKIP
>>> plan.cost, plan.sited_mw                     # doctest: +SKIP

Every published number is reproduced by :mod:`qgridx.experiments`; see the
README for the one-command entry points.
"""

__version__ = "1.0.1"

__all__ = [
    "analysis",
    "baselines",
    "decoder",
    "encoding",
    "experiments",
    "generator",
    "grid",
    "hardware",
    "maxcut",
    "pipeline",
    "problems",
    "utils",
]
