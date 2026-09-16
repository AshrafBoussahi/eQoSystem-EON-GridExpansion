"""Reproduction drivers, one per reported result.

Each module is the code that produced the numbers, not a reimplementation of
them. Every driver writes a CSV under :func:`qgridx.utils.paths.results_dir`
and prints the summary table it produced, so a run can be compared against the
archived copy shipped with the package.

Every driver also accepts ``--smoke``, which runs the same code path on a
handful of instances at a reduced evaluation budget. Use it first: the full runs
are hours of CPU, and a smoke run confirms the environment before you spend
them.

Drivers
-------
benchmark    Main comparison table: the correlation solver against simulated
             annealing, tabu search, greedy restarts, and an
             unconstrained-input control, all at a matched 10,000-evaluation
             budget on 80 certified instances.
gate_budget  The design sweep that identified the generator's gate allowance as
             the one choice controlling solution quality.
scenarios    Thirteen-scenario stochastic formulation, and the demonstration
             that scenario count costs nothing at the quantum layer.
resilience   N-1 security screening of decoded plans under a 50 to 500 MW
             data-center load, with and without the sited storage on identical
             outages.
microgrid    Storage sizing plus microgrid islanding decisions in one encoding,
             and what the extra decisions cost in qubits.
resources    Measured resource table across problem sizes, and the
             planner-scale capacity projection.
hardware     Device campaign: emit, verify offline, submit, collect, analyze.

Reproduction order
------------------
``benchmark`` and ``gate_budget`` establish the method's behaviour, ``scenarios``,
``resilience`` and ``microgrid`` establish the planning results, ``resources``
summarizes, and ``hardware`` is the only one that needs a device.
"""

__all__ = [
    "benchmark",
    "gate_budget",
    "hardware",
    "microgrid",
    "resilience",
    "resources",
    "scenarios",
]
