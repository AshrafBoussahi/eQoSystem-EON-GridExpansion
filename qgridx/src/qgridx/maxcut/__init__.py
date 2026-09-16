"""Graph-partitioning benchmark used to stress the encoding at scale.

Maximum cut is where the encoding is pushed hardest: the problem size is the
encoding capacity ``3 * C(n, k)``, so sweeping ``(n, k)`` sweeps problem size
directly, up to 9,009 decisions on 15 qubits. Reference cut values come from a
low-rank semidefinite relaxation solved by Burer-Monteiro factorization, which
is a strong heuristic and not a certificate; ratios above 1.0 mean the pipeline
beat that reference run, not that it exceeded the true optimum.

Modules
-------
graphs            Random-graph ensemble and instance IO.
burer_monteiro    Classical reference cut.
local_search      The 1-bit / 2-bit post-processor shared by every arm.
pce_direct        Directly trained circuits, the reference quantum arm.
reference_ansatz  The fixed brickwork ansatz that arm optimizes.
train, reward     Generative arm for MaxCut.
blind             Random-input control through the identical decoder.
"""
__all__ = [
    "blind", "burer_monteiro", "graphs", "local_search",
    "pce_direct", "reference_ansatz", "reward", "train",
]
