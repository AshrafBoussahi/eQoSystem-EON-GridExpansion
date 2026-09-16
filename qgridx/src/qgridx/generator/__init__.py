"""Generative circuit synthesis: a transformer writes the circuit.

A decoder-only transformer emits gate tokens one at a time. Training ranks
sampled circuits by the cost of the fully decoded, feasible plan they produce
and updates only the transformer's weights, so no gradient is ever taken
through a circuit parameter and no ansatz has to be chosen in advance.

Modules
-------
vocab              Gate-token vocabulary: axis rotations, entangling rotations,
                   a discrete angle grid, and the sequence-length ceiling that
                   turned out to be the binding design choice.
model              The decoder-only transformer.
executor           Exact execution of a token sequence, via a precomputed
                   per-token unitary table.
executor_scalable  Batched executor with O(2^n) memory, needed past n = 9.
reward             Decoded-cost reward, including the margin-aware variant.
train              The training loop: sample, decode, rank, update.
"""
from qgridx.generator.vocab import GQEVocab, ANGLE_GRID
from qgridx.generator.executor import build_unitary_table, execute_batch
from qgridx.generator.train import train_regime_a, RegimeAConfig, RegimeAResult

__all__ = [
    "GQEVocab", "ANGLE_GRID",
    "build_unitary_table", "execute_batch",
    "train_regime_a", "RegimeAConfig", "RegimeAResult",
]
