"""Project 2: scalable circuit executor. The original executor.py precomputes
a full (2^n, 2^n) dense unitary per vocabulary token (build_unitary_table)
and does one batched matmul per sequence position -- correct, and fast for
small n, but the table itself is O(vocab_size * 4^n): 908MB at n=8, 4.5GB at
n=9, 22GB at n=10 (confirmed infeasible on a 16GB machine), and a
nonsensical 49.5TB at n=15. That blowup is an artifact of caching a dense
matrix per token, not a limit of statevector simulation itself, which only
ever needs O(2^n) memory for the state.

apply_1q/apply_2q (qgridx/encoding/simulator.py) already implement the correct,
local, O(2^n)-cost gate application via reshape+einsum -- in fact
build_unitary_table's own dense matrices are DERIVED by applying them to
basis vectors. The only reason execute_batch couldn't call them directly
before now: apply_1q/apply_2q apply ONE gate (a single qubit-index, or
qubit-pair) to an entire batch of states at once -- they support a
per-row-different ANGLE (a (batch,2,2) gate tensor), but not a per-row
different qubit PLACEMENT, which is exactly what GQE needs (different rows
in a training batch sample different tokens -- different gate types and
qubit placements -- at the same sequence position).

The fix: at each sequence position, group the batch's rows by
(gate kind, qubit placement) -- typically a small number of distinct groups
even for large batches, since the vocabulary's qubit placements are shared
across many possible angles -- and apply the local gate to each group's
sub-batch of state rows, scattering results back. Total memory: O(2^n) for
the state, full stop, independent of vocab size. No dense (2^n,2^n) matrix
is ever constructed.
"""
import numpy as np
import torch

from qgridx.encoding.simulator import CDTYPE, _rot_matrix, _rxx_yy_zz_matrix, apply_1q, apply_2q, zero_state
from qgridx.generator.vocab import GQEVocab


def execute_batch_scalable(tokens: torch.Tensor, vocab: GQEVocab, n: int) -> torch.Tensor:
    """Drop-in replacement for executor.execute_batch (same signature minus
    the now-unnecessary unitary_table) -- O(2^n) memory throughout, tractable
    well past the n=9 ceiling of the dense-table approach. tokens: (batch, L)
    int64, tokens[:,0] assumed BOS. Returns final statevector (batch, 2^n)."""
    batch, L = tokens.shape
    state = zero_state(batch, n)
    tokens_np = tokens.numpy()

    for pos in range(L):
        col = tokens_np[:, pos]
        # group row indices by (kind, qubits, ) -- angle stays per-row within a group
        groups = {}
        for row in range(batch):
            tok = vocab.tokens[col[row]]
            if tok.kind == "special":
                continue  # identity: no-op, row's amplitudes untouched
            key = (tok.kind, tok.axis, tok.qubits)
            groups.setdefault(key, []).append(row)

        for (kind, axis, qubits), rows in groups.items():
            row_idx = torch.tensor(rows, dtype=torch.long)
            angles = torch.tensor([vocab.tokens[col[r]].angle for r in rows], dtype=torch.float64)
            sub_state = state[row_idx]
            if kind == "1q":
                gate = _rot_matrix(axis, angles)  # (len(rows), 2, 2)
                sub_out = apply_1q(sub_state, gate, qubits[0], n)
            else:
                gate = _rxx_yy_zz_matrix(axis, angles)  # (len(rows), 4, 4)
                sub_out = apply_2q(sub_state, gate, qubits[0], qubits[1], n)
            state = state.index_copy(0, row_idx, sub_out)

    return state
