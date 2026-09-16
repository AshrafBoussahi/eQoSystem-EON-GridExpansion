"""Sprint 9, A1: the circuit executor. Since every GQE token has a FIXED
discrete angle (unlike the brickwork ansatz's continuous per-instance
parameters), every token's full n-qubit unitary is precomputable exactly
once and cached in a lookup table -- turning batched execution (where
different rows in a batch can have entirely different gates at the same
sequence position) into a single gather + batched matrix-vector product per
step, instead of needing per-row qubit-indexed tensor contractions.

EOS (and any padding after it) maps to the identity matrix, so a row that
terminated early is naturally a no-op for the rest of the sequence -- no
separate "active mask" bookkeeping needed.
"""
import numpy as np
import torch

from qgridx.encoding.simulator import CDTYPE, _rot_matrix, _rxx_yy_zz_matrix, apply_1q, apply_2q, zero_state
from qgridx.generator.vocab import GQEVocab


def _embed_1q(gate_2x2: torch.Tensor, qubit: int, n: int) -> torch.Tensor:
    """gate_2x2: (2,2). Returns the full (2^n, 2^n) unitary."""
    dim = 2 ** n
    basis = torch.eye(dim, dtype=CDTYPE)  # "batch" of dim standard basis (row) vectors
    out = apply_1q(basis, gate_2x2, qubit, n)  # out[i] = U @ e_i = i-th column of U
    return out.T.contiguous()


def _embed_2q(gate_4x4: torch.Tensor, q1: int, q2: int, n: int) -> torch.Tensor:
    dim = 2 ** n
    basis = torch.eye(dim, dtype=CDTYPE)
    out = apply_2q(basis, gate_4x4, q1, q2, n)
    return out.T.contiguous()


def build_unitary_table(vocab: GQEVocab) -> torch.Tensor:
    """Returns (vocab_size, 2^n, 2^n) complex128 -- one full unitary per token,
    precomputed once via the existing, already-tested apply_1q/apply_2q."""
    n = vocab.n
    dim = 2 ** n
    table = torch.zeros(vocab.vocab_size, dim, dim, dtype=CDTYPE)
    identity = torch.eye(dim, dtype=CDTYPE)
    for tid, tok in enumerate(vocab.tokens):
        if tok.kind == "special":
            table[tid] = identity
        elif tok.kind == "1q":
            gate = _rot_matrix(tok.axis, torch.tensor([tok.angle], dtype=torch.float64))[0]
            table[tid] = _embed_1q(gate, tok.qubits[0], n)
        else:
            gate = _rxx_yy_zz_matrix(tok.axis, torch.tensor([tok.angle], dtype=torch.float64))[0]
            table[tid] = _embed_2q(gate, tok.qubits[0], tok.qubits[1], n)
    return table


def execute_batch(tokens: torch.Tensor, unitary_table: torch.Tensor, n: int) -> torch.Tensor:
    """tokens: (batch, L) int64, tokens[:,0] assumed BOS (skipped -- BOS also
    maps to identity in the table, so including it would be harmless too).
    Returns the final statevector (batch, 2^n)."""
    batch, L = tokens.shape
    state = zero_state(batch, n)
    for pos in range(L):
        U = unitary_table[tokens[:, pos]]  # (batch, dim, dim) gather
        state = torch.einsum('bij,bj->bi', U, state)
    return state


def sequence_gate_count(tokens: np.ndarray, vocab: GQEVocab) -> dict:
    """Real (non-padding) gate count + type mix for one token sequence (up to
    and excluding the first EOS after position 0)."""
    n_1q, n_2q = 0, 0
    for tid in tokens[1:]:
        if tid == vocab.EOS_ID:
            break
        tok = vocab.tokens[tid]
        if tok.kind == "1q":
            n_1q += 1
        elif tok.kind == "2q":
            n_2q += 1
    return dict(n_1q=n_1q, n_2q=n_2q, n_total=n_1q + n_2q)
