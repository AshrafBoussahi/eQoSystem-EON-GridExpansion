"""Sprint 9, A1: full-gate GQE vocabulary. One token = one gate (compound
token: gate type x placement x discrete angle). {RX,RY,RZ} x n qubits x an
8-value angle grid, union {RXX,RYY,RZZ} x C(n,2) pairs x the same grid, union
{BOS, EOS}. n=6 -> 3*6*8 + 3*15*8 + 2 = 144 + 360 + 2 = 506 tokens, matching
the plan's stated vocabulary size exactly.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

ANGLE_GRID = [np.pi / 3, -np.pi / 3, np.pi / 4, -np.pi / 4,
              np.pi / 5, -np.pi / 5, np.pi / 8, -np.pi / 8]
AXES = ["X", "Y", "Z"]


@dataclass
class Token:
    kind: str          # "special" | "1q" | "2q"
    name: str = ""      # for special tokens
    axis: str = ""
    qubits: tuple = ()
    angle: float = 0.0


class GQEVocab:
    """Sprint 9 A1: fixed-angle full-gate vocabulary. Deterministic token
    ordering (BOS, EOS, then all 1q tokens, then all 2q tokens) so vocab
    construction is reproducible across n."""

    def __init__(self, n: int, angle_grid=None, max_len: int = None):
        self.n = n
        self.angle_grid = angle_grid or ANGLE_GRID
        self.tokens: list[Token] = [Token(kind="special", name="BOS"),
                                     Token(kind="special", name="EOS")]
        self.BOS_ID = 0
        self.EOS_ID = 1
        for axis in AXES:
            for q in range(n):
                for a in self.angle_grid:
                    self.tokens.append(Token(kind="1q", axis=axis, qubits=(q,), angle=a))
        for axis in AXES:
            for (q1, q2) in combinations(range(n), 2):
                for a in self.angle_grid:
                    self.tokens.append(Token(kind="2q", axis=axis, qubits=(q1, q2), angle=a))
        self.vocab_size = len(self.tokens)
        self.max_len = max_len or (3 * n + 2)  # 3n gates + BOS + EOS, per the plan's starting point
        self.min_gates = 4  # GQCO's rule: EOS illegal before 4 gates

        self._reverse = {}
        for tid, t in enumerate(self.tokens):
            if t.kind == "special":
                key = ("special", t.name)
            else:
                key = (t.kind, t.axis, t.qubits, round(t.angle, 10))
            self._reverse[key] = tid

    def gate_to_id(self, kind: str, axis: str, qubits: tuple, angle: float) -> int:
        return self._reverse[(kind, axis, qubits, round(angle, 10))]

    def special_id(self, name: str) -> int:
        return self._reverse[("special", name)]

    def token_str(self, tid: int) -> str:
        t = self.tokens[tid]
        if t.kind == "special":
            return t.name
        if t.kind == "1q":
            return f"R{t.axis}{t.qubits[0]}({t.angle:+.3f})"
        return f"R{t.axis}{t.axis}{t.qubits}({t.angle:+.3f})"

    def random_valid_sequence(self, rng: np.random.Generator, length: int = None) -> np.ndarray:
        """A random grammar-valid sequence: BOS, >=min_gates gate tokens
        (never BOS/EOS), then EOS, padded with EOS to self.max_len."""
        length = length if length is not None else rng.integers(self.min_gates, self.max_len - 1)
        gate_ids = rng.integers(2, self.vocab_size, size=length)
        seq = np.concatenate([[self.BOS_ID], gate_ids, [self.EOS_ID]])
        if len(seq) < self.max_len:
            seq = np.concatenate([seq, np.full(self.max_len - len(seq), self.EOS_ID)])
        return seq[:self.max_len]
