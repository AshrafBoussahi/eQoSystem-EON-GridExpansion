"""Phase 2, Ablation 1: alternative per-bus capacity encodings (domain-wall,
one-hot, binary-log), holding the underlying per-bus level-cost table fixed so
only the *encoding* varies. Each encoder exposes: bits per bus, level<->bits
maps, a raw-decode validity check, and how to build (c, Q) from a per-bus
cumulative cost table cost_b[0..L] (cost_b[0] = 0 by construction).
"""
from itertools import product

import numpy as np

ONEHOT_PENALTY = 3.0  # quadratic penalty weight for "not exactly one" violations


class DomainWallEncoder:
    name = "domain_wall"

    def __init__(self, L: int):
        self.L = L
        self.bits_per_bus = L

    def level_to_bits(self, level: int) -> np.ndarray:
        b = np.zeros(self.L, dtype=int)
        b[:level] = 1
        return b

    def is_valid(self, bits: np.ndarray) -> bool:
        seen_zero = False
        for v in bits:
            if v == 0:
                seen_zero = True
            elif seen_zero:
                return False
        return True

    def level_from_bits_repaired(self, bits: np.ndarray) -> int:
        return int(np.sum(bits))  # sort-repair: level = count of ones

    def bus_c(self, cost_b: np.ndarray) -> np.ndarray:
        # cost_b: (L+1,) cumulative cost per level. Since chain is a monotone
        # sum, per-position marginal price is linear-additive with no
        # intra-bus quadratic term needed.
        return np.diff(cost_b)  # (L,) marginal price per chain position

    def local_flip_delta_levels(self, rng, n_trials=2000):
        deltas = []
        for _ in range(n_trials):
            level = rng.integers(0, self.L + 1)
            bits = self.level_to_bits(level)
            pos = rng.integers(0, self.L)
            bits2 = bits.copy()
            bits2[pos] = 1 - bits2[pos]
            new_level = self.level_from_bits_repaired(bits2)  # sort-repair after flip
            deltas.append(abs(new_level - level))
        return np.array(deltas)


class OneHotEncoder:
    name = "one_hot"

    def __init__(self, L: int):
        self.L = L
        self.bits_per_bus = L + 1

    def level_to_bits(self, level: int) -> np.ndarray:
        b = np.zeros(self.L + 1, dtype=int)
        b[level] = 1
        return b

    def is_valid(self, bits: np.ndarray) -> bool:
        return int(np.sum(bits)) == 1

    def level_from_bits_repaired(self, bits: np.ndarray) -> int:
        s = int(np.sum(bits))
        if s == 1:
            return int(np.argmax(bits))
        if s == 0:
            return 0  # repair: default to "not sited"
        # multi-hot: repair to the lowest-cost active level (deterministic tie-break: smallest index)
        return int(np.argmax(bits))

    def bus_qubo(self, cost_b: np.ndarray, penalty: float = ONEHOT_PENALTY):
        """Returns (c, Q) for this bus's (L+1) one-hot bits, cost_b[level] +
        penalty*(sum(bits)-1)^2 expanded into linear/quadratic terms."""
        Lp1 = self.L + 1
        c = cost_b.copy().astype(float)
        Q = np.zeros((Lp1, Lp1))
        # penalty * (sum b_i - 1)^2 = penalty*(sum b_i^2 - 2 sum b_i + sum_ij b_i b_j (i!=j) + 1)
        # b_i^2=b_i (binary) -> contributes penalty*(1-2)*b_i = -penalty*b_i to c
        c += -penalty
        for i in range(Lp1):
            for j in range(Lp1):
                if i != j:
                    Q[i, j] += penalty  # symmetric double counted like elsewhere in this codebase
        return c, Q

    def local_flip_delta_levels(self, rng, n_trials=2000):
        deltas = []
        for _ in range(n_trials):
            level = rng.integers(0, self.L + 1)
            bits = self.level_to_bits(level)
            pos = rng.integers(0, self.L + 1)
            bits2 = bits.copy()
            bits2[pos] = 1 - bits2[pos]
            new_level = self.level_from_bits_repaired(bits2)
            deltas.append(abs(new_level - level))
        return np.array(deltas)


class BinaryLogEncoder:
    name = "binary_log"

    def __init__(self, L: int):
        self.L = L
        self.bits_per_bus = max(1, int(np.ceil(np.log2(L + 1))))

    def level_to_bits(self, level: int) -> np.ndarray:
        return np.array([(level >> b) & 1 for b in range(self.bits_per_bus)], dtype=int)

    def is_valid(self, bits: np.ndarray) -> bool:
        val = sum(int(b) << i for i, b in enumerate(bits))
        return val <= self.L

    def level_from_bits_repaired(self, bits: np.ndarray) -> int:
        val = sum(int(b) << i for i, b in enumerate(bits))
        return min(val, self.L)  # clip-repair: out-of-range codes clamp to max level

    def bus_c(self, cost_b: np.ndarray) -> np.ndarray:
        """Least-squares linear fit of cost_b over the log-encoded bits
        (exact if the encoding admits an affine fit, i.e. essentially always
        for L+1 <= 2^bits with no big gaps; residual quadratic coupling
        dropped -- this is exactly the 'destroys locality/smoothness' issue
        the framework warns log encoding has, made concrete here as a
        genuine linear-fit residual)."""
        nb = self.bits_per_bus
        codes = np.array([self.level_to_bits(l) for l in range(self.L + 1)])  # (L+1, nb)
        A = np.hstack([codes, np.ones((self.L + 1, 1))])
        coef, *_ = np.linalg.lstsq(A, cost_b, rcond=None)
        return coef[:nb]  # drop intercept (global constant, irrelevant to argmin)

    def local_flip_delta_levels(self, rng, n_trials=2000):
        deltas = []
        for _ in range(n_trials):
            level = rng.integers(0, self.L + 1)
            bits = self.level_to_bits(level)
            pos = rng.integers(0, self.bits_per_bus)
            bits2 = bits.copy()
            bits2[pos] = 1 - bits2[pos]
            new_level = self.level_from_bits_repaired(bits2)
            deltas.append(abs(new_level - level))
        return np.array(deltas)


def build_multi_bus_qubo(encoder_cls, cost_tables: list, L: int):
    """cost_tables: list of (L+1,) cumulative cost-by-level arrays, one per
    bus. Returns (c, Q, bus_slices, encoder)."""
    encoder = encoder_cls(L)
    bpb = encoder.bits_per_bus
    B = len(cost_tables)
    m = B * bpb
    c = np.zeros(m)
    Q = np.zeros((m, m))
    slices = []
    for bi, cost_b in enumerate(cost_tables):
        s, e = bi * bpb, (bi + 1) * bpb
        slices.append((s, e))
        if encoder.name == "one_hot":
            cb, qb = encoder.bus_qubo(cost_b)
            c[s:e] = cb
            Q[s:e, s:e] = qb
        else:
            c[s:e] = encoder.bus_c(cost_b)
    return c, Q, slices, encoder
