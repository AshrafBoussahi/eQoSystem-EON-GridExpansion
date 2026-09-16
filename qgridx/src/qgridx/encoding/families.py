"""PCE Pauli-string family Pi^(k): 3 mutually-exclusive-axis subsets of
permutations of X^k, Y^k, Z^k over n qubits (paper Eq. 1 / Fig. 1).
Assignment of m problem variables to m of these 3*C(n,k) strings is a design
choice (P4 in the framework) -- Phase 1 uses a random assignment; Phase 2
ablates a graph-aware alternative.
"""
from itertools import combinations

import numpy as np


def max_capacity(n: int, k: int) -> int:
    from math import comb
    return 3 * comb(n, k)


def all_strings(n: int, k: int):
    """All available (qubits_tuple, axis) correlator definitions for Pi^(k)."""
    strs = []
    for axis in ("X", "Y", "Z"):
        for qubits in combinations(range(n), k):
            strs.append((qubits, axis))
    return strs


def random_assignment(n: int, k: int, m: int, seed: int):
    """Randomly select m of the 3*C(n,k) available strings and assign them to
    variable indices 0..m-1 (Phase 1 default -- 'random Pauli assignment for now')."""
    strs = all_strings(n, k)
    cap = len(strs)
    if m > cap:
        raise ValueError(f"m={m} exceeds PCE capacity {cap} for n={n}, k={k}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(cap, size=m, replace=False)
    return [strs[i] for i in idx]


def graph_aware_assignment(n: int, k: int, m: int, coupling: np.ndarray, seed: int = 0):
    """Design principle 4: assign strongly |Q_ij|-coupled variable pairs to
    Pauli strings with overlapping qubit support. Greedy heuristic:
      1. Rank variable pairs by |coupling_ij| descending.
      2. Rank candidate-string pairs by qubit-support overlap descending.
      3. Walk both lists together; whenever both endpoints of a variable pair
         are still unassigned, assign them to the next available high-overlap
         string pair. Remaining variables get whatever strings are left,
         highest-degree (in |coupling| row-sum) first for stability.
    """
    m_vars = coupling.shape[0]
    assert m_vars == m
    strs = all_strings(n, k)
    cap = len(strs)
    if m > cap:
        raise ValueError(f"m={m} exceeds PCE capacity {cap} for n={n}, k={k}")

    # variable pairs ranked by |coupling|
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            if coupling[i, j] != 0:
                pairs.append((abs(coupling[i, j]), i, j))
    pairs.sort(reverse=True)

    # string-pairs ranked by support overlap (share >=1 qubit), among *distinct axis* pairs
    # (same-axis strings with overlapping support commute trivially and are cheap to steer together)
    str_pairs = []
    for a in range(cap):
        qa, axa = strs[a]
        sa = set(qa)
        for b in range(a + 1, cap):
            qb, axb = strs[b]
            sb = set(qb)
            overlap = len(sa & sb)
            if overlap > 0:
                str_pairs.append((overlap, a, b))
    str_pairs.sort(reverse=True)

    assigned_var = {}
    used_strings = set()
    sp_ptr = 0

    def next_free_string_pair():
        nonlocal sp_ptr
        while sp_ptr < len(str_pairs):
            _, a, b = str_pairs[sp_ptr]
            sp_ptr += 1
            if a not in used_strings and b not in used_strings:
                return a, b
        return None

    for _, i, j in pairs:
        if i in assigned_var and j in assigned_var:
            continue
        if i in assigned_var or j in assigned_var:
            continue  # keep it simple: only assign fresh pairs together
        sp = next_free_string_pair()
        if sp is None:
            break
        a, b = sp
        assigned_var[i] = a
        assigned_var[j] = b
        used_strings.add(a)
        used_strings.add(b)

    # remaining variables (unpaired, or ran out of string-pairs): assign leftover strings
    remaining_vars = [v for v in range(m) if v not in assigned_var]
    remaining_strings = [s for s in range(cap) if s not in used_strings]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining_strings)
    for v, s in zip(remaining_vars, remaining_strings):
        assigned_var[v] = s

    return [strs[assigned_var[v]] for v in range(m)]
