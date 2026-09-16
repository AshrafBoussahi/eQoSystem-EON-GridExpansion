"""Standard IEEE 14-bus test system data (MATPOWER case14 topology/parameters).

Public-domain test-system data. Used only to derive a physically structured
DC-OPF surrogate (LMPs, PTDFs) for the storage-siting instance factory -- this
is not a power-flow-accuracy exercise, so generator cost coefficients are the
standard MATPOWER case14 values but reactive limits / voltage data are dropped
since a DC (real-power, lossless) OPF is all Phase 0 needs.
"""
import numpy as np

# bus_i, Pd (MW), Qd (MVAr)
BUS = np.array([
    [1, 0.0, 0.0],
    [2, 21.7, 12.7],
    [3, 94.2, 19.0],
    [4, 47.8, -3.9],
    [5, 7.6, 1.6],
    [6, 11.2, 7.5],
    [7, 0.0, 0.0],
    [8, 0.0, 0.0],
    [9, 29.5, 16.6],
    [10, 9.0, 5.8],
    [11, 3.5, 1.8],
    [12, 6.1, 1.6],
    [13, 13.5, 5.8],
    [14, 14.9, 5.0],
])

N_BUS = 14
SLACK_BUS = 1  # bus 1, generator with the largest capacity, 1-indexed

# fbus, tbus, x (p.u. reactance), rateA (MW) -- rateA chosen (not the original
# unconstrained MATPOWER values, which are 0/"unlimited") so that DC-OPF
# produces nontrivial congestion and nonzero line-limit duals; this is a
# documented Phase-0 simplification, see report.
BRANCH = np.array([
    [1, 2, 0.05917, 150],
    [1, 5, 0.22304, 60],
    [2, 3, 0.19797, 65],
    [2, 4, 0.17632, 60],
    [2, 5, 0.17388, 45],
    [3, 4, 0.17103, 45],
    [4, 5, 0.04211, 45],
    [4, 7, 0.20912, 55],
    [4, 9, 0.55618, 30],
    [5, 6, 0.25202, 40],
    [6, 11, 0.19890, 25],
    [6, 12, 0.25581, 20],
    [6, 13, 0.13027, 25],
    [7, 8, 0.17615, 30],
    [7, 9, 0.11001, 30],
    [9, 10, 0.08450, 20],
    [9, 14, 0.27038, 20],
    [10, 11, 0.19207, 20],
    [12, 13, 0.19988, 15],
    [13, 14, 0.34802, 15],
])

# gen bus, Pmax (MW), Pmin (MW), linear marginal cost ($/MWh, from MATPOWER
# case14 gencost linear term 'b'; quadratic term dropped for a DC-OPF LP)
GEN = np.array([
    [1, 332.4, 0.0, 20.0],
    [2, 140.0, 0.0, 20.0],
    [3, 100.0, 0.0, 40.0],
    [6, 100.0, 0.0, 40.0],
    [8, 100.0, 0.0, 40.0],
])


def bus_index(bus_num: int) -> int:
    return int(bus_num) - 1


def incidence_and_susceptance():
    """Return (A, b, from_idx, to_idx): branch-bus incidence matrix, branch
    susceptances (1/x), and 0-indexed from/to bus arrays."""
    n_branch = BRANCH.shape[0]
    A = np.zeros((n_branch, N_BUS))
    b = np.zeros(n_branch)
    from_idx = np.zeros(n_branch, dtype=int)
    to_idx = np.zeros(n_branch, dtype=int)
    for l, (fb, tb, x, _rate) in enumerate(BRANCH):
        fi, ti = bus_index(fb), bus_index(tb)
        A[l, fi] = 1.0
        A[l, ti] = -1.0
        b[l] = 1.0 / x
        from_idx[l] = fi
        to_idx[l] = ti
    return A, b, from_idx, to_idx


def compute_ptdf() -> np.ndarray:
    """Standard DC PTDF matrix (n_branch x n_bus), slack-referenced."""
    A, b, _, _ = incidence_and_susceptance()
    Bf = (b[:, None]) * A  # (n_branch, n_bus), flow = Bf @ theta
    Bbus = A.T @ Bf  # (n_bus, n_bus) nodal susceptance matrix

    slack = bus_index(SLACK_BUS)
    keep = [i for i in range(N_BUS) if i != slack]
    Bbus_red = Bbus[np.ix_(keep, keep)]
    Xbus_red = np.linalg.inv(Bbus_red)

    Xbus = np.zeros((N_BUS, N_BUS))
    Xbus[np.ix_(keep, keep)] = Xbus_red
    # slack row/col are zero by construction (angle reference)

    ptdf = Bf @ Xbus  # (n_branch, n_bus)
    return ptdf


def rate_a() -> np.ndarray:
    return BRANCH[:, 3].astype(float)


def gen_buses():
    return GEN[:, 0].astype(int)
