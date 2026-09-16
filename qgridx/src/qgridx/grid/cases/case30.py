"""Standard IEEE 30-bus test system data (MATPOWER case30 topology). Sprint 8,
Track B: the second grid for the cross-grid factory extension (PR-25/PR-34).

Same design philosophy as `ieee_case14.py`: public-domain test-system
topology, used only to derive a physically structured DC-OPF surrogate
(LMPs, PTDFs) for the storage-siting instance factory -- not a power-flow-
accuracy exercise. Reactive limits/voltage data dropped (DC OPF only);
generator costs are simplified to linear coefficients (quadratic term
dropped, matching case14's own simplification) and rescaled to the same
order of magnitude as case14's 20-40 range so the two grids' factories share
comparable congestion-pricing behavior without a second calibration
philosophy.
"""
import numpy as np

# bus_i, Pd (MW), Qd (MVAr)
BUS = np.array([
    [1, 0.0, 0.0],
    [2, 21.7, 12.7],
    [3, 2.4, 1.2],
    [4, 7.6, 1.6],
    [5, 94.2, 19.0],
    [6, 0.0, 0.0],
    [7, 22.8, 10.9],
    [8, 30.0, 30.0],
    [9, 0.0, 0.0],
    [10, 5.8, 2.0],
    [11, 0.0, 0.0],
    [12, 11.2, 7.5],
    [13, 0.0, 0.0],
    [14, 6.2, 1.6],
    [15, 8.2, 2.5],
    [16, 3.5, 1.8],
    [17, 9.0, 5.8],
    [18, 3.2, 0.9],
    [19, 9.5, 3.4],
    [20, 2.2, 0.7],
    [21, 17.5, 11.2],
    [22, 0.0, 0.0],
    [23, 3.2, 1.6],
    [24, 8.7, 6.7],
    [25, 0.0, 0.0],
    [26, 3.5, 2.3],
    [27, 0.0, 0.0],
    [28, 0.0, 0.0],
    [29, 2.4, 0.9],
    [30, 10.6, 1.9],
])

N_BUS = 30
SLACK_BUS = 1

# fbus, tbus, x (p.u. reactance), rateA (MVA). rateA = 0.8 x the standard
# MATPOWER case30 values (Sprint 8 B1 calibration): at the unscaled published
# ratings, nominal-load DC-OPF has zero binding lines (max flow/rate = 0.93,
# never quite congested) -- the same "produce nontrivial congestion" fix
# case14's own module already documents needing, just discovered here instead
# of inherited. 0.8x was chosen from a sweep {0.95,...,0.6}: it's the loosest
# scale that already produces a large, stable congestion-driven locational-LMP
# spread (mean |LMP - lambda_sys| ~ 8.9 across the candidate pool, vs. ~0 at
# 1.0x/0.95x and identical results from 0.8x through 0.7x -- i.e. comfortably
# inside a stable regime, not a knife-edge choice).
BRANCH = np.array([
    [1, 2, 0.0575, 104],
    [1, 3, 0.1652, 104],
    [2, 4, 0.1737, 52],
    [3, 4, 0.0379, 104],
    [2, 5, 0.1983, 104],
    [2, 6, 0.1763, 52],
    [4, 6, 0.0414, 72],
    [5, 7, 0.1160, 56],
    [6, 7, 0.0820, 104],
    [6, 8, 0.0420, 25.6],
    [6, 9, 0.2080, 52],
    [6, 10, 0.5560, 25.6],
    [9, 11, 0.2080, 52],
    [9, 10, 0.1100, 52],
    [4, 12, 0.2560, 52],
    [12, 13, 0.1400, 52],
    [12, 14, 0.2559, 25.6],
    [12, 15, 0.1304, 25.6],
    [12, 16, 0.1987, 25.6],
    [14, 15, 0.1997, 12.8],
    [16, 17, 0.1932, 12.8],
    [15, 18, 0.2185, 12.8],
    [18, 19, 0.1292, 12.8],
    [19, 20, 0.0680, 25.6],
    [10, 20, 0.2090, 25.6],
    [10, 17, 0.0845, 25.6],
    [10, 21, 0.0749, 25.6],
    [10, 22, 0.1499, 25.6],
    [21, 22, 0.0236, 25.6],
    [15, 23, 0.2020, 12.8],
    [22, 24, 0.1790, 12.8],
    [23, 24, 0.2700, 12.8],
    [24, 25, 0.3292, 12.8],
    [25, 26, 0.3800, 12.8],
    [25, 27, 0.2087, 12.8],
    [28, 27, 0.3960, 52],
    [27, 29, 0.4153, 12.8],
    [27, 30, 0.6027, 12.8],
    [29, 30, 0.4533, 12.8],
    [8, 28, 0.2000, 25.6],
    [6, 28, 0.0599, 25.6],
])

# gen bus, Pmax (MW), Pmin (MW), linear marginal cost ($/MWh -- rescaled from
# case30's gencost linear coefficients to case14's 20-40 order of magnitude)
GEN = np.array([
    [1, 200.0, 0.0, 20.0],
    [2, 80.0, 0.0, 17.5],
    [5, 50.0, 0.0, 10.0],
    [8, 35.0, 0.0, 32.5],
    [11, 30.0, 0.0, 30.0],
    [13, 40.0, 0.0, 30.0],
])


def bus_index(bus_num: int) -> int:
    return int(bus_num) - 1


def incidence_and_susceptance():
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
    A, b, _, _ = incidence_and_susceptance()
    Bf = (b[:, None]) * A
    Bbus = A.T @ Bf

    slack = bus_index(SLACK_BUS)
    keep = [i for i in range(N_BUS) if i != slack]
    Bbus_red = Bbus[np.ix_(keep, keep)]
    Xbus_red = np.linalg.inv(Bbus_red)

    Xbus = np.zeros((N_BUS, N_BUS))
    Xbus[np.ix_(keep, keep)] = Xbus_red

    ptdf = Bf @ Xbus
    return ptdf


def rate_a() -> np.ndarray:
    return BRANCH[:, 3].astype(float)


def gen_buses():
    return GEN[:, 0].astype(int)
