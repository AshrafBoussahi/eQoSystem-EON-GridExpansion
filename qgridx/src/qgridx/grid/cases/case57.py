"""Standard IEEE 57-bus test system data (MATPOWER case57 topology, via
pandapower's bundled case57() network). Sprint 12+ (DOE Phase 3 scaling push):
the third grid for the cross-grid factory extension (Tier 3, PR-4x).

Same design philosophy as `ieee_case14.py`/`ieee_case30.py`: public-domain
test-system topology, used only to derive a physically structured DC-OPF
surrogate (LMPs, PTDFs) for the storage-siting instance factory -- not a
power-flow-accuracy exercise.

Two documented departures from the case14/case30 modules, both because this
grid's public MATPOWER data was sourced through pandapower's bundled network
builder rather than hand-transcribed, and pandapower does not carry a usable
per-branch thermal rating or economic-dispatch cost table for this case:
  (1) BRANCH ratings are *synthesized*, not the published rateA column:
      rateA_l = 15.0 / x_l (inverse-reactance proxy for thermal capacity,
      a standard approximation when explicit ratings are unavailable),
      floored at 5 MVA, with the 15.0 scale chosen by the same sweep
      methodology case30's own module documents (loosest scale, in a stable
      plateau comfortably above the infeasibility cliff, that still produces
      a large, non-knife-edge congestion-driven LMP spread -- verified stable
      across 30/30 random per-instance load draws in [0.7x, 1.3x] nominal).
  (2) GEN linear costs are synthesized in a merit-order pattern (larger
      nameplate capacity -> cheaper, a standard baseload-vs-peaker
      assumption), rescaled to case14/case30's 10-40 $/MWh order of
      magnitude, since the published topology carries no gencost table here.
  Reactances (x), bus real loads (Pd), and generator Pmax/Pmin are the real
  published per-unit/MW values (topology and physics are NOT synthesized).
  The slack bus's Pmax (an "infinite" sentinel in pandapower's ppc extraction) is
  replaced with the real published value from a second, independent source
  (pypower's bundled case57.py, itself sourced from the original MATPOWER case
  file): 575.88 MW -- verified to keep the system feasible across
  30/30 random per-instance load draws in [0.7x, 1.3x] nominal, same as the
  branch-rating check.
"""
import numpy as np

# bus_i, Pd (MW), Qd (MVAr)
BUS = np.array([
    [1, 55.0, 17.0],
    [2, 3.0, 88.0],
    [3, 41.0, 21.0],
    [4, 0.0, 0.0],
    [5, 13.0, 4.0],
    [6, 75.0, 2.0],
    [7, 0.0, 0.0],
    [8, 150.0, 22.0],
    [9, 121.0, 26.0],
    [10, 5.0, 2.0],
    [11, 0.0, 0.0],
    [12, 377.0, 24.0],
    [13, 18.0, 2.3],
    [14, 10.5, 5.3],
    [15, 22.0, 5.0],
    [16, 43.0, 3.0],
    [17, 42.0, 8.0],
    [18, 27.2, 9.8],
    [19, 3.3, 0.6],
    [20, 2.3, 1.0],
    [21, 0.0, 0.0],
    [22, 0.0, 0.0],
    [23, 6.3, 2.1],
    [24, 0.0, 0.0],
    [25, 6.3, 3.2],
    [26, 0.0, 0.0],
    [27, 9.3, 0.5],
    [28, 4.6, 2.3],
    [29, 17.0, 2.6],
    [30, 3.6, 1.8],
    [31, 5.8, 2.9],
    [32, 1.6, 0.8],
    [33, 3.8, 1.9],
    [34, 0.0, 0.0],
    [35, 6.0, 3.0],
    [36, 0.0, 0.0],
    [37, 0.0, 0.0],
    [38, 14.0, 7.0],
    [39, 0.0, 0.0],
    [40, 0.0, 0.0],
    [41, 6.3, 3.0],
    [42, 7.1, 4.4],
    [43, 2.0, 1.0],
    [44, 12.0, 1.8],
    [45, 0.0, 0.0],
    [46, 0.0, 0.0],
    [47, 29.7, 11.6],
    [48, 0.0, 0.0],
    [49, 18.0, 8.5],
    [50, 21.0, 10.5],
    [51, 18.0, 5.3],
    [52, 4.9, 2.2],
    [53, 20.0, 10.0],
    [54, 4.1, 1.4],
    [55, 6.8, 3.4],
    [56, 7.6, 2.2],
    [57, 6.7, 2.0]
])

N_BUS = 57
SLACK_BUS = 1

# fbus, tbus, x (p.u. reactance), rateA (MVA, synthesized -- see module docstring)
BRANCH = np.array([
    [1, 2, 0.028000, 535.71],
    [2, 3, 0.085000, 176.47],
    [3, 4, 0.036600, 409.84],
    [4, 5, 0.132000, 113.64],
    [4, 6, 0.148000, 101.35],
    [6, 7, 0.102000, 147.06],
    [6, 8, 0.173000, 86.71],
    [8, 9, 0.050500, 297.03],
    [9, 10, 0.167900, 89.34],
    [9, 11, 0.084800, 176.89],
    [9, 12, 0.295000, 50.85],
    [9, 13, 0.158000, 94.94],
    [13, 14, 0.043400, 345.62],
    [13, 15, 0.086900, 172.61],
    [1, 15, 0.091000, 164.84],
    [1, 16, 0.206000, 72.82],
    [1, 17, 0.108000, 138.89],
    [3, 15, 0.053000, 283.02],
    [5, 6, 0.064100, 234.01],
    [7, 8, 0.071200, 210.67],
    [10, 12, 0.126200, 118.86],
    [11, 13, 0.073200, 204.92],
    [12, 13, 0.058000, 258.62],
    [12, 16, 0.081300, 184.50],
    [12, 17, 0.179000, 83.80],
    [14, 15, 0.054700, 274.22],
    [18, 19, 0.685000, 21.90],
    [19, 20, 0.434000, 34.56],
    [21, 22, 0.117000, 128.21],
    [22, 23, 0.015200, 986.84],
    [23, 24, 0.256000, 58.59],
    [26, 27, 0.254000, 59.06],
    [27, 28, 0.095400, 157.23],
    [28, 29, 0.058700, 255.54],
    [25, 30, 0.202000, 74.26],
    [30, 31, 0.497000, 30.18],
    [31, 32, 0.755000, 19.87],
    [32, 33, 0.036000, 416.67],
    [34, 35, 0.078000, 192.31],
    [35, 36, 0.053700, 279.33],
    [36, 37, 0.036600, 409.84],
    [37, 38, 0.100900, 148.66],
    [37, 39, 0.037900, 395.78],
    [36, 40, 0.046600, 321.89],
    [22, 38, 0.029500, 508.47],
    [41, 42, 0.352000, 42.61],
    [41, 43, 0.412000, 36.41],
    [38, 44, 0.058500, 256.41],
    [46, 47, 0.068000, 220.59],
    [47, 48, 0.023300, 643.78],
    [48, 49, 0.129000, 116.28],
    [49, 50, 0.128000, 117.19],
    [50, 51, 0.220000, 68.18],
    [29, 52, 0.187000, 80.21],
    [52, 53, 0.098400, 152.44],
    [53, 54, 0.232000, 64.66],
    [54, 55, 0.226500, 66.23],
    [44, 45, 0.124200, 120.77],
    [56, 41, 0.549000, 27.32],
    [56, 42, 0.354000, 42.37],
    [57, 56, 0.260000, 57.69],
    [38, 49, 0.177000, 84.75],
    [38, 48, 0.048200, 311.20],
    [18, 4, 0.555000, 27.03],
    [18, 4, 0.430000, 34.88],
    [20, 21, 0.776700, 19.31],
    [25, 24, 1.182000, 12.69],
    [25, 24, 1.230000, 12.20],
    [26, 24, 0.047300, 317.12],
    [29, 7, 0.064800, 231.48],
    [32, 34, 0.953000, 15.74],
    [41, 11, 0.749000, 20.03],
    [45, 15, 0.104200, 143.95],
    [46, 14, 0.073500, 204.08],
    [51, 10, 0.071200, 210.67],
    [49, 13, 0.191000, 78.53],
    [43, 11, 0.153000, 98.04],
    [56, 40, 1.195000, 12.55],
    [57, 39, 1.355000, 11.07],
    [55, 9, 0.120500, 124.48]
])

# gen bus, Pmax (MW), Pmin (MW), linear marginal cost ($/MWh, synthesized merit order)
GEN = np.array([
    [1, 575.9, 0.0, 12.00],
    [2, 100.0, 0.0, 33.67],
    [3, 140.0, 0.0, 25.00],
    [6, 100.0, 0.0, 29.33],
    [8, 550.0, 0.0, 16.33],
    [9, 100.0, 0.0, 38.00],
    [12, 410.0, 0.0, 20.67]
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
