"""PEGASE-1354 transmission network (public-domain European high-voltage
model, via pandapower's bundled case1354pegase() network) -- Sprint X.13
(DOE Phase 3 scaling demonstration): a large-scale grid for the m=252
(n=9, k=3) siting benchmark, drawing candidate buses from a much larger
pool than the IEEE 14/30/57/118 systems.

Same design philosophy as ieee_case118.py: public-domain topology, used
only to derive a physically structured DC-OPF surrogate (LMPs, PTDFs) for
the storage-siting instance factory, not a power-flow-accuracy exercise.
Built programmatically from pandapower's own MATPOWER-format internal
representation (`net._ppc`, populated by `pandapower.rundcpp`) rather than
hardcoded as a literal array, since embedding ~2000 branch rows as source
code is impractical and error-prone to transcribe by hand -- this module
instead recomputes the same fixed, versioned, public dataset at import
time, which is exactly as reproducible as a hardcoded array would be.

Three documented departures, all because PEGASE's public data (like
case118's) carries no economic-dispatch cost table and no reliable
per-branch thermal rating in pandapower's bundled version:
  (1) BRANCH ratings are *synthesized* via the same inverse-reactance proxy
      case118 uses (rateA_l = SCALE / x_l, floored at 5 MVA), but with a
      PEGASE-specific SCALE constant (70, not case118's 8.0) -- PEGASE's
      reactances are roughly two orders of magnitude smaller than case118's,
      so reusing case118's constant gave a completely uncongested system
      (0/10 trials with any binding line -- caught directly, not assumed:
      an initial guess of 390 produced all-zero Q coupling and NaN linear
      costs from a divide-by-zero in the LMP reference scale). Swept
      scale in {15,18,20,25,30,50,70,90,100}: infeasibility cliff sits at
      ~25-30 (3/30 and 22/30 feasible respectively), a stable plateau of
      30/30 feasible + 30/30 congested runs from 30 to at least 100. 70 is
      comfortably mid-plateau, not hugging the cliff.
  (2) GEN linear costs are synthesized in the same merit-order pattern as
      case118 (larger nameplate capacity -> cheaper), rescaled to case118's
      10-40 $/MWh order of magnitude.
  (3) The slack bus's generator carries an unconstrained-sentinel Pmax/Pmin
      (+-1e9) in the source data (one generator only, bus 640); replaced
      with Pmax = 1.5x total nominal system load (a generous but finite
      swing-generator capacity, standard practice when a source case's
      slack unit has no realistic published rating) and Pmin = 0.0
      (matching every other generator in this module, and case118's
      convention).
  Bus real/reactive loads (Pd, Qd), branch per-unit reactances, and
  generator Pmax values (except the slack bus, see (3)) are the real
  published data -- topology and physics are NOT synthesized.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn

RATE_SCALE = 70.0  # see module docstring (1); calibrated for this grid's reactance scale


def _build():
    net = pn.case1354pegase()
    pp.rundcpp(net)
    ppc = net._ppc
    bus_ppc, branch_ppc, gen_ppc = ppc["bus"], ppc["branch"], ppc["gen"]

    n_bus = bus_ppc.shape[0]
    assert np.array_equal(np.sort(bus_ppc[:, 0]), np.arange(n_bus)), \
        "PEGASE-1354 bus numbering is not contiguous 0..N-1 as expected"

    bus = np.stack([bus_ppc[:, 0] + 1, bus_ppc[:, 2], bus_ppc[:, 3]], axis=1)  # 1-indexed bus_i, Pd, Qd

    slack_row = np.where(bus_ppc[:, 1] == 3)[0]
    assert len(slack_row) == 1, f"expected exactly one slack bus, found {len(slack_row)}"
    slack_bus = int(bus_ppc[slack_row[0], 0]) + 1

    x = branch_ppc[:, 3].astype(float)
    assert np.all(x > 0), "non-positive branch reactance found"
    rate_a = np.maximum(RATE_SCALE / x, 5.0)
    branch = np.stack([branch_ppc[:, 0] + 1, branch_ppc[:, 1] + 1, x, rate_a], axis=1)

    pmax = gen_ppc[:, 8].astype(float).copy()
    gen_bus_1idx = (gen_ppc[:, 0] + 1).astype(int)
    total_load = bus[:, 1].sum()
    slack_gen_mask = gen_bus_1idx == slack_bus
    pmax[slack_gen_mask] = 1.5 * total_load  # see module docstring (3)
    pmin = np.zeros(len(pmax))

    # merit-order cost synthesis, same pattern as case118: larger Pmax -> cheaper,
    # rescaled to case118's 10-40 $/MWh band
    order = np.argsort(-pmax)  # largest capacity first
    rank = np.empty(len(pmax))
    rank[order] = np.arange(len(pmax))
    cost = 10.0 + 30.0 * (rank / max(len(pmax) - 1, 1))

    gen = np.stack([gen_bus_1idx, pmax, pmin, cost], axis=1)
    return bus, branch, gen, n_bus, slack_bus


BUS, BRANCH, GEN, N_BUS, SLACK_BUS = _build()


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
