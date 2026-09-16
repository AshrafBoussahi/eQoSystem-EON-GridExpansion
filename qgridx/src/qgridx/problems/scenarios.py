"""Weather scenarios, and why they cost nothing at the quantum layer.

Siting is a first-stage decision, taken before the operating condition is
revealed. The expected-value problem is therefore

    min_x  E_s[ c_s' x + x' Q_s x ]  =  (sum_s p_s c_s)' x + x' (sum_s p_s Q_s) x

which is *itself* a quadratic binary program of exactly the same size. The
aggregation is exact and it happens classically, before a qubit is touched, so
thirteen operating conditions cost the same qubits, the same circuit depth and
the same shot budget as one. Scenario coverage is limited by how many power
flows a planner is willing to solve, not by the device.

The scenario set is built by clustering a full year of real five-minute demand,
wind and solar series jointly, then adding one explicit high-stress condition
drawn from the top decile of net demand. Joint clustering matters: demand and
wind availability correlate at -0.287 in this data, so high demand coincides
with low wind, and sampling them independently would erase exactly the
correlated tail risk a planner hedges against.
"""
import numpy as np
import pandas as pd

from qgridx.problems.siting import build_instance
from qgridx.utils.paths import results_dir, rts_gmlc_dir

#: Number of k-means clusters over the joint (load, wind, PV) series.
N_CLUSTERS = 12

#: Net-demand quantile defining the explicit high-stress scenario.
TAIL_DECILE = 0.90

#: Probability assigned to the high-stress scenario; the cluster weights are
#: renormalized over the remainder.
TAIL_PROBABILITY = 0.10


def load_archived_scenarios(path=None):
    """The thirteen-scenario set used for every reported result.

    Columns: ``scenario_id``, ``kind`` (``cluster`` or ``tail``), ``prob``,
    ``n_intervals``, ``load_scale``, ``wind_scale``, ``pv_scale``,
    ``net_load_scale``. Scales are relative to the annual mean.
    """
    p = path or (results_dir() / "doe_phase3" / "e3a_scenarios.csv")
    return pd.read_csv(p)


def load_rts_series():
    """Raw RTS-GMLC five-minute series, summed to system totals.

    Returns three pandas Series (load, wind, PV) over 105,408 intervals.
    """
    base = rts_gmlc_dir()
    # shipped gzipped: 19 MB of five-minute series compresses to 5 MB, and
    # pandas decompresses transparently from the suffix
    ld = pd.read_csv(base / "REAL_TIME_regional_Load.csv.gz")
    wd = pd.read_csv(base / "REAL_TIME_wind.csv.gz")
    pv = pd.read_csv(base / "REAL_TIME_pv.csv.gz")
    key = [c for c in ld.columns if not np.issubdtype(ld[c].dtype, np.number)]
    key = key or list(ld.columns[:4])
    return (ld.drop(columns=key, errors="ignore").sum(axis=1),
            wd.drop(columns=key, errors="ignore").sum(axis=1),
            pv.drop(columns=key, errors="ignore").sum(axis=1))


def series_correlations(load, wind, pv):
    """The correlations that make joint clustering necessary."""
    return {
        "load_wind": float(np.corrcoef(load, wind)[0, 1]),
        "load_pv": float(np.corrcoef(load, pv)[0, 1]),
        "wind_pv": float(np.corrcoef(wind, pv)[0, 1]),
    }


def scenario_weighted_instance(row, scenarios, rebuild, **kwargs):
    """Build the probability-weighted QUBO over a scenario set.

    Each scenario contributes its own DC-OPF prices and congestion duals; the
    weighted instance reuses the structure of the last build (identical B, L,
    budget cap and bit slices by construction) with expectation-weighted
    coefficients.

    Parameters
    ----------
    row : Mapping
        A registry row identifying the instance.
    scenarios : DataFrame
        As returned by :func:`load_archived_scenarios`.
    rebuild : callable
        ``rebuild(row, load_scale_vector=...) -> QUBOInstance``, normally
        :func:`qgridx.problems.registry.rebuild_instance` with a scenario's
        per-bus load scaling applied.

    Returns
    -------
    QUBOInstance
        Same size and structure as a single-scenario instance. This identity is
        the whole point: the quantum layer cannot tell how many scenarios went
        into it.
    """
    probs = scenarios["prob"].to_numpy(dtype=float)
    probs = probs / probs.sum()

    c_acc = None
    q_acc = None
    inst = None
    for p, (_, scen) in zip(probs, scenarios.iterrows()):
        inst = rebuild(row, scen, **kwargs)
        c_acc = p * inst.c if c_acc is None else c_acc + p * inst.c
        q_acc = p * inst.Q if q_acc is None else q_acc + p * inst.Q

    from dataclasses import replace
    return replace(inst, c=c_acc, Q=q_acc)


def value_of_stochastic_solution(cost_mean_plan, cost_stochastic_plan):
    """How much the scenario-aware plan gains under the high-stress condition.

    Positive means the scenario-weighted decision is better in the tail. Across
    the reported 40 instances this is positive on 77.5 percent and negative on
    none.
    """
    return float(cost_mean_plan - cost_stochastic_plan)


__all__ = [
    "N_CLUSTERS", "TAIL_DECILE", "TAIL_PROBABILITY",
    "load_archived_scenarios", "load_rts_series", "series_correlations",
    "scenario_weighted_instance", "value_of_stochastic_solution",
]
