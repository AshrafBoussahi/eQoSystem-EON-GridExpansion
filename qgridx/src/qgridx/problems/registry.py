"""The frozen benchmark registry: 160 instances with certified optima.

Every grid-side number in the paper is computed on this registry, so it is
shipped with the package rather than regenerated. Each row records the
generation seed and configuration needed to rebuild the exact instance, plus
the mixed-integer optimum used to score it.

The registry is deliberately calibrated to be non-trivial in both directions:
the empty plan is optimal on only 4.4 percent of instances, no instance is
solved by building everything at maximum tier, the optimum sites 2.0 buses on
average, and a single-start greedy heuristic certifies the optimum on 10
percent. Candidate buses are not hand-picked either; each grid's buses are
ranked by mean congestion-only locational value over forty stochastic demand
draws, which is also what a planner would do.
"""
import numpy as np
import pandas as pd

from qgridx.grid.cases import get_case
from qgridx.problems.siting import build_instance
from qgridx.utils.paths import registry_csv

#: Candidate buses per grid, in screening order (highest mean congestion-only
#: locational value first). An instance with B candidate buses uses the first B.
SCREENED_BUSES = {
    "IEEE-14": [9, 10, 4, 14, 11, 3, 13, 12, 6],
    "IEEE-30": [8, 28, 29, 27, 30, 26, 25, 6, 9],
    "IEEE-57": [16, 12, 10, 51, 9, 50, 55, 11, 43],
    "IEEE-118": [106, 75, 107, 118, 42, 74, 76, 105, 41],
}

#: Calibration constants that define the registry family. Changing any of these
#: produces a different family, so they are pinned here rather than passed in.
REGISTRY_DEFAULTS = {
    "benefit_magnitude": 20.0,
    "budget_fraction_range": (0.15, 0.35),
    "coupling_scale_mult": 40.0,
}


def load_registry(path=None):
    """Load the instance registry as a DataFrame.

    Columns include ``instance_id``, ``grid``, ``config``, ``B``, ``L``, ``m``,
    ``gen_seed``, ``optimum_value``, ``mip_time_ms``, ``trivial`` and
    ``saturated``.
    """
    return pd.read_csv(path or registry_csv())


def rebuild_instance(row, extra_loads=None, **overrides):
    """Rebuild the exact QUBO instance described by a registry row.

    Instances are never stored as matrices; they are rebuilt from their seed and
    configuration so that the physics layer stays in the loop and any change to
    it is visible rather than silently bypassed.

    Parameters
    ----------
    row : Mapping
        A registry row, as a dict or a pandas Series.
    extra_loads : dict, optional
        ``{bus_number: extra_MW}`` added on top of the stochastic demand before
        the power flow, used for the data-center load-growth sweep. Because it
        enters before the DC-OPF it reshapes the prices and therefore the
        objective itself.
    **overrides
        Any keyword accepted by :func:`qgridx.problems.siting.build_instance`,
        overriding :data:`REGISTRY_DEFAULTS`.
    """
    grid = row["grid"]
    case = get_case(grid)
    B, L = int(row["B"]), int(row["L"])
    kwargs = dict(REGISTRY_DEFAULTS)
    kwargs.update(overrides)
    return build_instance(
        seed=int(row["gen_seed"]),
        candidate_buses=SCREENED_BUSES[grid][:B],
        L=L,
        case_module=case,
        lmp_reference_pool=SCREENED_BUSES[grid],
        extra_loads=extra_loads,
        **kwargs,
    )


def load_vector(row, load_scale, extra_loads=None):
    """Per-bus demand for a registry row, including any data-center overlay.

    This is the demand the security screen sees, and it is built the same way
    the instance factory builds it, so the objective and the screen never
    disagree about what the load is.
    """
    case = get_case(row["grid"])
    load = case.BUS[:, 1].copy() * np.asarray(load_scale, dtype=float)
    if extra_loads:
        for bus_num, mw in extra_loads.items():
            load[case.bus_index(bus_num)] += mw
    return load


def nontrivial(reg, threshold=0.1):
    """Rows whose optimum is not simply "site nothing".

    An instance whose optimum is the empty plan cannot show a siting response
    to load growth, so the load-growth sweep restricts itself to these.
    """
    return reg[reg.optimum_value.abs() > threshold]
