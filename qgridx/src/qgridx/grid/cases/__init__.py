"""IEEE test systems, one module per network.

Every case module exposes the same surface: ``BUS``, ``BRANCH``, ``GEN``
arrays, ``N_BUS``, ``SLACK_BUS``, ``bus_index()``, ``rate_a()``,
``incidence_and_susceptance()`` and ``compute_ptdf()``. That uniformity is what
lets an experiment sweep across grids without special-casing any of them.
"""
from qgridx.grid.cases import case14, case30, case57, case118

CASES = {
    "IEEE-14": case14,
    "IEEE-30": case30,
    "IEEE-57": case57,
    "IEEE-118": case118,
}


def get_case(name):
    """Look up a case module by the name used throughout the result files."""
    try:
        return CASES[name]
    except KeyError:
        raise KeyError(
            f"unknown grid {name!r}; available: {sorted(CASES)}") from None


__all__ = ["case14", "case30", "case57", "case118", "CASES", "get_case"]
