"""Where qGridX looks for data and results.

The package ships its input data (RTS-GMLC series, the frozen instance
registry) and the archived results of every reported experiment. Both are
resolved relative to the installed package, so an editable checkout and a
``pip install`` behave identically, and neither depends on the current working
directory.

Set ``QGRIDX_DATA`` or ``QGRIDX_RESULTS`` to point somewhere else, which is
what the qBraid launcher does when it stages large inputs outside the wheel.
"""
import os
from pathlib import Path


def package_root() -> Path:
    """Directory containing the installed :mod:`qgridx` package."""
    return Path(__file__).resolve().parents[1]


def _repo_relative(name: str) -> Path:
    """Fall back to the source checkout when running from a git clone."""
    return package_root().parents[1] / name


def data_dir() -> Path:
    """Input data: RTS-GMLC time series and the instance registry."""
    env = os.environ.get("QGRIDX_DATA")
    if env:
        return Path(env)
    bundled = package_root() / "data"
    return bundled if bundled.exists() else _repo_relative("data")


def results_dir() -> Path:
    """Archived results of every reported experiment."""
    env = os.environ.get("QGRIDX_RESULTS")
    if env:
        return Path(env)
    bundled = package_root() / "results"
    return bundled if bundled.exists() else _repo_relative("results")


def registry_csv() -> Path:
    """The frozen 160-instance benchmark registry with certified optima."""
    return data_dir() / "registry" / "instances_v2.csv"


def rts_gmlc_dir() -> Path:
    """Directory holding the RTS-GMLC five-minute load, wind and PV series."""
    return data_dir() / "rts_gmlc"
