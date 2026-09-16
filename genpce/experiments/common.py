"""Shared helpers for experiment scripts: benchmark instances, caching, result paths."""

from __future__ import annotations

import json
from pathlib import Path

from genpce.pce import CorrelatorSet
from genpce.problems import MaxCutInstance, best_known, erdos_renyi, random_regular, with_pm1_weights

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INSTANCES = RESULTS / "instances"

# Sizes chosen so that m equals the cubic PCE capacity 3·C(n, 3) exactly (n = 6 … 13).
CUBIC_SIZES = {6: 60, 7: 105, 8: 168, 9: 252, 10: 360, 11: 495, 12: 660, 13: 858}


def benchmark_instance(family: str, m: int, seed: int, *, exact_max_m: int = 120) -> MaxCutInstance:
    """A cached benchmark instance with its best-known (exact when affordable) cut value.

    Families: ``reg3`` (random 3-regular), ``er4`` (Erdős–Rényi, average degree 4), and their
    ``_pm1`` variants with ±1 weights. Sciorilli et al. post-select random instances with average
    degree ≥ 3 and random-cut ratio < 0.82; all families here satisfy both.
    """
    base, _, weighted = family.partition("_")
    if base == "reg3":
        m -= m % 2  # a 3-regular graph needs an even number of vertices
        inst = random_regular(m, 3, seed=seed)
    elif base == "er4":
        inst = erdos_renyi(m, 4.0, seed=seed)
    else:
        raise ValueError(f"unknown family {family!r}")
    if weighted == "pm1":
        inst = with_pm1_weights(inst, seed=seed)
    path = INSTANCES / f"{inst.name}.json"
    if path.exists():
        return MaxCutInstance.load(path)
    inst = best_known(inst, cache_dir=INSTANCES, exact_max_m=exact_max_m, seed=seed)
    inst.save(path)
    return inst


def cubic_cset(m: int) -> CorrelatorSet:
    """Smallest cubic (k = 3) correlator set holding ``m`` variables."""
    n = CorrelatorSet.min_qubits(m, 3)
    return CorrelatorSet.build(n, 3, m=m)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=float))
