"""Metrics harness: optimality gap, feasibility rate, wall-clock, seeds, and
experiment logging to disk (CSV per experiment run, one row per instance).
"""
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@contextmanager
def timer():
    t0 = time.perf_counter()
    out = {}
    yield out
    out["wall_clock_s"] = time.perf_counter() - t0


@dataclass
class MetricsHarness:
    rows: list = field(default_factory=list)

    def log(self, **kwargs):
        self.rows.append(kwargs)

    def add_optimality_gap(self, instance_seed, method_cost, exact_cost, **extra):
        gap = None
        if exact_cost is not None and abs(exact_cost) > 1e-9:
            gap = (method_cost - exact_cost) / abs(exact_cost)
        elif exact_cost is not None:
            gap = method_cost - exact_cost
        row = dict(seed=instance_seed, method_cost=method_cost, exact_cost=exact_cost,
                   optimality_gap=gap)
        row.update(extra)
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)

    def summary(self) -> dict:
        df = self.to_frame()
        out = {}
        if "optimality_gap" in df.columns:
            gaps = df["optimality_gap"].dropna()
            out["mean_gap"] = float(gaps.mean()) if len(gaps) else None
            out["median_gap"] = float(gaps.median()) if len(gaps) else None
            out["exact_match_rate"] = float((gaps.abs() < 1e-9).mean()) if len(gaps) else None
        if "feasible" in df.columns:
            out["feasibility_rate"] = float(df["feasible"].mean())
        if "wall_clock_s" in df.columns:
            out["mean_wall_clock_s"] = float(df["wall_clock_s"].mean())
        out["n"] = len(df)
        return out


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
