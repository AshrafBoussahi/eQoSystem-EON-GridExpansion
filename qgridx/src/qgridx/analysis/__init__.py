"""Statistics, scoring, and figure generation.

Paired comparisons use the Wilcoxon signed-rank test, proportions use Wilson
intervals, and paired proportions use McNemar's exact test. The unit of
analysis is always the instance, never the individual run, so extra training
restarts cannot inflate a sample size.

Modules
-------
stats    Confidence intervals and paired tests.
metrics  Exact-match rate, distance from the optimum, approximation ratio.
scoring  Exact versus sampled scoring modes.
figures  Regenerates every figure from the archived result files.
"""
from qgridx.analysis.stats import wilson_ci

__all__ = ["wilson_ci"]
