"""Sprint 1 (paper de-risking), S1.0d: shared statistics utilities. Every
percentage this sprint produces ships as (value, n, ci_low, ci_high) via
`wilson_ci`; paired method comparisons use `paired_bootstrap_diff` and
`mcnemar_exact`.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, binomtest


@dataclass
class ProportionEstimate:
    value: float
    n: int
    ci_low: float
    ci_high: float

    def __str__(self):
        return f"{self.value:.1%} (n={self.n}, 95% CI [{self.ci_low:.1%}, {self.ci_high:.1%}])"


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> ProportionEstimate:
    """Wilson score interval for a binomial proportion -- better-behaved than
    the normal approximation at small n or extreme p (both common here: many
    cells have n in the 20-50 range and proportions near 0% or 100%)."""
    if n == 0:
        return ProportionEstimate(value=float("nan"), n=0, ci_low=float("nan"), ci_high=float("nan"))
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return ProportionEstimate(value=p, n=n, ci_low=max(0.0, center - half), ci_high=min(1.0, center + half))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000,
                           seed: int = 0, confidence: float = 0.95):
    """a, b: (n,) paired 0/1 (or continuous) arrays, same instances, e.g.
    method-A exact_match vs method-B exact_match per instance. Returns
    (mean_diff, ci_low, ci_high) for mean(a) - mean(b), via paired resampling
    (resample instance INDICES jointly, preserving pairing)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo = np.percentile(diffs, 100 * (1 - confidence) / 2)
    hi = np.percentile(diffs, 100 * (1 + confidence) / 2)
    return float(a.mean() - b.mean()), float(lo), float(hi)


def mcnemar_exact(a: np.ndarray, b: np.ndarray):
    """Exact McNemar test for paired binary outcomes (e.g. exact_match of
    method A vs method B on the same instances). Returns (n01, n10, p_value)
    where n01 = A wrong/B right, n10 = A right/B wrong -- the test only uses
    the discordant pairs, via the exact binomial test on n10 out of n01+n10."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n10 = int(np.sum(a & ~b))   # A right, B wrong
    n01 = int(np.sum(~a & b))   # A wrong, B right
    n_disc = n10 + n01
    if n_disc == 0:
        return n01, n10, 1.0
    p = binomtest(n10, n_disc, 0.5).pvalue
    return n01, n10, float(p)


if __name__ == "__main__":
    # smoke test
    est = wilson_ci(18, 20)
    print("wilson_ci(18,20) =", est)
    a = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1])
    b = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
    print("paired_bootstrap_diff:", paired_bootstrap_diff(a, b, n_resamples=2000))
    print("mcnemar_exact:", mcnemar_exact(a, b))
