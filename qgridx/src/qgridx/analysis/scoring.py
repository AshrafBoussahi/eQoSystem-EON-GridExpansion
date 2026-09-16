"""Sprint 1 (paper de-risking), S1.0c: one shared scoring-mode entry point,
`scoring in {"exact", "sampled"}`, used by every method (GQE, fixed-ansatz
PCE, SPSA) so headline comparisons are never accidentally asymmetric (the
exact failure mode this task exists to prevent: GQE scored exact, SPSA/PCE
scored under shot noise, or vice versa).

"exact": correlators computed analytically from the statevector
(`qgridx.encoding.loss.correlators`), zero shots, zero noise.
"sampled": correlators estimated from N shots per of 3 measurement settings
(`qgridx.encoding.shot_noise`), the deployed-budget standing convention (N=1024).
"""
import numpy as np
import torch

from qgridx.encoding.loss import correlators
from qgridx.encoding.shot_noise import all_string_shot_estimates, gather_assignment_estimates

N_SHOTS_DEFAULT = 1024


def score_state(state: torch.Tensor, assignment, n: int, k: int, scoring: str,
                 N_shots: int = N_SHOTS_DEFAULT, seed: int = 0) -> np.ndarray:
    """state: (batch, 2^n) statevector. Returns (batch, m) correlator estimate
    array under the requested scoring mode."""
    if scoring == "exact":
        with torch.no_grad():
            pi = correlators(state, assignment, n)
        return pi.numpy()
    elif scoring == "sampled":
        with torch.no_grad():
            mu_all, sigma_all, strings = all_string_shot_estimates(state, n, k, N_shots, seed=seed)
            mu, _ = gather_assignment_estimates(mu_all, sigma_all, strings, assignment)
        return mu.numpy()
    else:
        raise ValueError(f"scoring must be 'exact' or 'sampled', got {scoring!r}")


def n_evaluations_per_circuit(scoring: str, N_shots: int = N_SHOTS_DEFAULT) -> int:
    """S1.0b: the shared 'one circuit evaluation' unit -- one circuit prepared
    and measured at the deployed budget (3 settings), then one decode. This
    returns the shot count consumed (informational; the *evaluation* count
    used for budget accounting is always 1 per circuit regardless of scoring
    mode -- 'sampled' additionally consumes 3*N_shots real/simulated shots,
    which is a separate, reportable resource axis, not a different evaluation
    count)."""
    return 3 * N_shots if scoring == "sampled" else 0
