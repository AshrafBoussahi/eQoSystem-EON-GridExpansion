"""Project 2, Arm B: MaxCut-specific reward function for GQE. Deliberately
NOT reward.py's reward_batch_single_instance -- that decoder is domain-wall
repair + budget projection (power-grid specific), which protocol Section D
explicitly forbids for the MaxCut arms ("Do not use domain-wall repair,
budget-constrained greedy removal, or any other decoder from your
power-grid work. This is pure MaxCut local search."). The only decode step
here is sign-readout + the exact 1-bit/2-bit local search (maxcut_local_
search.py), identical to Arms A and C, per protocol 1.4 step 4.
"""
import numpy as np
import torch

from qgridx.maxcut.local_search import cut_value, local_search_1bit_2bit
from qgridx.decoder.repair import sign_readout
from qgridx.encoding.correlators import correlators_fast as correlators
from qgridx.generator.executor_scalable import execute_batch_scalable


def reward_batch_maxcut(tokens: torch.Tensor, vocab, n: int, assignment, W: np.ndarray, edges: list):
    """tokens: (batch, L). Returns (costs, rewards, raw_cuts): costs[i] =
    -E_full[i] (lower is better, matching this codebase's best_cost-tracking
    convention), rewards[i] = E_full[i] (higher is better), raw_cuts[i] =
    E_raw[i] (pre-local-search cut, protocol 1.4 step 6: "for direct
    comparison, also report r_circuit ... by skipping step 4")."""
    with torch.no_grad():
        state = execute_batch_scalable(tokens, vocab, n)
        pi = correlators(state, assignment, n).numpy()
    costs = np.zeros(len(tokens))
    raw_cuts = np.zeros(len(tokens))
    for i in range(len(tokens)):
        x_raw = sign_readout(pi[i].copy())
        raw_cuts[i] = cut_value(x_raw, W)
        _, E_full, _ = local_search_1bit_2bit(x_raw, W, edges=edges)
        costs[i] = -E_full
    rewards = -costs
    return costs, rewards, raw_cuts
