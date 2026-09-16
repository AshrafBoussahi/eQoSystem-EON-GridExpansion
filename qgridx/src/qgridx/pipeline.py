"""One call from a planning instance to a buildable plan.

This is the short path through the package, for people who want the method
rather than its parts. It wires the four stages together with the settings used
for every reported result:

    instance  ->  encode  ->  generate  ->  decode  ->  plan

``solve`` assigns each decision to a Pauli string, trains the generator on the
decoded cost of the plans its circuits produce, executes the best circuit it
found, and repairs the resulting correlation signs into a budget-feasible plan.

The defaults are the paper's. In particular the gate budget is three times the
starting allowance, which the design sweep identified as the one choice that
actually controls solution quality: raising it from 19 to 59 gates lifted the
share of instances solved exactly from 23.8 to 48.8 percent, and raising it
further changed nothing detectable.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from qgridx.decoder.portfolio import decode_portfolio
from qgridx.decoder.repair import (
    joint_neighborhood_search, repair_budget, repair_domain_wall, sign_readout,
)
from qgridx.encoding.families import max_capacity, random_assignment
from qgridx.encoding.loss import correlators
from qgridx.generator.executor import build_unitary_table, execute_batch
from qgridx.generator.train import RegimeAConfig, train_regime_a
from qgridx.generator.vocab import GQEVocab

#: Gate-budget multiplier over the ``3n + 2`` starting allowance. Selected by
#: the design sweep; see :mod:`qgridx.experiments.gate_budget`.
GATE_BUDGET_MULTIPLIER = 3

#: Objective evaluations per instance. Every arm in every reported comparison
#: receives exactly this many, which is what makes the comparisons meaningful.
DEFAULT_EVALS = 10_000


@dataclass
class Plan:
    """A decoded, budget-feasible plan and how it was obtained."""

    x: np.ndarray
    cost: float
    levels: np.ndarray
    n_qubits: int
    n_gates: int
    correlators: np.ndarray = field(repr=False, default=None)

    @property
    def sited_mw(self):
        """Installed power per tier level, at 25 MW per tier."""
        from qgridx.grid.contingency import MW_PER_CAPACITY_TIER
        return float(self.levels.sum() * MW_PER_CAPACITY_TIER)

    @property
    def n_sites(self):
        return int((self.levels > 0).sum())


def smallest_register(m, k=2):
    """Fewest qubits whose k-body capacity ``3 * C(n, k)`` holds ``m`` decisions."""
    n = k
    while max_capacity(n, k) < m:
        n += 1
    return n


def solve(inst, n_qubits=None, k=2, evals=DEFAULT_EVALS, seed=101,
          assignment_seed=None, gate_budget=GATE_BUDGET_MULTIPLIER,
          do_nothing_guard=True):
    """Encode, generate, and decode an instance into a feasible plan.

    Parameters
    ----------
    inst : QUBOInstance
        From :func:`qgridx.problems.build_instance` or the registry.
    n_qubits : int, optional
        Register width. Defaults to the smallest that fits, which is the
        interesting case: the encoding is being used at capacity.
    k : int
        Pauli string locality. Higher ``k`` buys more decisions per qubit.
    evals : int
        Objective evaluations. Reduce for a smoke run.
    do_nothing_guard : bool
        Clamp the reported cost at zero. The empty plan is always feasible with
        cost exactly zero, so a positive cost means the solver returned
        something worse than doing nothing. Applied identically to every arm.

    Returns
    -------
    Plan
    """
    n = n_qubits or smallest_register(inst.m, k)
    if max_capacity(n, k) < inst.m:
        raise ValueError(
            f"{inst.m} decisions exceed capacity {max_capacity(n, k)} at n={n}, k={k}")

    assignment = random_assignment(n, k, inst.m,
                                   seed=assignment_seed if assignment_seed is not None
                                   else seed)
    vocab = GQEVocab(n=n, max_len=(3 * n + 2) * gate_budget)
    table = build_unitary_table(vocab)

    cfg = RegimeAConfig(max_evals=evals, checkpoints=(evals,), seed=seed)
    res = train_regime_a(inst, vocab, table, n, assignment, [], cfg)

    with torch.no_grad():
        state = execute_batch(
            torch.tensor(res.best_tokens[None], dtype=torch.long), table, n)
        pi = correlators(state, assignment, n).numpy()[0]

    x, cost = decode(pi, inst, do_nothing_guard=do_nothing_guard)
    return Plan(
        x=x,
        cost=cost,
        levels=inst.levels_from_x(x),
        n_qubits=n,
        n_gates=int((res.best_tokens != vocab.EOS_ID).sum() - 1),
        correlators=pi,
    )


def decode(pi, inst, do_nothing_guard=True):
    """Turn correlation values into the best feasible plan the decoder can find.

    Runs both decode paths, the repair-and-search path and the integer-program
    portfolio, and keeps whichever plan is cheaper. Every solver arm in every
    reported comparison shares this function, so differences between arms are
    attributable to the input alone.
    """
    x_naive = repair_budget(repair_domain_wall(sign_readout(np.asarray(pi).copy()),
                                               inst), inst)
    x_naive = joint_neighborhood_search(x_naive, inst)
    port = decode_portfolio(pi, inst)

    c_naive = inst.cost(x_naive) if inst.is_feasible(x_naive) else np.inf
    x, cost = (x_naive, c_naive) if c_naive <= port.cost else (port.x, port.cost)
    cost = float(cost)
    return x, (min(cost, 0.0) if do_nothing_guard else cost)


__all__ = ["Plan", "solve", "decode", "smallest_register",
           "GATE_BUDGET_MULTIPLIER", "DEFAULT_EVALS"]
