"""Classical references, all run at a matched evaluation budget.

Reporting these honestly is part of the method rather than an afterthought: at
the scale where optima can be certified, an exact solver finishes in
milliseconds and good metaheuristics beat the quantum arm on solution quality.
What qGridX claims is a resource profile, not a speed-up.

Modules
-------
mip         Exact branch-and-bound via HiGHS, with the quadratic term
            linearized by a McCormick envelope. Supplies the certified optima.
heuristics  Simulated annealing and tabu search on the domain-wall
            representation.
oracle      Brute-force enumeration, for small instances only.
"""
from qgridx.baselines.mip import solve_mip
from qgridx.baselines.heuristics import simulated_annealing, tabu_search

__all__ = ["solve_mip", "simulated_annealing", "tabu_search"]
