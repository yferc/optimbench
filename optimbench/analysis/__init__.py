"""Offline analysis that is deliberately kept out of the scoring path.

The verifier and the task score stay deterministic and dependency-light: they measure the
agent against the heuristic reference solve. This layer answers a different, heavier question,
"how tight is that heuristic reference against a true optimum," using an exact-ish OR-Tools
solve. It depends on the optional solver extra and is never imported by the core package.
"""
from optimbench.analysis.optimality import optimal_cost

__all__ = ["optimal_cost"]
