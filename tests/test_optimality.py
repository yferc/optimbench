"""The OR-Tools optimality analysis. Skipped cleanly when the solver extra is not installed."""
from __future__ import annotations

import pytest

pytest.importorskip("ortools")

from optimbench.analysis import optimal_cost
from optimbench.domain import Difficulty, reference_cost
from optimbench.generation import DispatchScenarioGenerator

GEN = DispatchScenarioGenerator()


@pytest.mark.parametrize("difficulty", list(Difficulty))
def test_optimum_never_exceeds_the_heuristic_reference(difficulty):
    # The reference is a strong heuristic, so a true optimum must be at least as cheap.
    for seed in range(3):
        state = GEN.generate(seed, difficulty).state
        optimum = optimal_cost(state, time_limit_s=2)
        assert 0.0 < optimum <= reference_cost(state) + 1e-6
