import numpy as np
import pytest

from optimbench.agents import GreedyDispatcher
from optimbench.domain import ActionType, Difficulty, IntegrityFlag
from optimbench.evaluation import Evaluator, iqm, task_score
from optimbench.evaluation.metrics import bootstrap_ci
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier, VerificationResult

GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()


def _solve(seed: int, difficulty: Difficulty) -> DispatchEnvironment:
    env = DispatchEnvironment()
    env.reset(GEN.generate(seed, difficulty))
    agent = GreedyDispatcher()
    while not env.done:
        action, args = agent.act(env.observation())
        env.step(action, args)
    return env


def _verify(env: DispatchEnvironment):
    return VERIFIER.verify(env.state, env.trajectory, env.scenario.reference_time)


def test_greedy_solution_verifies_feasible_easy():
    result = _verify(_solve(3, Difficulty.EASY))
    assert result.feasible and result.objective and result.objective > 0


def test_never_committing_flags_integrity():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    env.step(ActionType.LIST_ORDERS, {"filter": "live"})
    result = _verify(env)
    assert IntegrityFlag.NEVER_COMMITTED in result.integrity_flags
    assert not result.integrity_ok


def test_task_score_zero_when_infeasible():
    result = VerificationResult(feasible=False, objective=100.0, reference=100.0)
    assert task_score(result) == 0.0


def test_task_score_capped_at_one():
    result = VerificationResult(feasible=True, objective=50.0, reference=100.0)
    assert task_score(result) == 1.0


def test_iqm_trims_tails():
    assert iqm([0.0, 1.0, 1.0, 1.0, 1.0, 100.0]) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_mean():
    values = list(np.linspace(0.4, 0.8, 20))
    low, high = bootstrap_ci(values)
    assert low <= np.mean(values) <= high


def test_evaluator_report_ranges():
    report = Evaluator().evaluate(GreedyDispatcher, Difficulty.EASY, range(10))
    assert report.episodes == 10
    assert 0.0 <= report.feasibility_rate <= 1.0
    assert 0.0 <= report.task.mean <= 1.0
    assert report.feasibility_rate >= 0.9
