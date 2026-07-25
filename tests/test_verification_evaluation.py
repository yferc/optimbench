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
    expected_commits = len(env.scenario.disruptions) + 1
    return VERIFIER.verify(env.state, env.trajectory, expected_commits)


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


def test_greedy_resolves_all_waves_with_full_integrity():
    report = Evaluator().evaluate(GreedyDispatcher, Difficulty.HARD, range(8))
    assert report.integrity.mean == 1.0
    assert report.robustness.mean == 1.0


def test_dodging_disruptions_fails_integrity_and_robustness():
    scenario = GEN.generate(1, Difficulty.HARD)
    env = DispatchEnvironment()
    env.reset(scenario)
    agent = GreedyDispatcher()
    while env.state.wave == 0 and not env.done:
        action, args = agent.act(env.observation())
        env.step(action, args)
    env.step(ActionType.DISPATCH, {})  # face wave 1, then stall
    while not env.done:
        env.step(ActionType.REFUSE, {"reason": "stalling"})
    result = _verify(env)
    assert IntegrityFlag.DISRUPTIONS_UNRESOLVED in result.integrity_flags
    assert not result.integrity_ok


def test_reference_cost_beats_or_matches_naive_route():
    env = _solve(4, Difficulty.HARD)
    result = _verify(env)
    assert result.reference <= result.objective + 1e-6


def test_task_score_rewards_routing_efficiency():
    efficient = VerificationResult(feasible=True, objective=100.0, reference=90.0)
    wasteful = VerificationResult(feasible=True, objective=150.0, reference=90.0)
    assert task_score(efficient) == pytest.approx(0.9)
    assert task_score(wasteful) == pytest.approx(0.6)


def test_task_metric_is_not_saturated_at_one():
    scores = [task_score(_verify(_solve(seed, Difficulty.HARD))) for seed in range(12)]
    assert all(s <= 1.0 for s in scores)
    assert min(scores) < 1.0
