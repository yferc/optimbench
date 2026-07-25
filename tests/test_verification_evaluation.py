import numpy as np
import pytest

from optimbench.agents import GreedyDispatcher
from optimbench.domain import (
    ActionType,
    Difficulty,
    DispatchState,
    IntegrityFlag,
    Order,
    RoadNetwork,
    Vehicle,
    euclidean_time_matrix,
    fleet_cost,
    reference_cost,
)
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


def test_reference_is_agent_independent_and_penalizes_fragmentation():
    coords = np.array([[0, 0], [90, 0], [90, 5], [90, 10]], dtype=float)
    times = euclidean_time_matrix(coords)
    net = RoadNetwork(coords, times, times.copy())
    orders = {f"o{i}": Order(f"o{i}", i, 1, 0, 10_000, 0) for i in (1, 2, 3)}
    fragmented = DispatchState(net, dict(orders), {
        "v0": Vehicle("v0", 12, 0, 10_000, assigned=["o1"], route=[0, 1, 0]),
        "v1": Vehicle("v1", 12, 0, 10_000, assigned=["o2"], route=[0, 2, 0]),
        "v2": Vehicle("v2", 12, 0, 10_000, assigned=["o3"], route=[0, 3, 0]),
    })
    consolidated = DispatchState(net, dict(orders), {
        "v0": Vehicle("v0", 12, 0, 10_000, assigned=["o1", "o2", "o3"], route=[0, 1, 2, 3, 0]),
        "v1": Vehicle("v1", 12, 0, 10_000),
        "v2": Vehicle("v2", 12, 0, 10_000),
    })
    assert reference_cost(fragmented) == pytest.approx(reference_cost(consolidated))

    frag = VerificationResult(True, objective=fleet_cost(fragmented), reference=reference_cost(fragmented))
    cons = VerificationResult(True, objective=fleet_cost(consolidated), reference=reference_cost(consolidated))
    assert task_score(frag) < task_score(cons)
    assert task_score(cons) == pytest.approx(1.0, abs=0.05)


def test_unresolved_flag_reflects_resolution_not_commit_count():
    env = _solve(3, Difficulty.HARD)
    expected = len(env.scenario.disruptions) + 1
    resolved = VERIFIER.verify(env.state, env.trajectory, expected, resolved_commits=1)
    assert IntegrityFlag.DISRUPTIONS_UNRESOLVED in resolved.integrity_flags


def test_few_legitimate_rejections_are_not_spam():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    for _ in range(5):
        env.step(ActionType.ASSIGN_ORDER, {"order_id": "ghost", "vehicle_id": "ghost"})
    result = VERIFIER.verify(env.state, env.trajectory, expected_commits=2)
    assert IntegrityFlag.INVALID_ACTION_SPAM not in result.integrity_flags


def test_task_score_rewards_routing_efficiency():
    efficient = VerificationResult(feasible=True, objective=100.0, reference=90.0)
    wasteful = VerificationResult(feasible=True, objective=150.0, reference=90.0)
    assert task_score(efficient) == pytest.approx(0.9)
    assert task_score(wasteful) == pytest.approx(0.6)


def test_task_metric_is_not_saturated_at_one():
    scores = [task_score(_verify(_solve(seed, Difficulty.HARD))) for seed in range(12)]
    assert all(s <= 1.0 for s in scores)
    assert min(scores) < 1.0
