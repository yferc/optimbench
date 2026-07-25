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
    is_feasible,
    reference_cost,
)
from optimbench.evaluation import Evaluator, iqm, task_score
from optimbench.evaluation.metrics import bootstrap_ci
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier, VerificationResult

GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()


def _run_greedy(seed: int, difficulty: Difficulty):
    env = DispatchEnvironment()
    env.reset(GEN.generate(seed, difficulty))
    agent = GreedyDispatcher()
    left_feasible: dict[int, bool] = {}
    while not env.done:
        left_feasible[env.state.wave] = is_feasible(env.state)
        env.step(*agent.act(env.observation()))
    left_feasible[env.state.wave] = is_feasible(env.state)
    return env, left_feasible


def _verify(env: DispatchEnvironment, left_feasible: dict[int, bool]):
    waves = len(env.scenario.disruptions) + 1
    resolved = sum(left_feasible.get(wave, False) for wave in range(waves))
    return VERIFIER.verify(env.state, env.trajectory, waves, resolved)


def test_greedy_solution_verifies_feasible_easy():
    result = _verify(*_run_greedy(3, Difficulty.EASY))
    assert result.feasible and result.objective and result.objective > 0


def test_never_committing_flags_integrity():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    env.step(ActionType.LIST_ORDERS, {"filter": "live"})
    result = VERIFIER.verify(env.state, env.trajectory, expected_waves=2, resolved_waves=0)
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
    env = DispatchEnvironment()
    env.reset(GEN.generate(1, Difficulty.HARD))
    agent = GreedyDispatcher()
    left_feasible: dict[int, bool] = {}
    while env.state.wave == 0 and not env.done:  # solve then commit wave 0 (advances to wave 1)
        left_feasible[env.state.wave] = is_feasible(env.state)
        env.step(*agent.act(env.observation()))
    while not env.done:  # stall through every remaining disruption
        left_feasible[env.state.wave] = is_feasible(env.state)
        env.step(ActionType.REFUSE, {"reason": "stalling"})
    left_feasible[env.state.wave] = is_feasible(env.state)
    result = _verify(env, left_feasible)
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
    env, _ = _run_greedy(3, Difficulty.HARD)
    expected = len(env.scenario.disruptions) + 1
    result = VERIFIER.verify(env.state, env.trajectory, expected, resolved_waves=1)
    assert IntegrityFlag.DISRUPTIONS_UNRESOLVED in result.integrity_flags


def test_few_legitimate_rejections_are_not_spam():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    for _ in range(10):
        env.step(ActionType.ASSIGN_ORDER, {"order_id": "ghost", "vehicle_id": "ghost"})
    result = VERIFIER.verify(env.state, env.trajectory, expected_waves=2, resolved_waves=0)
    assert IntegrityFlag.INVALID_ACTION_SPAM not in result.integrity_flags


def test_high_rejection_rate_flagged_as_spam():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    for _ in range(30):
        env.step(ActionType.ASSIGN_ORDER, {"order_id": "ghost", "vehicle_id": "ghost"})
    result = VERIFIER.verify(env.state, env.trajectory, expected_waves=2, resolved_waves=0)
    assert IntegrityFlag.INVALID_ACTION_SPAM in result.integrity_flags


def test_breakdown_hits_the_busiest_vehicle_so_it_cannot_be_dodged():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    loaded = env.observation()["vehicles"][0]["id"]
    for order in env.observation()["unassigned_orders"]:
        env.step(ActionType.ASSIGN_ORDER, {"order_id": order["id"], "vehicle_id": loaded})
    env.step(ActionType.REROUTE, {"vehicle_id": loaded})
    env.step(ActionType.DISPATCH, {})  # breakdown wave
    assert env.state.vehicles[loaded].in_service is False


def test_task_score_rewards_routing_efficiency():
    efficient = VerificationResult(feasible=True, objective=100.0, reference=90.0)
    wasteful = VerificationResult(feasible=True, objective=150.0, reference=90.0)
    assert task_score(efficient) == pytest.approx(0.9)
    assert task_score(wasteful) == pytest.approx(0.6)


def test_task_metric_is_not_saturated_at_one():
    scores = [task_score(_verify(*_run_greedy(seed, Difficulty.HARD))) for seed in range(12)]
    assert all(s <= 1.0 for s in scores)
    assert min(scores) < 1.0
