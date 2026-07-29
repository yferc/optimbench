import pytest

from optimbench.agents import GreedyDispatcher
from optimbench.domain import ActionType, Difficulty, Note
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment

GEN = DispatchScenarioGenerator()


def _fresh() -> DispatchEnvironment:
    env = DispatchEnvironment()
    env.reset(GEN.generate(1, Difficulty.EASY))
    return env


def test_reset_returns_observation():
    obs = DispatchEnvironment().reset(GEN.generate(0, Difficulty.EASY))
    assert "vehicles" in obs and "unassigned_orders" in obs


def test_assign_then_unassign_roundtrip():
    env = _fresh()
    order = env.observation()["unassigned_orders"][0]["id"]
    vehicle = env.observation()["vehicles"][0]["id"]
    env.step(ActionType.ASSIGN_ORDER, {"order_id": order, "vehicle_id": vehicle})
    assert order in env.state.vehicles[vehicle].assigned
    env.step(ActionType.UNASSIGN_ORDER, {"order_id": order})
    assert order not in env.state.vehicles[vehicle].assigned


def test_assign_to_out_of_service_is_rejected():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.HARD))
    offline = next(v for v in env.state.vehicles.values() if not v.in_service)
    order = env.observation()["unassigned_orders"][0]["id"]
    out = env.step(ActionType.ASSIGN_ORDER, {"order_id": order, "vehicle_id": offline.id})
    assert out["accepted"] is False


def test_rejected_action_surfaces_note_in_next_observation():
    # a rejected action tells the agent why in the following observation, so it can recover
    # instead of blindly repeating the invalid call
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.HARD))
    offline = next(v for v in env.state.vehicles.values() if not v.in_service)
    order = env.observation()["unassigned_orders"][0]["id"]
    env.step(ActionType.ASSIGN_ORDER, {"order_id": order, "vehicle_id": offline.id})
    last = env.observation()["last_action"]
    assert last["accepted"] is False
    assert last["note"] == Note.VEHICLE_OUT_OF_SERVICE.value


def test_reset_and_clean_read_report_no_last_action():
    env = _fresh()
    assert "last_action" not in env.observation()
    env.step(ActionType.LIST_ORDERS, {"filter": "all"})
    assert "last_action" not in env.observation()


def test_reroute_covers_assigned_nodes():
    env = _fresh()
    order = env.observation()["unassigned_orders"][0]
    vehicle = env.observation()["vehicles"][0]["id"]
    env.step(ActionType.ASSIGN_ORDER, {"order_id": order["id"], "vehicle_id": vehicle})
    env.step(ActionType.REROUTE, {"vehicle_id": vehicle})
    assert order["node"] in env.state.vehicles[vehicle].route


def test_dispatch_advances_waves_then_finishes():
    env = _fresh()
    waves = len(env.scenario.disruptions)
    for _ in range(waves + 1):
        env.step(ActionType.DISPATCH, {})
    assert env.done


def test_breakdown_unassigns_vehicle_orders():
    env = DispatchEnvironment()
    env.reset(GEN.generate(1, Difficulty.EASY))
    agent = GreedyDispatcher()
    while env.state.wave == 0 and not env.done:
        action, args = agent.act(env.observation())
        env.step(action, args)
    offline = [v for v in env.state.vehicles.values() if not v.in_service]
    assert all(not v.assigned for v in offline)


def test_set_route_rejects_out_of_range_stops():
    env = _fresh()
    vehicle = env.observation()["vehicles"][0]["id"]
    out = env.step(ActionType.SET_ROUTE, {"vehicle_id": vehicle, "stops": [0, 999, 0]})
    assert out["accepted"] is False


def test_query_traffic_rejects_out_of_range_node():
    env = _fresh()
    assert env.step(ActionType.QUERY_TRAFFIC, {"a": 0, "b": 999})["accepted"] is False


def test_trajectory_logs_every_decision():
    env = _fresh()
    env.step(ActionType.LIST_ORDERS, {"filter": "live"})
    env.step(ActionType.CHECK_FEASIBILITY, {})
    assert len(env.trajectory.decisions) == 2


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        DispatchEnvironment().step(ActionType.LIST_ORDERS, {})


def test_random_policy_runs_to_termination():
    from optimbench.agents import RandomDispatcher

    env = _fresh()
    agent = RandomDispatcher(seed=0)
    steps = 0
    while not env.done and steps < 5000:
        env.step(*agent.act(env.observation()))
        steps += 1
    assert env.done
