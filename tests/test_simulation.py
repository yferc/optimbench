import pytest

from optimbench.agents import GreedyDispatcher
from optimbench.domain import ActionType, Difficulty
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


def test_query_traffic_rejects_bad_node():
    env = _fresh()
    assert env.step(ActionType.QUERY_TRAFFIC, {"a": 0, "b": 999})["accepted"] is False
    assert env.step(ActionType.QUERY_TRAFFIC, {"a": 0})["accepted"] is False


def test_trajectory_logs_every_decision():
    env = _fresh()
    env.step(ActionType.LIST_ORDERS, {"filter": "live"})
    env.step(ActionType.CHECK_FEASIBILITY, {})
    assert len(env.trajectory.decisions) == 2


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        DispatchEnvironment().step(ActionType.LIST_ORDERS, {})


def test_malformed_args_are_rejected_not_crashing():
    env = _fresh()
    unhashable = [
        (ActionType.LIST_ORDERS, {"filter": [1]}),
        (ActionType.GET_VEHICLE, {"vehicle_id": [1]}),
        (ActionType.ASSIGN_ORDER, {"order_id": {}, "vehicle_id": "veh_0"}),
        (ActionType.SET_ROUTE, {"vehicle_id": [1], "stops": [0, 1, 0]}),
        (ActionType.REROUTE, {"vehicle_id": {}}),
        (ActionType.UNASSIGN_ORDER, {"order_id": [1]}),
    ]
    for action, args in unhashable:
        assert env.step(action, args)["accepted"] is False


def test_as_node_rejects_bool_float_and_string():
    env = _fresh()
    for bad in (True, 1.9, "3"):
        assert env.step(ActionType.QUERY_TRAFFIC, {"a": bad, "b": 2})["accepted"] is False
