import numpy as np
import pytest

from optimbench.domain import (
    DispatchState,
    Order,
    RoadNetwork,
    Vehicle,
    ViolationType,
    euclidean_time_matrix,
    is_feasible,
    route_time,
    schedule,
    violations,
)


@pytest.fixture
def network() -> RoadNetwork:
    coords = np.array([[0, 0], [0, 3], [4, 0]], dtype=float)
    times = euclidean_time_matrix(coords)
    return RoadNetwork(coords, times, times.copy())


def _order(oid: str, node: int, demand: int = 1, close: int = 10_000) -> Order:
    return Order(oid, node, demand, 0, close, 0)


def test_route_time_is_sum_of_legs(network):
    assert route_time(network, [0, 1, 2]) == pytest.approx(3.0 + 5.0)
    assert route_time(network, [0]) == 0.0


def test_euclidean_matrix_symmetric_zero_diagonal(network):
    assert np.allclose(network.true_time, network.true_time.T)
    assert np.allclose(np.diag(network.true_time), 0.0)


def test_feasible_state_has_no_violations(network):
    orders = {"o1": _order("o1", 1)}
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[0, 1, 0])
    assert is_feasible(DispatchState(network, orders, {"v1": veh}))


def test_capacity_violation_detected(network):
    orders = {"o1": _order("o1", 1, demand=99)}
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[0, 1, 0])
    violation_types = {v.type for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationType.CAPACITY_EXCEEDED in violation_types


def test_unassigned_live_order_detected(network):
    orders = {"o1": _order("o1", 1)}
    veh = Vehicle("v1", 10, 0, 10_000)
    violation_types = {v.type for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationType.UNASSIGNED_LIVE_ORDER in violation_types


def test_missing_route_stop_detected(network):
    orders = {"o1": _order("o1", 1)}
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[0, 2, 0])
    violation_types = {v.type for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationType.ROUTE_MISSING_STOP in violation_types


def test_time_window_violation_detected(network):
    orders = {"o1": _order("o1", 1, close=1)}
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[0, 1, 0])
    violation_types = {v.type for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationType.TIME_WINDOW_MISSED in violation_types


def test_depot_anchoring_required(network):
    orders = {"o1": _order("o1", 1)}
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[1, 2])
    violation_types = {v.type for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationType.ROUTE_NOT_DEPOT_ANCHORED in violation_types


def test_out_of_range_route_flagged_without_crashing(network):
    orders = {"o1": _order("o1", 1)}
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[0, 999, 0])
    violation_types = {v.type for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationType.ROUTE_MISSING_STOP in violation_types


def test_shift_end_exceeded_detected(network):
    orders = {"o1": _order("o1", 1)}
    veh = Vehicle("v1", 10, 0, 1, assigned=["o1"], route=[0, 1, 0])
    violation_types = {v.type for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationType.SHIFT_END_EXCEEDED in violation_types


def test_schedule_handles_empty_route(network):
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[])
    arrival, return_time = schedule(DispatchState(network, {}, {"v1": veh}), veh)
    assert arrival == {} and return_time == 0.0


def test_violations_tolerate_ghost_assigned_id(network):
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["ghost"], route=[0, 1, 0])
    assert is_feasible(DispatchState(network, {}, {"v1": veh}))
