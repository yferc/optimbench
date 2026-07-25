import numpy as np
import pytest

from optimbench.domain import (
    DispatchState,
    Order,
    RoadNetwork,
    Vehicle,
    ViolationKind,
    euclidean_time_matrix,
    is_feasible,
    route_time,
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
    kinds = {v.kind for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationKind.CAPACITY_EXCEEDED in kinds


def test_unassigned_live_order_detected(network):
    orders = {"o1": _order("o1", 1)}
    veh = Vehicle("v1", 10, 0, 10_000)
    kinds = {v.kind for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationKind.UNASSIGNED_LIVE_ORDER in kinds


def test_missing_route_stop_detected(network):
    orders = {"o1": _order("o1", 1)}
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[0, 2, 0])
    kinds = {v.kind for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationKind.ROUTE_MISSING_STOP in kinds


def test_time_window_violation_detected(network):
    orders = {"o1": _order("o1", 1, close=1)}
    veh = Vehicle("v1", 10, 0, 10_000, assigned=["o1"], route=[0, 1, 0])
    kinds = {v.kind for v in violations(DispatchState(network, orders, {"v1": veh}))}
    assert ViolationKind.TIME_WINDOW_MISSED in kinds
