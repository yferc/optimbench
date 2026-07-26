from __future__ import annotations

from dataclasses import dataclass

from optimbench.domain.enums import ViolationType
from optimbench.domain.models import DEPOT, DispatchState, Order, Vehicle


@dataclass(frozen=True)
class Violation:
    type: ViolationType
    ref: str


def violations(state: DispatchState) -> list[Violation]:
    found: list[Violation] = []
    found += _unassigned_live_orders(state)
    for vehicle in state.vehicles.values():
        found += _vehicle_violations(state, vehicle)
    return found


def is_feasible(state: DispatchState) -> bool:
    return not violations(state)


def schedule(state: DispatchState, vehicle: Vehicle) -> tuple[dict[int, float], float]:
    """Arrival time at each stop and the return-to-depot time, with travel,
    wait-until-window_open and service time. This is the executable timeline."""
    if len(vehicle.route) < 2:
        return {}, 0.0
    service_at, open_at = _stop_timing(state, vehicle)
    arrival: dict[int, float] = {}
    clock = 0.0
    previous = vehicle.route[0]
    for node in vehicle.route:
        clock += state.network.true_time[previous, node]
        if node not in arrival:
            arrival[node] = clock
        clock = max(clock, open_at.get(node, clock)) + service_at.get(node, 0.0)
        previous = node
    return arrival, clock


def _unassigned_live_orders(state: DispatchState) -> list[Violation]:
    assigned = state.assigned_ids()
    return [
        Violation(ViolationType.UNASSIGNED_LIVE_ORDER, order.id)
        for order in state.live_orders()
        if order.id not in assigned
    ]


def _vehicle_violations(state: DispatchState, vehicle: Vehicle) -> list[Violation]:
    if not vehicle.assigned:
        return []
    if not vehicle.in_service:
        return [Violation(ViolationType.OUT_OF_SERVICE_VEHICLE, vehicle.id)]
    if _route_out_of_bounds(state, vehicle):
        return [Violation(ViolationType.ROUTE_MISSING_STOP, vehicle.id)]

    found: list[Violation] = []
    if vehicle.load(state.orders) > vehicle.capacity:
        found.append(Violation(ViolationType.CAPACITY_EXCEEDED, vehicle.id))
    if not _depot_anchored(vehicle):
        return found + [Violation(ViolationType.ROUTE_NOT_DEPOT_ANCHORED, vehicle.id)]

    route_nodes = set(vehicle.route)
    if any(order.node not in route_nodes for order in _assigned_orders(state, vehicle)):
        found.append(Violation(ViolationType.ROUTE_MISSING_STOP, vehicle.id))
        return found

    arrival, return_time = schedule(state, vehicle)
    for order in _assigned_orders(state, vehicle):
        if arrival[order.node] > order.window_close:
            found.append(Violation(ViolationType.TIME_WINDOW_MISSED, order.id))
    if return_time > vehicle.shift_end:
        found.append(Violation(ViolationType.SHIFT_END_EXCEEDED, vehicle.id))
    return found


def _depot_anchored(vehicle: Vehicle) -> bool:
    return len(vehicle.route) >= 2 and vehicle.route[0] == DEPOT and vehicle.route[-1] == DEPOT


def _route_out_of_bounds(state: DispatchState, vehicle: Vehicle) -> bool:
    size = state.network.size
    return any(node < 0 or node >= size for node in vehicle.route)


def _assigned_orders(state: DispatchState, vehicle: Vehicle) -> list[Order]:
    return [state.orders[o] for o in vehicle.assigned if o in state.orders]


def _stop_timing(state: DispatchState, vehicle: Vehicle):
    service_at: dict[int, float] = {}
    open_at: dict[int, float] = {}
    for order in _assigned_orders(state, vehicle):
        service_at[order.node] = service_at.get(order.node, 0.0) + order.service_time
        open_at[order.node] = max(open_at.get(order.node, 0.0), order.window_open)
    return service_at, open_at
