from __future__ import annotations

from dataclasses import dataclass

from .enums import OrderStatus, ViolationKind
from .geometry import arrival_times
from .models import DispatchState, Vehicle


@dataclass(frozen=True)
class Violation:
    kind: ViolationKind
    ref: str


def violations(state: DispatchState) -> list[Violation]:
    found: list[Violation] = []
    found += _unassigned_live_orders(state)
    for vehicle in state.vehicles.values():
        found += _vehicle_violations(state, vehicle)
    return found


def is_feasible(state: DispatchState) -> bool:
    return not violations(state)


def _unassigned_live_orders(state: DispatchState) -> list[Violation]:
    assigned = state.assigned_ids()
    return [
        Violation(ViolationKind.UNASSIGNED_LIVE_ORDER, order.id)
        for order in state.live_orders()
        if order.id not in assigned
    ]


def _vehicle_violations(state: DispatchState, vehicle: Vehicle) -> list[Violation]:
    if not vehicle.assigned:
        return []
    if not vehicle.in_service:
        return [Violation(ViolationKind.OUT_OF_SERVICE_VEHICLE, vehicle.id)]

    found: list[Violation] = []
    if vehicle.load(state.orders) > vehicle.capacity:
        found.append(Violation(ViolationKind.CAPACITY_EXCEEDED, vehicle.id))

    route_nodes = set(vehicle.route)
    arrival_at = _arrival_by_node(state, vehicle)
    for order_id in vehicle.assigned:
        order = state.orders[order_id]
        if order.node not in route_nodes:
            found.append(Violation(ViolationKind.ROUTE_MISSING_STOP, vehicle.id))
            continue
        if arrival_at.get(order.node, float("inf")) > order.window_close:
            found.append(Violation(ViolationKind.TIME_WINDOW_MISSED, order_id))
    return found


def _arrival_by_node(state: DispatchState, vehicle: Vehicle) -> dict[int, float]:
    times = arrival_times(state.network, vehicle.route)
    return {node: float(t) for node, t in zip(vehicle.route, times)}
