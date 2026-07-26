from __future__ import annotations

import math

import numpy as np

from optimbench.domain.geometry import route_time
from optimbench.domain.models import DEPOT, DispatchState, Vehicle

_IMPROVEMENT_EPS = 1e-9  # only accept a 2-opt swap that beats the current tour by more than float noise


def reference_cost(state: DispatchState) -> float:
    """Fleet time of a strong deterministic dispatch of the live orders over the
    in-service fleet: sweep assignment, then nearest-neighbour + 2-opt routing.
    Independent of how the agent assigned or routed; task_score measures the
    agent's committed cost against this canonical solution of the same instance."""
    vehicles = [v for v in state.vehicles.values() if v.in_service]
    if not vehicles:
        return 0.0
    times = state.network.true_time
    return float(sum(
        route_time(state.network, _two_opt(times, _nearest_neighbour(times, nodes)))
        for nodes in _sweep_assign(state, vehicles)
        if nodes
    ))


def _sweep_assign(state: DispatchState, vehicles: list[Vehicle]) -> list[list[int]]:
    ordered = sorted(state.live_orders(), key=lambda o: (_angle(state, o.node), o.id))
    capacities = [v.capacity for v in vehicles]
    loads = [0] * len(vehicles)
    groups: list[list[int]] = [[] for _ in vehicles]
    for order in ordered:
        target = _first_vehicle_with_room(loads, capacities, order.demand)
        groups[target].append(order.node)
        loads[target] += order.demand
    return [sorted(set(group)) for group in groups]


def _first_vehicle_with_room(loads: list[int], capacities: list[int], demand: int) -> int:
    for index, (load, capacity) in enumerate(zip(loads, capacities)):
        if load + demand <= capacity:
            return index
    return min(range(len(loads)), key=lambda k: loads[k])


def _angle(state: DispatchState, node: int) -> float:
    x, y = state.network.coordinates[node] - state.network.coordinates[DEPOT]
    return math.atan2(y, x)


def _nearest_neighbour(times: np.ndarray, nodes: list[int]) -> list[int]:
    route = [DEPOT]
    remaining = list(nodes)
    while remaining:
        here = route[-1]
        nearest = min(remaining, key=lambda node: times[here, node])
        route.append(nearest)
        remaining.remove(nearest)
    route.append(DEPOT)
    return route


def _two_opt(times: np.ndarray, route: list[int]) -> list[int]:
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                if _cost(times, candidate) + _IMPROVEMENT_EPS < _cost(times, best):
                    best, improved = candidate, True
    return best


def _cost(times: np.ndarray, route: list[int]) -> float:
    stops = np.asarray(route, dtype=int)
    return float(times[stops[:-1], stops[1:]].sum())
