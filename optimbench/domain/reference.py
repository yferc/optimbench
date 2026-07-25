from __future__ import annotations

import numpy as np

from .geometry import route_time
from .models import DEPOT, DispatchState, Vehicle


def reference_cost(state: DispatchState) -> float:
    """Fleet time of a strong deterministic solution: each in-service vehicle's
    assigned stops routed by nearest-neighbour, then improved with 2-opt.
    task_score measures how close an agent's routing gets to this."""
    return float(sum(
        _best_route_cost(state, vehicle)
        for vehicle in state.vehicles.values()
        if vehicle.in_service
    ))


def _best_route_cost(state: DispatchState, vehicle: Vehicle) -> float:
    nodes = _assigned_nodes(state, vehicle)
    if not nodes:
        return 0.0
    times = state.network.true_time
    return route_time(state.network, _two_opt(times, _nearest_neighbour(times, nodes)))


def _assigned_nodes(state: DispatchState, vehicle: Vehicle) -> list[int]:
    return sorted({state.orders[o].node for o in vehicle.assigned if o in state.orders})


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
                if _cost(times, candidate) + 1e-9 < _cost(times, best):
                    best, improved = candidate, True
    return best


def _cost(times: np.ndarray, route: list[int]) -> float:
    stops = np.asarray(route, dtype=int)
    return float(times[stops[:-1], stops[1:]].sum())
