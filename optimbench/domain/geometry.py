from __future__ import annotations

import numpy as np

from optimbench.domain.models import DispatchState, RoadNetwork


def euclidean_time_matrix(coordinates: np.ndarray, speed: float = 1.0) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt((delta**2).sum(axis=-1)) / speed


def route_time(network: RoadNetwork, route: list[int]) -> float:
    if len(route) < 2:
        return 0.0
    stops = np.asarray(route, dtype=int)
    return float(network.true_time[stops[:-1], stops[1:]].sum())


def total_fleet_time(network: RoadNetwork, routes: list[list[int]]) -> float:
    return float(sum(route_time(network, route) for route in routes))


def fleet_cost(state: DispatchState) -> float:
    return float(sum(
        route_time(state.network, vehicle.route)
        for vehicle in state.vehicles.values()
        if vehicle.in_service
    ))
