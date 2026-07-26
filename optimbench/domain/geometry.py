from __future__ import annotations

import numpy as np

from optimbench.domain.models import DispatchState, RoadNetwork


def euclidean_time_matrix(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt((delta**2).sum(axis=-1))


def route_time(network: RoadNetwork, route: list[int]) -> float:
    if len(route) < 2:
        return 0.0
    stops = np.asarray(route, dtype=int)
    return float(network.true_time[stops[:-1], stops[1:]].sum())


def fleet_cost(state: DispatchState) -> float:
    return float(sum(
        route_time(state.network, vehicle.route)
        for vehicle in state.vehicles.values()
        if vehicle.in_service
    ))
