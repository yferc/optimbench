from __future__ import annotations

import numpy as np

from .models import RoadNetwork


def euclidean_time_matrix(coordinates: np.ndarray, speed: float = 1.0) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt((delta**2).sum(axis=-1)) / speed


def route_time(network: RoadNetwork, route: list[int], use_true: bool = True) -> float:
    if len(route) < 2:
        return 0.0
    stops = np.asarray(route, dtype=int)
    matrix = network.true_time if use_true else network.observed_time
    return float(matrix[stops[:-1], stops[1:]].sum())


def arrival_times(network: RoadNetwork, route: list[int], start: float = 0.0) -> np.ndarray:
    if not route:
        return np.zeros(0)
    stops = np.asarray(route, dtype=int)
    legs = network.true_time[stops[:-1], stops[1:]]
    return start + np.concatenate([[0.0], np.cumsum(legs)])


def total_fleet_time(network: RoadNetwork, routes: list[list[int]]) -> float:
    return float(sum(route_time(network, route) for route in routes))
