from __future__ import annotations

import numpy as np

from ..domain import (
    Difficulty,
    DispatchState,
    DisruptionKind,
    Order,
    OrderStatus,
    Priority,
    ReferenceKind,
    RoadNetwork,
    Vehicle,
    euclidean_time_matrix,
    total_fleet_time,
)
from .scenario import DIFFICULTY, DifficultySpec, Disruption, Scenario

DEPOT = 0
_HORIZON_MARGIN = 1.8
_HORIZON_BUFFER = 30.0
_SHIFT_HORIZON = 10_000


class DispatchScenarioGenerator:
    def generate(self, seed: int, difficulty: Difficulty) -> Scenario:
        rng = np.random.default_rng(seed)
        spec = DIFFICULTY[difficulty]
        network = self._network(rng, spec)
        vehicles, orders, routes, horizon = self._feasible_construction(rng, spec, network)
        self._add_distractors(rng, spec, vehicles, orders, horizon)
        disruptions = self._disruptions(rng, spec, vehicles, orders, horizon)
        state = DispatchState(network, orders, self._cleared(vehicles))
        return Scenario(
            seed, difficulty, state, disruptions,
            total_fleet_time(network, routes), ReferenceKind.HEURISTIC,
        )

    def _network(self, rng: np.random.Generator, spec: DifficultySpec) -> RoadNetwork:
        coordinates = rng.random((spec.nodes, 2)) * 100.0
        coordinates[DEPOT] = [50.0, 50.0]
        true_time = euclidean_time_matrix(coordinates)
        observed_time = true_time.copy()
        stale = rng.random(true_time.shape) < spec.stale_fraction
        observed_time[stale] *= rng.uniform(0.6, 1.5, true_time.shape)[stale]
        np.fill_diagonal(observed_time, 0.0)
        return RoadNetwork(coordinates, true_time, observed_time)

    def _feasible_construction(
        self, rng: np.random.Generator, spec: DifficultySpec, network: RoadNetwork
    ) -> tuple[list[Vehicle], dict[str, Order], list[list[int]], float]:
        nodes = list(range(1, spec.nodes))
        rng.shuffle(nodes)
        quota = self._quota(spec.orders, spec.vehicles)
        placements: list[tuple[int, int, int, int, float]] = []
        vehicles: list[Vehicle] = []
        routes: list[list[int]] = []
        cursor = 0

        for index in range(spec.vehicles):
            vehicle = Vehicle(f"veh_{index}", spec.capacity, DEPOT, _SHIFT_HORIZON)
            route = [DEPOT]
            clock = 0.0
            load = 0
            for _ in range(quota[index]):
                if cursor >= len(nodes):
                    break
                node = nodes[cursor]
                cursor += 1
                demand = int(rng.integers(1, 4))
                if load + demand > vehicle.capacity:
                    break
                arrival = clock + network.true_time[route[-1], node]
                service = int(rng.integers(1, 4))
                placements.append((index, node, demand, service, arrival))
                route.append(node)
                load += demand
                clock = arrival + service
            route.append(DEPOT)
            vehicle.route = route
            vehicles.append(vehicle)
            routes.append(route)

        horizon = max((p[4] for p in placements), default=0.0) * _HORIZON_MARGIN + _HORIZON_BUFFER
        orders: dict[str, Order] = {}
        for i, (vehicle_index, node, demand, service, _arrival) in enumerate(placements):
            order = Order(f"ord_{i}", node, demand, 0, int(horizon), service)
            orders[order.id] = order
            vehicles[vehicle_index].assigned.append(order.id)
        return vehicles, orders, routes, horizon

    def _add_distractors(
        self, rng: np.random.Generator, spec: DifficultySpec,
        vehicles: list[Vehicle], orders: dict[str, Order], horizon: float,
    ) -> None:
        for index in range(spec.cancelled_distractors):
            node = int(rng.integers(1, spec.nodes))
            orders[f"ord_cancel_{index}"] = Order(
                f"ord_cancel_{index}", node, int(rng.integers(1, 4)),
                0, int(horizon), 1, status=OrderStatus.CANCELLED)
        for index in range(spec.out_of_service):
            vehicles.append(
                Vehicle(f"veh_offline_{index}", spec.capacity, DEPOT, _SHIFT_HORIZON, in_service=False))

    def _disruptions(
        self, rng: np.random.Generator, spec: DifficultySpec,
        vehicles: list[Vehicle], orders: dict[str, Order], horizon: float,
    ) -> list[Disruption]:
        plan = [DisruptionKind.BREAKDOWN, DisruptionKind.RUSH_ORDER, DisruptionKind.CANCELLATION]
        live_vehicles = [v for v in vehicles if v.in_service and v.assigned]
        disruptions: list[Disruption] = []
        for kind in plan[: spec.waves]:
            if kind is DisruptionKind.BREAKDOWN and live_vehicles:
                disruptions.append(Disruption(kind, vehicle_id=live_vehicles[0].id))
            elif kind is DisruptionKind.RUSH_ORDER:
                node = int(rng.integers(1, spec.nodes))
                disruptions.append(Disruption(kind, order=Order(
                    "ord_rush", node, int(rng.integers(1, 3)),
                    0, int(horizon), 2, priority=Priority.RUSH)))
            elif kind is DisruptionKind.CANCELLATION and orders:
                disruptions.append(Disruption(kind, order_id=next(iter(orders))))
        return disruptions

    @staticmethod
    def _quota(orders: int, vehicles: int) -> list[int]:
        base, extra = divmod(orders, vehicles)
        return [base + (1 if i < extra else 0) for i in range(vehicles)]

    @staticmethod
    def _cleared(vehicles: list[Vehicle]) -> dict[str, Vehicle]:
        return {
            v.id: Vehicle(v.id, v.capacity, v.depot, v.shift_end, v.in_service)
            for v in vehicles
        }
