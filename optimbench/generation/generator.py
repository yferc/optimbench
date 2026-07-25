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
_SHIFT_SLACK = 90.0
_WINDOW_SLACK = 60


class DispatchScenarioGenerator:
    def generate(self, seed: int, difficulty: Difficulty) -> Scenario:
        rng = np.random.default_rng(seed)
        spec = DIFFICULTY[difficulty]
        network = self._network(rng, spec)
        vehicles, orders, reference_routes = self._feasible_construction(rng, spec, network)
        self._add_distractors(rng, spec, vehicles, orders)
        disruptions = self._disruptions(rng, spec, vehicles, orders, network)
        reference_time = total_fleet_time(network, reference_routes)
        state = DispatchState(network, orders, self._cleared(vehicles))
        return Scenario(seed, difficulty, state, disruptions, reference_time, ReferenceKind.HEURISTIC)

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
    ) -> tuple[list[Vehicle], dict[str, Order], list[list[int]]]:
        nodes = list(range(1, spec.nodes))
        rng.shuffle(nodes)
        quota = self._quota(spec.orders, spec.vehicles)

        vehicles: list[Vehicle] = []
        orders: dict[str, Order] = {}
        reference_routes: list[list[int]] = []
        cursor = 0

        for index in range(spec.vehicles):
            vehicle = Vehicle(f"veh_{index}", spec.capacity, DEPOT, 0)
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
                order = Order(
                    id=f"ord_{len(orders)}",
                    node=node,
                    demand=demand,
                    window_open=max(0, int(arrival - rng.integers(5, 20))),
                    window_close=int(arrival + rng.integers(20, _WINDOW_SLACK)),
                    service_time=int(rng.integers(1, 4)),
                )
                orders[order.id] = order
                vehicle.assigned.append(order.id)
                route.append(node)
                load += demand
                clock = arrival + order.service_time
            route.append(DEPOT)
            clock += network.true_time[route[-2], DEPOT]
            vehicle.shift_end = int(clock + _SHIFT_SLACK)
            vehicle.route = route
            vehicles.append(vehicle)
            reference_routes.append(route)
        return vehicles, orders, reference_routes

    def _add_distractors(
        self, rng: np.random.Generator, spec: DifficultySpec,
        vehicles: list[Vehicle], orders: dict[str, Order],
    ) -> None:
        for index in range(spec.cancelled_distractors):
            node = int(rng.integers(1, spec.nodes))
            orders[f"ord_cancel_{index}"] = Order(
                id=f"ord_cancel_{index}", node=node, demand=int(rng.integers(1, 4)),
                window_open=0, window_close=10_000, service_time=1,
                status=OrderStatus.CANCELLED,
            )
        for index in range(spec.out_of_service):
            vehicles.append(Vehicle(
                f"veh_offline_{index}", spec.capacity, DEPOT, 10_000, in_service=False))

    def _disruptions(
        self, rng: np.random.Generator, spec: DifficultySpec,
        vehicles: list[Vehicle], orders: dict[str, Order], network: RoadNetwork,
    ) -> list[Disruption]:
        plan = [DisruptionKind.BREAKDOWN, DisruptionKind.RUSH_ORDER, DisruptionKind.CANCELLATION]
        live = [v for v in vehicles if v.in_service and v.assigned]
        disruptions: list[Disruption] = []
        for kind in plan[: spec.waves]:
            if kind is DisruptionKind.BREAKDOWN and live:
                disruptions.append(Disruption(kind, vehicle_id=live[0].id))
            elif kind is DisruptionKind.RUSH_ORDER:
                node = int(rng.integers(1, spec.nodes))
                disruptions.append(Disruption(kind, order=Order(
                    id="ord_rush", node=node, demand=int(rng.integers(1, 3)),
                    window_open=0, window_close=int(network.true_time[DEPOT, node] * 3 + 30),
                    service_time=2, priority=Priority.RUSH)))
            elif kind is DisruptionKind.CANCELLATION and orders:
                target = next(iter(orders))
                disruptions.append(Disruption(kind, order_id=target))
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
