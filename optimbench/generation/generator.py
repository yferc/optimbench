from __future__ import annotations

import numpy as np

from ..domain import (
    DEPOT,
    Difficulty,
    Disruption,
    DisruptionKind,
    DispatchState,
    Order,
    OrderStatus,
    Priority,
    ReferenceKind,
    RoadNetwork,
    Scenario,
    Vehicle,
    euclidean_time_matrix,
    total_fleet_time,
)
from .scenario import DIFFICULTY, DifficultySpec

_RUSH_RESERVE = 3
_HORIZON_BUFFER = 60.0


class DispatchScenarioGenerator:
    def generate(self, seed: int, difficulty: Difficulty) -> Scenario:
        rng = np.random.default_rng(seed)
        spec = DIFFICULTY[difficulty]
        network = self._network(rng, spec)
        horizon = self._horizon(network, spec)
        vehicles, orders, routes = self._feasible_construction(rng, spec, network, horizon)
        self._add_distractors(rng, spec, vehicles, orders, horizon)
        disruptions = self._disruptions(rng, spec, vehicles, orders, horizon)
        state = DispatchState(network, orders, self._cleared(vehicles, horizon))
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

    def _horizon(self, network: RoadNetwork, spec: DifficultySpec) -> float:
        out_and_back = 2.0 * float(network.true_time[DEPOT, 1:].sum())
        max_service = 3.0 * spec.orders
        return out_and_back + max_service + _HORIZON_BUFFER

    def _demand_budget(self, spec: DifficultySpec) -> float:
        usable = (spec.vehicles - 1) * spec.capacity - _RUSH_RESERVE
        return usable / spec.slack_ratio

    def _feasible_construction(
        self, rng: np.random.Generator, spec: DifficultySpec,
        network: RoadNetwork, horizon: float,
    ) -> tuple[list[Vehicle], dict[str, Order], list[list[int]]]:
        nodes = list(range(1, spec.nodes))
        rng.shuffle(nodes)
        quota = self._quota(spec.orders, spec.vehicles)
        budget = self._demand_budget(spec)

        vehicles: list[Vehicle] = []
        orders: dict[str, Order] = {}
        routes: list[list[int]] = []
        cursor = 0
        total_demand = 0

        for index in range(spec.vehicles):
            vehicle = Vehicle(f"veh_{index}", spec.capacity, DEPOT, int(horizon))
            route = [DEPOT]
            load = 0
            for _ in range(quota[index]):
                if cursor >= len(nodes):
                    break
                demand = int(rng.integers(1, 4))
                if load + demand > vehicle.capacity or total_demand + demand > budget:
                    break
                node = nodes[cursor]
                cursor += 1
                order = Order(f"ord_{len(orders)}", node, demand, 0, int(horizon),
                              int(rng.integers(1, 4)))
                orders[order.id] = order
                vehicle.assigned.append(order.id)
                route.append(node)
                load += demand
                total_demand += demand
            route.append(DEPOT)
            vehicle.route = route
            vehicles.append(vehicle)
            routes.append(route)
        return vehicles, orders, routes

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
                Vehicle(f"veh_offline_{index}", spec.capacity, DEPOT, int(horizon), in_service=False))

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
    def _cleared(vehicles: list[Vehicle], horizon: float) -> dict[str, Vehicle]:
        return {
            v.id: Vehicle(v.id, v.capacity, v.depot, int(horizon), v.in_service)
            for v in vehicles
        }
