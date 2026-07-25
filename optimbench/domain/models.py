from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .enums import OrderStatus, Priority


@dataclass(frozen=True)
class Order:
    id: str
    node: int
    demand: int
    window_open: int
    window_close: int
    service_time: int
    priority: Priority = Priority.NORMAL
    status: OrderStatus = OrderStatus.LIVE

    def with_status(self, status: OrderStatus) -> "Order":
        return replace(self, status=status)


@dataclass
class Vehicle:
    id: str
    capacity: int
    depot: int
    shift_end: int
    in_service: bool = True
    assigned: list[str] = field(default_factory=list)
    route: list[int] = field(default_factory=list)

    def load(self, orders: dict[str, Order]) -> int:
        return sum(orders[o].demand for o in self.assigned if o in orders)


@dataclass(frozen=True, eq=False)
class RoadNetwork:
    coordinates: np.ndarray          # (n, 2)
    true_time: np.ndarray            # (n, n) ground-truth travel time
    observed_time: np.ndarray        # (n, n) cached, occasionally stale

    @property
    def size(self) -> int:
        return len(self.coordinates)


@dataclass
class DispatchState:
    network: RoadNetwork
    orders: dict[str, Order]
    vehicles: dict[str, Vehicle]
    wave: int = 0
    committed: bool = False

    def live_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.status is OrderStatus.LIVE]

    def assigned_ids(self) -> set[str]:
        return {oid for v in self.vehicles.values() for oid in v.assigned}
