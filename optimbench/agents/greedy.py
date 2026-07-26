from __future__ import annotations

from typing import Any

from optimbench.domain import ActionType


class GreedyDispatcher:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._rerouted: set[str] = set()

    def act(self, observation: dict[str, Any]) -> tuple[ActionType, dict[str, Any]]:
        vehicles = [v for v in observation["vehicles"] if v["in_service"]]
        unassigned = observation["unassigned_orders"]

        if unassigned and vehicles:
            self._rerouted.clear()
            order = unassigned[0]
            vehicle = self._best_fit(order, vehicles)
            return ActionType.ASSIGN_ORDER, {"order_id": order["id"], "vehicle_id": vehicle["id"]}

        pending = [v for v in vehicles if v["assigned"] and v["id"] not in self._rerouted]
        if pending:
            self._rerouted.add(pending[0]["id"])
            return ActionType.REROUTE, {"vehicle_id": pending[0]["id"]}

        return ActionType.DISPATCH, {}

    @staticmethod
    def _best_fit(order: dict[str, Any], vehicles: list[dict[str, Any]]) -> dict[str, Any]:
        def headroom(vehicle: dict[str, Any]) -> int:
            return vehicle["capacity"] - vehicle["load"]

        fitting = [v for v in vehicles if headroom(v) >= order["demand"]]
        return max(fitting or vehicles, key=headroom)
