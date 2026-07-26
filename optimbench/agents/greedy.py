from __future__ import annotations

from typing import Any

from optimbench.domain import ActionType, Arg, Field


class GreedyDispatcher:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._rerouted: set[str] = set()

    def act(self, observation: dict[Field, Any]) -> tuple[ActionType, dict[Arg, Any]]:
        vehicles = [v for v in observation[Field.VEHICLES] if v[Field.IN_SERVICE]]
        unassigned = observation[Field.UNASSIGNED_ORDERS]

        if unassigned and vehicles:
            self._rerouted.clear()
            order = unassigned[0]
            vehicle = self._best_fit(order, vehicles)
            return ActionType.ASSIGN_ORDER, {Arg.ORDER_ID: order[Field.ID], Arg.VEHICLE_ID: vehicle[Field.ID]}

        pending = [v for v in vehicles if v[Field.ASSIGNED] and v[Field.ID] not in self._rerouted]
        if pending:
            self._rerouted.add(pending[0][Field.ID])
            return ActionType.REROUTE, {Arg.VEHICLE_ID: pending[0][Field.ID]}

        return ActionType.DISPATCH, {}

    @staticmethod
    def _best_fit(order: dict[Field, Any], vehicles: list[dict[Field, Any]]) -> dict[Field, Any]:
        def headroom(vehicle: dict[Field, Any]) -> int:
            return vehicle[Field.CAPACITY] - vehicle[Field.LOAD]

        fitting = [v for v in vehicles if headroom(v) >= order[Field.DEMAND]]
        return max(fitting if fitting else vehicles, key=headroom)
