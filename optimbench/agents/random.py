from __future__ import annotations

import random
from typing import Any

from ..domain import ActionType


class RandomDispatcher:
    """Uniform random policy over the parameterized action space, with arguments
    sampled from the current observation. The zero-skill baseline — handy for
    sanity-checking the environment and seeing what an unstructured rollout does."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def act(self, observation: dict[str, Any]) -> tuple[ActionType, dict[str, Any]]:
        action = self._rng.choice(list(ActionType))
        return action, self._args(action, observation)

    def _args(self, action: ActionType, obs: dict[str, Any]) -> dict[str, Any]:
        vehicle_ids = [v["id"] for v in obs["vehicles"]]
        assigned_ids = [o for v in obs["vehicles"] for o in v["assigned"]]
        order_ids = [o["id"] for o in obs["unassigned_orders"]] + assigned_ids
        nodes = self._nodes(obs)

        if action is ActionType.LIST_ORDERS:
            return {"filter": self._rng.choice(["live", "unassigned", "rush", "all"])}
        if action is ActionType.GET_VEHICLE:
            return {"vehicle_id": self._pick(vehicle_ids)}
        if action is ActionType.QUERY_TRAFFIC:
            return {"a": self._pick(nodes, 0), "b": self._pick(nodes, 0)}
        if action is ActionType.ASSIGN_ORDER:
            return {"order_id": self._pick(order_ids), "vehicle_id": self._pick(vehicle_ids)}
        if action is ActionType.UNASSIGN_ORDER:
            return {"order_id": self._pick(assigned_ids)}
        if action is ActionType.SET_ROUTE:
            return {"vehicle_id": self._pick(vehicle_ids), "stops": self._route(nodes)}
        if action is ActionType.REROUTE:
            return {"vehicle_id": self._pick(vehicle_ids)}
        if action is ActionType.REFUSE:
            return {"reason": "random"}
        return {}  # CHECK_FEASIBILITY, DISPATCH

    @staticmethod
    def _nodes(obs: dict[str, Any]) -> list[int]:
        seen = {0}
        seen.update(o["node"] for o in obs["unassigned_orders"])
        seen.update(node for v in obs["vehicles"] for node in v["route"])
        return sorted(seen)

    def _pick(self, pool: list, default: Any = None) -> Any:
        return self._rng.choice(pool) if pool else default

    def _route(self, nodes: list[int]) -> list[int]:
        k = self._rng.randint(1, min(4, len(nodes)))
        return [0, *self._rng.sample(nodes, k), 0]
