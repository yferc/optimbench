from __future__ import annotations

import random
from typing import Any

from optimbench.domain import DEPOT, TOOLSET, ActionType, Arg, Field, OrderFilter

_TOOL_ACTIONS = [tool.action for tool in TOOLSET]


class RandomDispatcher:
    """Uniform random policy over the parameterized action space, with arguments
    sampled from the current observation. The zero-skill baseline, handy for
    sanity-checking the environment and seeing what an unstructured rollout does."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def act(self, observation: dict[Field, Any]) -> tuple[ActionType, dict[Arg, Any]]:
        action = self._rng.choice(_TOOL_ACTIONS)
        return action, self._args(action, observation)

    def _args(self, action: ActionType, obs: dict[Field, Any]) -> dict[Arg, Any]:
        vehicle_ids = [v[Field.ID] for v in obs[Field.VEHICLES]]
        assigned_ids = [o for v in obs[Field.VEHICLES] for o in v[Field.ASSIGNED]]
        order_ids = [o[Field.ID] for o in obs[Field.UNASSIGNED_ORDERS]] + assigned_ids
        nodes = self._nodes(obs)

        if action is ActionType.LIST_ORDERS:
            return {Arg.FILTER: self._rng.choice(list(OrderFilter)).value}
        if action is ActionType.GET_VEHICLE:
            return {Arg.VEHICLE_ID: self._pick(vehicle_ids)}
        if action is ActionType.QUERY_TRAFFIC:
            return {Arg.NODE_A: self._pick(nodes, 0), Arg.NODE_B: self._pick(nodes, 0)}
        if action is ActionType.ASSIGN_ORDER:
            return {Arg.ORDER_ID: self._pick(order_ids), Arg.VEHICLE_ID: self._pick(vehicle_ids)}
        if action is ActionType.UNASSIGN_ORDER:
            return {Arg.ORDER_ID: self._pick(assigned_ids)}
        if action is ActionType.SET_ROUTE:
            return {Arg.VEHICLE_ID: self._pick(vehicle_ids), Arg.STOPS: self._route(nodes)}
        if action is ActionType.REROUTE:
            return {Arg.VEHICLE_ID: self._pick(vehicle_ids)}
        if action is ActionType.REFUSE:
            return {Arg.REASON: "random"}
        return {}  # CHECK_FEASIBILITY, DISPATCH

    @staticmethod
    def _nodes(obs: dict[Field, Any]) -> list[int]:
        seen = {DEPOT}
        seen.update(o[Field.NODE] for o in obs[Field.UNASSIGNED_ORDERS])
        seen.update(node for v in obs[Field.VEHICLES] for node in v[Field.ROUTE])
        return sorted(seen)

    def _pick(self, pool: list, default: Any = None) -> Any:
        return self._rng.choice(pool) if pool else default

    def _route(self, nodes: list[int]) -> list[int]:
        n_stops = self._rng.randint(1, min(4, len(nodes)))
        return [DEPOT, *self._rng.sample(nodes, n_stops), DEPOT]
