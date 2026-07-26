from __future__ import annotations

from typing import Any

import torch
from torch import nn

from optimbench.domain import ActionType, Priority

_ORDER_FEATURES = 4
_VEHICLE_FEATURES = 4
_WORLD = 100.0


class AssignmentPolicy(nn.Module):
    """Scores every (unassigned order, in-service vehicle) pair; the agent assigns
    the highest-scoring feasible pair. Routing and dispatch stay rule-based, so the
    learned signal is the geometric assignment that greedy's best-fit misses."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_ORDER_FEATURES + _VEHICLE_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.net(pairs).squeeze(-1)


class LearnedDispatcher:
    def __init__(self, policy: AssignmentPolicy, training: bool = False) -> None:
        self._policy = policy
        self._training = training
        self.reset()

    def reset(self) -> None:
        self._rerouted: set[str] = set()
        self.log_probs: list[torch.Tensor] = []

    def act(self, observation: dict[str, Any]) -> tuple[ActionType, dict[str, Any]]:
        vehicles = [v for v in observation["vehicles"] if v["in_service"]]
        unassigned = observation["unassigned_orders"]

        if unassigned and vehicles:
            self._rerouted.clear()
            return self._assign(unassigned, vehicles)

        pending = [v for v in vehicles if v["assigned"] and v["id"] not in self._rerouted]
        if pending:
            self._rerouted.add(pending[0]["id"])
            return ActionType.REROUTE, {"vehicle_id": pending[0]["id"]}

        return ActionType.DISPATCH, {}

    def _assign(self, unassigned, vehicles) -> tuple[ActionType, dict[str, Any]]:
        fitting = self._feasible_pairs(unassigned, vehicles)
        scores = self._policy(torch.stack([features for _, _, features in fitting]))
        order, vehicle, _ = fitting[self._choose(scores)]
        return ActionType.ASSIGN_ORDER, {"order_id": order["id"], "vehicle_id": vehicle["id"]}

    def _choose(self, scores: torch.Tensor) -> int:
        if not self._training:
            return int(torch.argmax(scores))
        distribution = torch.distributions.Categorical(logits=scores)
        choice = distribution.sample()
        self.log_probs.append(distribution.log_prob(choice))
        return int(choice)

    @staticmethod
    def _feasible_pairs(unassigned, vehicles):
        pairs = []
        for order in unassigned:
            fitting = [v for v in vehicles if v["capacity"] - v["load"] >= order["demand"]]
            room = fitting if fitting else vehicles
            for vehicle in room:
                pairs.append((order, vehicle, _pair_features(order, vehicle)))
        return pairs


def _pair_features(order: dict[str, Any], vehicle: dict[str, Any]) -> torch.Tensor:
    capacity = max(vehicle["capacity"], 1)
    return torch.tensor([
        order["coord"][0] / _WORLD,
        order["coord"][1] / _WORLD,
        order["demand"] / capacity,
        1.0 if order["priority"] == Priority.RUSH.value else 0.0,
        (capacity - vehicle["load"]) / capacity,
        vehicle["centroid"][0] / _WORLD,
        vehicle["centroid"][1] / _WORLD,
        1.0 if vehicle["load"] == 0 else 0.0,
    ], dtype=torch.float32)
