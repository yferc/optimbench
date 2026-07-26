from __future__ import annotations

from dataclasses import dataclass

from optimbench.domain.enums import Difficulty, DisruptionType
from optimbench.domain.models import DispatchState, Order


@dataclass(frozen=True)
class Disruption:
    type: DisruptionType
    vehicle_id: str | None = None
    order: Order | None = None
    order_id: str | None = None


@dataclass
class Scenario:
    seed: int
    difficulty: Difficulty
    state: DispatchState
    disruptions: list[Disruption]
