from __future__ import annotations

from dataclasses import dataclass

from .enums import Difficulty, DisruptionKind
from .models import DispatchState, Order


@dataclass(frozen=True)
class Disruption:
    kind: DisruptionKind
    vehicle_id: str | None = None
    order: Order | None = None
    order_id: str | None = None


@dataclass
class Scenario:
    seed: int
    difficulty: Difficulty
    state: DispatchState
    disruptions: list[Disruption]
