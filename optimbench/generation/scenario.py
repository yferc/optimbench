from __future__ import annotations

from dataclasses import dataclass

from ..domain import Difficulty, DispatchState, DisruptionKind, Order, ReferenceKind


@dataclass(frozen=True)
class DifficultySpec:
    nodes: int
    vehicles: int
    orders: int
    capacity: int
    slack_ratio: float
    waves: int
    cancelled_distractors: int
    out_of_service: int
    stale_fraction: float


DIFFICULTY: dict[Difficulty, DifficultySpec] = {
    Difficulty.EASY: DifficultySpec(12, 3, 8, 12, 1.40, 1, 1, 0, 0.10),
    Difficulty.MEDIUM: DifficultySpec(20, 4, 14, 12, 1.25, 2, 2, 1, 0.15),
    Difficulty.HARD: DifficultySpec(30, 5, 22, 12, 1.12, 3, 3, 1, 0.20),
}


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
    reference_time: float
    reference_kind: ReferenceKind
