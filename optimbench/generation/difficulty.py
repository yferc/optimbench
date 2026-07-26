"""Difficulty presets: the knobs (fleet size, capacity, order count, time-window slack,
disruption count) that define the easy, medium, and hard scenario distributions.
"""
from __future__ import annotations

from dataclasses import dataclass

from optimbench.domain import Difficulty


@dataclass(frozen=True)
class DifficultySpec:
    nodes: int
    vehicles: int
    orders: int
    vehicle_capacity: int
    slack_ratio: float          # fleet capacity over total demand; higher is roomier, so easier
    disruption_waves: int
    cancelled_distractors: int
    offline_vehicles: int
    stale_fraction: float       # share of the observed travel-time matrix that is inaccurate


DIFFICULTY: dict[Difficulty, DifficultySpec] = {
    Difficulty.EASY: DifficultySpec(12, 3, 8, 12, 1.40, 1, 1, 0, 0.10),
    Difficulty.MEDIUM: DifficultySpec(20, 4, 14, 12, 1.25, 2, 2, 1, 0.15),
    Difficulty.HARD: DifficultySpec(30, 5, 22, 12, 1.12, 3, 3, 1, 0.20),
}
