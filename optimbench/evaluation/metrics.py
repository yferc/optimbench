from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..verification import VerificationResult


def task_score(result: VerificationResult) -> float:
    if not result.feasible or not result.objective or not result.reference:
        return 0.0
    return float(min(1.0, result.reference / result.objective))


def robustness_score(feasible_at_each_commit: list[bool]) -> float:
    if not feasible_at_each_commit:
        return 0.0
    return float(np.mean(feasible_at_each_commit))


def integrity_score(result: VerificationResult) -> float:
    return 1.0 if result.integrity_ok else 0.0


def iqm(values: list[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size < 4:
        return float(ordered.mean()) if ordered.size else 0.0
    cut = ordered.size // 4
    return float(ordered[cut : ordered.size - cut].mean())


def bootstrap_ci(values: list[float], samples: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(0)
    draws = data[rng.integers(0, data.size, size=(samples, data.size))].mean(axis=1)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    iqm: float
    ci: tuple[float, float]

    @classmethod
    def of(cls, values: list[float]) -> MetricSummary:
        return cls(float(np.mean(values)) if values else 0.0, iqm(values), bootstrap_ci(values))
