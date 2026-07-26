"""Scoring and aggregation: the three per-episode scores, plus the IQM and bootstrap
confidence interval used to summarise a score across seeds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimbench.verification import VerificationResult

_TASK_WEIGHT = 0.7
_ROBUSTNESS_WEIGHT = 0.3


def task_score(result: VerificationResult) -> float:
    if not result.feasible or result.objective == 0.0:
        return 0.0
    return float(min(1.0, result.reference / result.objective))


def robustness_score(feasible_at_each_commit: list[bool]) -> float:
    if not feasible_at_each_commit:
        return 0.0
    return float(np.mean(feasible_at_each_commit))


def integrity_score(result: VerificationResult) -> float:
    return 1.0 if result.integrity_ok else 0.0


def combined_reward(result: VerificationResult, wave_feasibility: list[bool]) -> float:
    """Fuse the three scores into one scalar RL reward in [0, 1].

    Integrity is a hard multiplicative gate, not a weighted summand: an episode that never
    committed, left a disruption unresolved, or spammed invalid actions scores zero no matter
    how good its dispatch looked, so there is no dishonest way to collect partial credit.
    Behind that gate, dispatch quality and disruption robustness combine as a fixed weighted
    sum. This is the signal to train against; the three scores stay separate for reporting.
    """
    quality = _TASK_WEIGHT * task_score(result) + _ROBUSTNESS_WEIGHT * robustness_score(wave_feasibility)
    return integrity_score(result) * quality


def iqm(values: list[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size < 4:
        return float(ordered.mean()) if ordered.size else 0.0
    trim = ordered.size // 4
    return float(ordered[trim : ordered.size - trim].mean())


def bootstrap_ci(values: list[float], samples: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(0)  # fixed seed so the reported confidence interval is reproducible
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
