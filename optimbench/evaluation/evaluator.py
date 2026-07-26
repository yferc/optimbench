"""Batch evaluation: roll an agent across seeded scenarios and summarise the three
scores with IQM and bootstrap confidence intervals.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from optimbench.agents import Agent
from optimbench.domain import BENCHMARK_VERSION, ActionType, Difficulty, Field, is_feasible
from optimbench.evaluation.metrics import (
    MetricSummary,
    integrity_score,
    robustness_score,
    task_score,
)
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier, VerificationResult


@dataclass(frozen=True)
class EvaluationReport:
    episodes: int
    difficulty: Difficulty
    feasibility_rate: float
    task: MetricSummary
    robustness: MetricSummary
    integrity: MetricSummary
    version: str = BENCHMARK_VERSION

    def format(self) -> str:
        def line(name: str, summary: MetricSummary) -> str:
            return (f"  {name:<11} mean {summary.mean:.3f}  iqm {summary.iqm:.3f}"
                    f"  95% CI [{summary.ci[0]:.3f}, {summary.ci[1]:.3f}]")

        return (
            f"{self.difficulty.value} · {self.episodes} episodes · benchmark v{self.version}\n"
            f"  feasibility {self.feasibility_rate:.2%}\n"
            f"{line('task', self.task)}\n"
            f"{line('robustness', self.robustness)}\n"
            f"{line('integrity', self.integrity)}"
        )


class Evaluator:
    """Runs an agent over seeded scenarios and aggregates the per-episode scores."""

    def __init__(
        self,
        generator: DispatchScenarioGenerator | None = None,
        verifier: DispatchVerifier | None = None,
    ) -> None:
        self._generator = generator if generator is not None else DispatchScenarioGenerator()
        self._verifier = verifier if verifier is not None else DispatchVerifier()

    def evaluate(
        self, make_agent: Callable[[], Agent], difficulty: Difficulty, seeds: Iterable[int]
    ) -> EvaluationReport:
        """Roll make_agent() across the given seeds at one difficulty and summarise the run.

        Returns an EvaluationReport with the feasibility rate and, for each of the task,
        robustness, and integrity scores, its mean, IQM, and 95% bootstrap CI.
        """
        tasks: list[float] = []
        robustness: list[float] = []
        integrity: list[float] = []
        feasible: list[float] = []

        for seed in seeds:
            result, wave_feasibility = self._run_episode(make_agent(), seed, difficulty)
            tasks.append(task_score(result))
            robustness.append(robustness_score(wave_feasibility))
            integrity.append(integrity_score(result))
            feasible.append(1.0 if result.feasible else 0.0)

        return EvaluationReport(
            episodes=len(tasks),
            difficulty=difficulty,
            feasibility_rate=sum(feasible) / len(feasible) if feasible else 0.0,
            task=MetricSummary.of(tasks),
            robustness=MetricSummary.of(robustness),
            integrity=MetricSummary.of(integrity),
        )

    def _run_episode(
        self, agent: Agent, seed: int, difficulty: Difficulty
    ) -> tuple[VerificationResult, list[bool]]:
        scenario = self._generator.generate(seed, difficulty)
        env = DispatchEnvironment()
        env.reset(scenario)
        agent.reset()
        committed_feasibility: list[bool] = []
        while not env.done:
            action, args = agent.act(env.observation())
            feasible = is_feasible(env.state) if action is ActionType.DISPATCH else False
            if env.step(action, args)[Field.ACCEPTED] and action is ActionType.DISPATCH:
                committed_feasibility.append(feasible)

        waves = len(scenario.disruptions) + 1
        resolved = sum(committed_feasibility[:waves])
        wave_feasibility = (committed_feasibility + [False] * waves)[:waves]
        result = self._verifier.verify(env.state, env.trajectory, waves, resolved)
        return result, wave_feasibility
