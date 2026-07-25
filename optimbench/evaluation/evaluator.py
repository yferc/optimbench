from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..agents import Agent
from ..domain import Difficulty, is_feasible
from ..generation import DispatchScenarioGenerator
from ..simulation import DispatchEnvironment
from ..verification import DispatchVerifier
from .metrics import (
    MetricSummary,
    integrity_score,
    robustness_score,
    task_score,
)


@dataclass(frozen=True)
class EvaluationReport:
    episodes: int
    difficulty: Difficulty
    feasibility_rate: float
    task: MetricSummary
    robustness: MetricSummary
    integrity: MetricSummary

    def format(self) -> str:
        def line(name: str, m: MetricSummary) -> str:
            return f"  {name:<11} mean {m.mean:.3f}  iqm {m.iqm:.3f}  95% CI [{m.ci[0]:.3f}, {m.ci[1]:.3f}]"

        return (
            f"{self.difficulty.value} · {self.episodes} episodes\n"
            f"  feasibility {self.feasibility_rate:.2%}\n"
            f"{line('task', self.task)}\n"
            f"{line('robustness', self.robustness)}\n"
            f"{line('integrity', self.integrity)}"
        )


class Evaluator:
    def __init__(
        self,
        generator: DispatchScenarioGenerator | None = None,
        verifier: DispatchVerifier | None = None,
    ) -> None:
        self._generator = generator or DispatchScenarioGenerator()
        self._verifier = verifier or DispatchVerifier()

    def evaluate(
        self, make_agent: Callable[[], Agent], difficulty: Difficulty, seeds: Iterable[int]
    ) -> EvaluationReport:
        tasks: list[float] = []
        robustness: list[float] = []
        integrity: list[float] = []
        feasible: list[float] = []

        for seed in seeds:
            result, commit_feasibility = self._run_episode(make_agent(), seed, difficulty)
            tasks.append(task_score(result))
            robustness.append(robustness_score(commit_feasibility))
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

    def _run_episode(self, agent: Agent, seed: int, difficulty: Difficulty):
        scenario = self._generator.generate(seed, difficulty)
        env = DispatchEnvironment()
        env.reset(scenario)
        agent.reset()
        left_feasible: dict[int, bool] = {}
        while not env.done:
            left_feasible[env.state.wave] = is_feasible(env.state)
            action, args = agent.act(env.observation())
            env.step(action, args)
        left_feasible[env.state.wave] = is_feasible(env.state)

        waves = len(scenario.disruptions) + 1
        wave_feasibility = [left_feasible.get(wave, False) for wave in range(waves)]
        resolved = sum(wave_feasibility)
        result = self._verifier.verify(env.state, env.trajectory, waves, resolved)
        return result, wave_feasibility
