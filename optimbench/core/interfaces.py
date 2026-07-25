"""Core abstractions for OptimBench.

The wedge is *infrastructure*, not a single benchmark: a small set of clean,
extensible interfaces so a new constrained-optimization problem can be added
without rewriting the framework. The load-bearing piece is the **Verifier** —
answering "is this agent's solution actually correct, and did it get there
honestly?" — which is exactly where explicit constraints and measurable
objectives make optimization a uniquely strong domain for verifiable rewards.

    ScenarioGenerator  -> produces a seeded, feasibility-guaranteed instance
    Environment        -> the agent operates it over turns (tools, disruptions)
    Verifier           -> deterministic check: feasibility + objective + integrity
    Metrics            -> task / robustness / integrity scores from a rollout
    Agent              -> baseline or LLM policy
    Evaluator          -> runs agents over scenarios, aggregates with CIs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# --- data carriers ---------------------------------------------------------


@dataclass
class Scenario:
    """One generated instance: the initial world plus its hidden dynamics."""

    seed: int
    difficulty: str
    world: Any                      # problem-specific state (a Pydantic model)
    disruptions: list[Any] = field(default_factory=list)   # hidden, fired on commit
    reference_objective: float | None = None               # solver/heuristic reference
    reference_kind: str = "unknown"                         # "optimal" | "heuristic"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """A single agent decision, for trajectory logging (why it failed, not just score)."""

    turn: int
    action: str
    args: dict[str, Any]
    rationale: str | None = None
    observation_after: dict[str, Any] | None = None


@dataclass
class VerificationResult:
    """The Verifier's verdict on a committed solution. Deterministic, no LLM judge."""

    feasible: bool                          # hard-constraint gate
    violations: list[str] = field(default_factory=list)
    objective: float | None = None          # measured cost of the committed solution
    reference: float | None = None          # reference cost for the ratio
    integrity_ok: bool = True               # did the agent avoid gaming the verifier?
    integrity_flags: list[str] = field(default_factory=list)  # e.g. invalid-action spam


# --- interfaces ------------------------------------------------------------


class ScenarioGenerator(ABC):
    """Procedurally generates instances that are feasible *by construction*."""

    @abstractmethod
    def generate(self, seed: int, difficulty: str) -> Scenario:
        """Return a Scenario whose initial state — and every post-disruption
        state — admits a feasible solution (via engineered slack)."""


class Environment(ABC):
    """A stateful, multi-turn world the agent operates through tools."""

    @abstractmethod
    def reset(self, scenario: Scenario) -> dict[str, Any]:
        """Load a scenario; return the first observation."""

    @abstractmethod
    def tools(self) -> dict[str, Any]:
        """The callable tool surface exposed to the agent this turn."""

    @abstractmethod
    def step(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        """Apply one action; return the new observation. Logs a Step."""

    @abstractmethod
    def commit(self) -> dict[str, Any]:
        """Commit the current solution; fire the next disruption wave if any."""

    @property
    @abstractmethod
    def trajectory(self) -> list[Step]:
        """The full decision log for this rollout."""

    @property
    @abstractmethod
    def done(self) -> bool:
        ...


class Verifier(ABC):
    """The crown jewel: deterministically decide if a committed solution is
    correct (feasibility + objective) and whether it was reached honestly."""

    @abstractmethod
    def verify(self, scenario: Scenario, env: "Environment") -> VerificationResult:
        ...


class Agent(ABC):
    """A policy: baseline heuristic, OR solver, or an LLM agent."""

    @abstractmethod
    def act(self, observation: dict[str, Any], tools: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Choose the next (action, args)."""


class Metrics:
    """Turn a VerificationResult + trajectory into the three headline scores."""

    @staticmethod
    def task(result: VerificationResult) -> float:
        """Optimization quality behind the feasibility gate (0 if infeasible)."""
        if not result.feasible or not result.objective or not result.reference:
            return 0.0
        return float(min(1.0, result.reference / result.objective))

    @staticmethod
    def robustness(results: list[VerificationResult]) -> float:
        """Fraction of post-disruption states left feasible across the rollout."""
        if not results:
            return 0.0
        return sum(r.feasible for r in results) / len(results)

    @staticmethod
    def integrity(result: VerificationResult) -> float:
        """1.0 if the agent solved it without gaming the verifier, else 0.0."""
        return 1.0 if result.integrity_ok else 0.0
