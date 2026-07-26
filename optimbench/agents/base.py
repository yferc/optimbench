"""The agent interface and the enum of built-in agent types."""
from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from optimbench.domain import ActionType, Arg, Field


class AgentType(str, Enum):
    RANDOM = "random"
    GREEDY = "greedy"
    LEARNED = "learned"
    LLM = "llm"


class Agent(Protocol):
    """The interface a dispatch agent implements.

    reset() clears any per-episode memory before a new scenario. act() receives the
    current observation (keyed by the Field enum) and returns the chosen action together
    with its arguments (keyed by the Arg enum).
    """

    def reset(self) -> None: ...

    def act(self, observation: dict[Field, Any]) -> tuple[ActionType, dict[Arg, Any]]: ...
