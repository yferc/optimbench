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
    def reset(self) -> None: ...

    def act(self, observation: dict[Field, Any]) -> tuple[ActionType, dict[Arg, Any]]: ...
