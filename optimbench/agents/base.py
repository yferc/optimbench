from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from optimbench.domain import ActionType


class AgentType(str, Enum):
    RANDOM = "random"
    GREEDY = "greedy"
    LEARNED = "learned"
    LLM = "llm"


class Agent(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any]) -> tuple[ActionType, dict[str, Any]]: ...
