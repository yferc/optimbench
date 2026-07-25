from __future__ import annotations

from typing import Any, Protocol

from ..domain import ActionType


class Agent(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any]) -> tuple[ActionType, dict[str, Any]]: ...
