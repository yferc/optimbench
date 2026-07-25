from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import ActionType


@dataclass(frozen=True)
class Decision:
    turn: int
    action: ActionType
    args: dict[str, Any]
    accepted: bool
    note: str = ""


@dataclass
class Trajectory:
    decisions: list[Decision] = field(default_factory=list)

    def record(self, decision: Decision) -> None:
        self.decisions.append(decision)

    @property
    def rejected(self) -> int:
        return sum(1 for d in self.decisions if not d.accepted)

    @property
    def commits(self) -> int:
        return sum(1 for d in self.decisions if d.action is ActionType.DISPATCH and d.accepted)
