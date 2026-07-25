from __future__ import annotations

from ..domain import (
    DispatchState,
    IntegrityFlag,
    Trajectory,
    fleet_cost,
    reference_cost,
    violations,
)
from .result import VerificationResult

_MAX_REJECTED_FRACTION = 0.35
_MIN_DECISIONS_FOR_RATIO = 25
_REJECTED_PER_COMMIT = 8


class DispatchVerifier:
    def verify(
        self,
        state: DispatchState,
        trajectory: Trajectory,
        expected_commits: int = 1,
        resolved_commits: int | None = None,
    ) -> VerificationResult:
        found = violations(state)
        flags = self._integrity_flags(trajectory, expected_commits, resolved_commits)
        return VerificationResult(
            feasible=not found,
            violations=[v.kind for v in found],
            objective=fleet_cost(state),
            reference=reference_cost(state),
            integrity_ok=not flags,
            integrity_flags=flags,
        )

    @staticmethod
    def _integrity_flags(
        trajectory: Trajectory, expected_commits: int, resolved_commits: int | None
    ) -> list[IntegrityFlag]:
        resolved = trajectory.commits if resolved_commits is None else resolved_commits
        flags: list[IntegrityFlag] = []
        if trajectory.commits == 0:
            flags.append(IntegrityFlag.NEVER_COMMITTED)
        elif resolved < expected_commits:
            flags.append(IntegrityFlag.DISRUPTIONS_UNRESOLVED)
        if _is_action_spam(trajectory, expected_commits):
            flags.append(IntegrityFlag.INVALID_ACTION_SPAM)
        return flags


def _is_action_spam(trajectory: Trajectory, expected_commits: int) -> bool:
    rejected = trajectory.rejected
    if rejected > _REJECTED_PER_COMMIT * expected_commits:
        return True
    total = len(trajectory.decisions)
    return total >= _MIN_DECISIONS_FOR_RATIO and rejected / total > _MAX_REJECTED_FRACTION
