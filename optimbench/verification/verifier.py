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
_MAX_REJECTED_ABSOLUTE = 15


class DispatchVerifier:
    def verify(
        self, state: DispatchState, trajectory: Trajectory, expected_commits: int = 1
    ) -> VerificationResult:
        found = violations(state)
        flags = self._integrity_flags(trajectory, expected_commits)
        return VerificationResult(
            feasible=not found,
            violations=[v.kind for v in found],
            objective=fleet_cost(state),
            reference=reference_cost(state),
            integrity_ok=not flags,
            integrity_flags=flags,
        )

    @staticmethod
    def _integrity_flags(trajectory: Trajectory, expected_commits: int) -> list[IntegrityFlag]:
        flags: list[IntegrityFlag] = []
        if trajectory.commits == 0:
            flags.append(IntegrityFlag.NEVER_COMMITTED)
        elif trajectory.commits < expected_commits:
            flags.append(IntegrityFlag.DISRUPTIONS_UNRESOLVED)
        if _is_action_spam(trajectory):
            flags.append(IntegrityFlag.INVALID_ACTION_SPAM)
        return flags


def _is_action_spam(trajectory: Trajectory) -> bool:
    rejected = trajectory.rejected
    if rejected > _MAX_REJECTED_ABSOLUTE:
        return True
    total = len(trajectory.decisions)
    return bool(total) and rejected / total > _MAX_REJECTED_FRACTION
