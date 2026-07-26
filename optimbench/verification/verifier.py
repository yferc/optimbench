from __future__ import annotations

from optimbench.domain import (
    DispatchState,
    IntegrityFlag,
    Trajectory,
    fleet_cost,
    reference_cost,
    violations,
)
from optimbench.verification.result import VerificationResult

_SPAM_RATE = 0.5
_SPAM_MIN_DECISIONS = 25


class DispatchVerifier:
    def verify(
        self,
        state: DispatchState,
        trajectory: Trajectory,
        expected_waves: int,
        resolved_waves: int,
    ) -> VerificationResult:
        found = violations(state)
        flags = self._integrity_flags(trajectory, expected_waves, resolved_waves)
        return VerificationResult(
            feasible=not found,
            violations=[v.type for v in found],
            objective=fleet_cost(state),
            reference=reference_cost(state),
            integrity_ok=not flags,
            integrity_flags=flags,
        )

    @staticmethod
    def _integrity_flags(
        trajectory: Trajectory, expected_waves: int, resolved_waves: int
    ) -> list[IntegrityFlag]:
        flags: list[IntegrityFlag] = []
        if trajectory.commits == 0:
            flags.append(IntegrityFlag.NEVER_COMMITTED)
        elif resolved_waves < expected_waves:
            flags.append(IntegrityFlag.DISRUPTIONS_UNRESOLVED)
        if _is_action_spam(trajectory):
            flags.append(IntegrityFlag.INVALID_ACTION_SPAM)
        return flags


def _is_action_spam(trajectory: Trajectory) -> bool:
    total = len(trajectory.decisions)
    return total >= _SPAM_MIN_DECISIONS and trajectory.rejected / total > _SPAM_RATE
