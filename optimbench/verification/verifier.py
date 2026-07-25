from __future__ import annotations

from ..domain import DispatchState, IntegrityFlag, Trajectory, fleet_cost, violations
from .result import VerificationResult

_MAX_REJECTED_FRACTION = 0.35


class DispatchVerifier:
    def verify(
        self, state: DispatchState, trajectory: Trajectory, reference_time: float
    ) -> VerificationResult:
        found = violations(state)
        flags = self._integrity_flags(trajectory)
        return VerificationResult(
            feasible=not found,
            violations=[v.kind for v in found],
            objective=fleet_cost(state),
            reference=reference_time,
            integrity_ok=not flags,
            integrity_flags=flags,
        )

    @staticmethod
    def _integrity_flags(trajectory: Trajectory) -> list[IntegrityFlag]:
        flags: list[IntegrityFlag] = []
        if trajectory.commits == 0:
            flags.append(IntegrityFlag.NEVER_COMMITTED)
        if trajectory.decisions:
            if trajectory.rejected / len(trajectory.decisions) > _MAX_REJECTED_FRACTION:
                flags.append(IntegrityFlag.INVALID_ACTION_SPAM)
        return flags
