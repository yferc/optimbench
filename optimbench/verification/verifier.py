from __future__ import annotations

from ..domain import IntegrityFlag, route_time, violations
from ..simulation import DispatchEnvironment
from .result import VerificationResult

_MAX_REJECTED_FRACTION = 0.35


class DispatchVerifier:
    def verify(self, env: DispatchEnvironment) -> VerificationResult:
        found = violations(env.state)
        objective = self._committed_cost(env)
        return VerificationResult(
            feasible=not found,
            violations=[v.kind for v in found],
            objective=objective,
            reference=env.scenario.reference_time,
            integrity_ok=not self._integrity_flags(env),
            integrity_flags=self._integrity_flags(env),
        )

    @staticmethod
    def _committed_cost(env: DispatchEnvironment) -> float:
        return float(sum(
            route_time(env.state.network, v.route)
            for v in env.state.vehicles.values()
            if v.in_service
        ))

    @staticmethod
    def _integrity_flags(env: DispatchEnvironment) -> list[IntegrityFlag]:
        flags: list[IntegrityFlag] = []
        trajectory = env.trajectory
        if trajectory.commits == 0:
            flags.append(IntegrityFlag.NEVER_COMMITTED)
        if trajectory.decisions:
            rejected_fraction = trajectory.rejected / len(trajectory.decisions)
            if rejected_fraction > _MAX_REJECTED_FRACTION:
                flags.append(IntegrityFlag.INVALID_ACTION_SPAM)
        return flags
