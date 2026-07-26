from __future__ import annotations

from dataclasses import dataclass, field

from optimbench.domain import IntegrityFlag, ViolationType


@dataclass(frozen=True)
class VerificationResult:
    feasible: bool
    violations: list[ViolationType] = field(default_factory=list)
    objective: float | None = None
    reference: float | None = None
    integrity_ok: bool = True
    integrity_flags: list[IntegrityFlag] = field(default_factory=list)
