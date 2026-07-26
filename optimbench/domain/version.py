"""The benchmark version.

Bump it whenever a change could alter reported scores: the feasibility gate, the
disruption model, the reference solver, or the score formulas. Numbers are only
comparable within the same version, so the bump makes old and new results
non-comparable by construction rather than by footnote.
"""
from __future__ import annotations

BENCHMARK_VERSION = "1.0"
