"""Prime Intellect hub entry point for OptimBench dynamic vehicle dispatch.

The implementation lives in the installed optimbench package (optimbench.hub). This module is
the thin, hub-conventional entry point that exposes load_environment.
"""
from optimbench.hub import load_environment

__all__ = ["load_environment"]
