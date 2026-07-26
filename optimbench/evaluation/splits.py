"""Canonical, disjoint seed splits for the benchmark.

Training draws from TRAIN_SEEDS, model selection uses VAL_SEEDS, and every reported
number uses TEST_SEEDS. Keeping the three ranges disjoint stops checkpoint selection
from peeking at the reported test set, which would inflate every leaderboard number.
"""
from __future__ import annotations

TRAIN_SEEDS = range(10_000, 1_000_000)
VAL_SEEDS = range(1_000, 1_050)
TEST_SEEDS = range(50)
