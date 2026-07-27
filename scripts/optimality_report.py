"""Report how tight the heuristic reference is against a true optimum (OR-Tools CVRP).

The task score's denominator is the heuristic reference solve (sweep + nearest-neighbour + 2-opt),
which keeps scoring deterministic and dependency-light. This offline report quantifies what that
choice costs: for a sample of seeds per difficulty it compares the heuristic reference cost to the
OR-Tools optimum and reports the gap, so a reader knows exactly how close to optimal the "1.0
ceiling" really is. Needs the solver extra (ortools).

    python scripts/optimality_report.py --seeds 20 --out papers/optimality_gap.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from optimbench.analysis import optimal_cost
from optimbench.domain import Difficulty, reference_cost
from optimbench.generation import DispatchScenarioGenerator

GEN = DispatchScenarioGenerator()
log = logging.getLogger("optimbench")


def _row(difficulty: Difficulty, seeds: int, time_limit: int) -> tuple[float, float, float]:
    heuristics, optima, gaps = [], [], []
    for seed in range(seeds):
        state = GEN.generate(seed, difficulty).state
        heuristic = reference_cost(state)
        optimum = optimal_cost(state, time_limit_s=time_limit)
        heuristics.append(heuristic)
        optima.append(optimum)
        gaps.append(heuristic / optimum - 1.0 if optimum else 0.0)
    n = len(gaps)
    return sum(heuristics) / n, sum(optima) / n, sum(gaps) / n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--time-limit", type=int, default=2)
    parser.add_argument("--out", default=None, help="write a markdown report to this path")
    args = parser.parse_args()

    rows = {d: _row(d, args.seeds, args.time_limit) for d in Difficulty}
    for difficulty, (heuristic, optimum, gap) in rows.items():
        log.info("%-6s heuristic %.1f  optimal %.1f  reference is %.1f%% above optimal",
                 difficulty.value, heuristic, optimum, 100 * gap)

    if args.out is not None:
        lines = [
            "# Reference optimality gap",
            "",
            (f"How far the heuristic reference (the task-score denominator) sits above a true "
             f"OR-Tools optimum, over {args.seeds} seeds per difficulty. Smaller is tighter. The "
             "reference stays heuristic on purpose, so scoring needs no solver and is deterministic; "
             "this table is the honest disclosure of what that costs."),
            "",
            "| difficulty | mean heuristic | mean optimal | reference above optimal |",
            "|------------|----------------|--------------|-------------------------|",
        ]
        for difficulty, (heuristic, optimum, gap) in rows.items():
            lines.append(f"| {difficulty.value} | {heuristic:.1f} | {optimum:.1f} | {100 * gap:.1f}% |")
        lines.append("")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
        log.info("wrote %s", out)


if __name__ == "__main__":
    main()
