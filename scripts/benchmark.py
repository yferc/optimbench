"""Benchmark every available agent across difficulties and log a score table.

Always runs the random and greedy baselines. Adds the learned RL agent if torch is
installed and models/assignment_policy.pt exists, and an LLM agent if OPTIMBENCH_LLM_MODEL
is set (see scripts/run_llm.py). With --out it writes a markdown leaderboard. Example:

    python scripts/benchmark.py --seeds 50 --out papers/leaderboard.md
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from collections.abc import Callable
from pathlib import Path

from optimbench.agents import (
    Agent,
    AgentType,
    GreedyDispatcher,
    RandomDispatcher,
    openai_compatible_agent,
)
from optimbench.domain import BENCHMARK_VERSION, Difficulty
from optimbench.evaluation import TEST_SEEDS, EvaluationReport, Evaluator

AgentFactory = Callable[[], Agent]

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("optimbench")


def _agents() -> dict[str, AgentFactory]:
    agents: dict[str, AgentFactory] = {
        AgentType.RANDOM.value: RandomDispatcher,
        AgentType.GREEDY.value: GreedyDispatcher,
    }
    model = ROOT / "models" / "assignment_policy.pt"
    if model.exists() and importlib.util.find_spec("torch") is not None:
        import torch

        from optimbench.agents.learned import AssignmentPolicy, LearnedDispatcher
        policy = AssignmentPolicy()
        policy.load_state_dict(torch.load(model))
        policy.eval()
        agents[AgentType.LEARNED.value] = lambda: LearnedDispatcher(policy)
    if "OPTIMBENCH_LLM_MODEL" in os.environ:
        agents[os.environ["OPTIMBENCH_LLM_MODEL"]] = openai_compatible_agent
    return agents


def _evaluate(seeds: list[int]) -> dict[str, dict[Difficulty, EvaluationReport]]:
    evaluator = Evaluator()
    return {
        name: {d: evaluator.evaluate(factory, d, seeds) for d in Difficulty}
        for name, factory in _agents().items()
    }


def _markdown(reports: dict[str, dict[Difficulty, EvaluationReport]], seeds: int) -> str:
    diffs = list(Difficulty)
    head = "| agent | " + " | ".join(f"{d.value} task" for d in diffs) + " |"
    rule = "|" + "---|" * (len(diffs) + 1)
    lines = [
        f"# OptimBench leaderboard (benchmark v{BENCHMARK_VERSION})",
        "",
        (f"Task score (IQM) on {seeds} held-out test seeds per difficulty, higher is better. "
         "The reference (a strong deterministic solve) sits at 1.0 by definition."),
        "",
        head,
        rule,
    ]
    for name, by_difficulty in reports.items():
        cells = " | ".join(f"{by_difficulty[d].task.iqm:.3f}" for d in diffs)
        lines.append(f"| {name} | {cells} |")
    lines += ["", "Robustness and integrity (mean over the same seeds):", ""]
    lines.append("| agent | " + " | ".join(f"{d.value} robust / integ" for d in diffs) + " |")
    lines.append(rule)
    for name, by_difficulty in reports.items():
        cells = " | ".join(
            f"{by_difficulty[d].robustness.mean:.2f} / {by_difficulty[d].integrity.mean:.2f}"
            for d in diffs
        )
        lines.append(f"| {name} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--out", default=None, help="write a markdown leaderboard to this path")
    args = parser.parse_args()

    seeds = list(TEST_SEEDS)[: args.seeds]
    reports = _evaluate(seeds)

    log.info("benchmark v%s - task score (IQM) on %d held-out test seeds (higher is better)",
             BENCHMARK_VERSION, len(seeds))
    header = f"| {'agent':<14} | " + " | ".join(d.value for d in Difficulty) + " |"
    log.info(header)
    log.info("|%s|", "-" * (len(header) - 2))
    for name, by_difficulty in reports.items():
        row = " | ".join(f"{by_difficulty[d].task.iqm:.3f}" for d in Difficulty)
        log.info("| %-14s | %s |", name, row)

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_markdown(reports, len(seeds)))
        log.info("wrote %s", out)


if __name__ == "__main__":
    main()
