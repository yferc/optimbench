"""Benchmark every available agent across difficulties and log a task-score table.

Always runs the greedy heuristic. Adds the learned RL agent if torch is installed
and models/assignment_policy.pt exists, and an LLM agent if OPTIMBENCH_LLM_BASE_URL
is set (see scripts/run_llm.py). Example:

    python scripts/benchmark.py --seeds 50
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path

from optimbench.agents import AgentType, GreedyDispatcher, openai_compatible_agent
from optimbench.domain import BENCHMARK_VERSION, Difficulty
from optimbench.evaluation import TEST_SEEDS, Evaluator

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("optimbench")


def _agents() -> dict[str, object]:
    agents: dict[str, object] = {AgentType.GREEDY.value: GreedyDispatcher}
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()

    evaluator = Evaluator()
    seeds = list(TEST_SEEDS)[: args.seeds]
    scores = {
        name: {d: evaluator.evaluate(factory, d, seeds).task.mean for d in Difficulty}
        for name, factory in _agents().items()
    }

    log.info("benchmark v%s · task score on %d held-out test seeds (higher is better)",
             BENCHMARK_VERSION, len(seeds))
    header = f"| {'agent':<14} | " + " | ".join(d.value for d in Difficulty) + " |"
    log.info(header)
    log.info("|%s|", "-" * (len(header) - 2))
    for name, by_difficulty in scores.items():
        row = " | ".join(f"{by_difficulty[d]:.3f}" for d in Difficulty)
        log.info("| %-14s | %s |", name, row)


if __name__ == "__main__":
    main()
