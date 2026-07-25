"""Benchmark every available agent across difficulties and print a task-score table.

Always runs the greedy heuristic. Adds the learned RL agent if torch is installed
and models/assignment_policy.pt exists, and an LLM agent if OPTIMBENCH_LLM_BASE_URL
is set (see scripts/run_llm.py). Example:

    python scripts/benchmark.py --seeds 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from optimbench.agents import GreedyDispatcher, openai_compatible_agent
from optimbench.domain import Difficulty
from optimbench.evaluation import Evaluator

ROOT = Path(__file__).resolve().parent.parent


def _agents(seeds: int) -> dict[str, object]:
    agents: dict[str, object] = {"greedy": GreedyDispatcher}
    model = ROOT / "models/assignment_policy.pt"
    if model.exists():
        try:
            import torch

            from optimbench.agents.learned import AssignmentPolicy, LearnedDispatcher

            policy = AssignmentPolicy()
            policy.load_state_dict(torch.load(model))
            policy.eval()
            agents["learned-rl"] = lambda: LearnedDispatcher(policy)
        except ImportError:
            pass
    if os.environ.get("OPTIMBENCH_LLM_BASE_URL"):
        agents[os.environ.get("OPTIMBENCH_LLM_MODEL", "llm")] = openai_compatible_agent
    return agents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()

    evaluator = Evaluator()
    agents = _agents(args.seeds)
    scores = {
        name: {d: evaluator.evaluate(factory, d, range(args.seeds)).task.mean for d in Difficulty}
        for name, factory in agents.items()
    }

    header = f"| {'agent':<12} | " + " | ".join(d.value for d in Difficulty) + " |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for name, by_difficulty in scores.items():
        row = " | ".join(f"{by_difficulty[d]:.3f}" for d in Difficulty)
        print(f"| {name:<12} | {row} |")


if __name__ == "__main__":
    main()
