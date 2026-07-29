"""Score the programmatic agents on the hub reward, so LLM runs are comparable to baselines.

`prime eval run` reports this environment's single scalar reward (integrity times weighted task
and robustness) on TEST_SEEDS at one difficulty, while scripts/benchmark.py reports task score
per difficulty. Those are different scales, so comparing a hub LLM number to the leaderboard is
an error. This puts greedy, learned and random on exactly the hub's metric and seeds. Example:

    python scripts/hub_baseline.py --seeds 3 --difficulty medium
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path

from optimbench.agents import Agent, AgentType, GreedyDispatcher, RandomDispatcher
from optimbench.domain import BENCHMARK_VERSION, ActionType, Difficulty, Field, is_feasible
from optimbench.evaluation import TEST_SEEDS, combined_reward, verify_episode
from optimbench.evaluation.metrics import integrity_score, robustness_score, task_score
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier

AgentFactory = Callable[[], Agent]

ROOT = Path(__file__).resolve().parent.parent
GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()
log = logging.getLogger("optimbench")


def _agents() -> dict[str, AgentFactory]:
    agents: dict[str, AgentFactory] = {
        AgentType.RANDOM.value: RandomDispatcher,
        AgentType.GREEDY.value: GreedyDispatcher,
    }
    model_path = ROOT / "models" / "assignment_policy.pt"
    if model_path.exists() and importlib.util.find_spec("torch") is not None:
        import torch

        from optimbench.agents.learned import AssignmentPolicy, LearnedDispatcher
        policy = AssignmentPolicy()
        policy.load_state_dict(torch.load(model_path))
        policy.eval()
        agents[AgentType.LEARNED.value] = lambda: LearnedDispatcher(policy)
    return agents


def rollout(factory: AgentFactory, seed: int, difficulty: Difficulty, max_turns: int) -> dict[str, float]:
    """Run one episode and score it exactly as the hub adapter does."""
    env = DispatchEnvironment(max_turns_per_wave=max_turns)
    env.reset(GEN.generate(seed, difficulty))
    agent = factory()
    agent.reset()
    committed: list[bool] = []
    while not env.done:
        action, args = agent.act(env.observation())
        # feasibility is read before the step, at the moment of the commit, like the adapter
        feasible = is_feasible(env.state) if action is ActionType.DISPATCH else False
        if env.step(action, args)[Field.ACCEPTED] and action is ActionType.DISPATCH:
            committed.append(feasible)
    result, wave_feasibility = verify_episode(VERIFIER, env, committed)
    return {
        "reward": combined_reward(result, wave_feasibility),
        "task": task_score(result),
        "robustness": robustness_score(wave_feasibility),
        "integrity": integrity_score(result),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="how many TEST_SEEDS, matching the eval's -n")
    parser.add_argument("--difficulty", choices=[d.value for d in Difficulty], default=Difficulty.MEDIUM.value)
    parser.add_argument("--max-turns", type=int, default=80)
    args = parser.parse_args()

    difficulty = Difficulty(args.difficulty)
    seeds = list(TEST_SEEDS)[: args.seeds]
    log.info("benchmark v%s  difficulty=%s  seeds=%s  metric=combined_reward (the hub reward)",
             BENCHMARK_VERSION, difficulty.value, seeds)
    log.info("%-10s %8s %8s %8s %8s   per-seed rewards", "agent", "reward", "task", "robust", "integ")
    for name, factory in _agents().items():
        runs = [rollout(factory, seed, difficulty, args.max_turns) for seed in seeds]
        mean = {key: sum(run[key] for run in runs) / len(runs) for key in runs[0]}
        log.info("%-10s %8.3f %8.3f %8.3f %8.3f   %s", name, mean["reward"], mean["task"],
                 mean["robustness"], mean["integrity"],
                 ", ".join(f"{run['reward']:.3f}" for run in runs))


if __name__ == "__main__":
    main()
