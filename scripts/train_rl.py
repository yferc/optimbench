"""Train the assignment policy with REINFORCE against the greedy baseline.

The learned agent keeps greedy's route-then-dispatch skeleton but replaces its
best-fit assignment with a policy scored on order/vehicle geometry, so training
targets exactly the gap greedy leaves against the sweep reference.

    pip install -e ".[rl]"
    python scripts/train_rl.py --episodes 800 --difficulty easy
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from optimbench.agents import GreedyDispatcher
from optimbench.agents.learned import AssignmentPolicy, LearnedDispatcher
from optimbench.domain import ActionType, Difficulty, is_feasible
from optimbench.evaluation import task_score
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier

ROOT = Path(__file__).resolve().parent.parent
GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()


def rollout(scenario, agent) -> float:
    env = DispatchEnvironment()
    env.reset(scenario)
    agent.reset()
    committed: list[bool] = []
    while not env.done:
        action, args = agent.act(env.observation())
        feasible = is_feasible(env.state) if action is ActionType.DISPATCH else False
        if env.step(action, args)["accepted"] and action is ActionType.DISPATCH:
            committed.append(feasible)
    waves = len(scenario.disruptions) + 1
    result = VERIFIER.verify(env.state, env.trajectory, waves, sum(committed[:waves]))
    return task_score(result)


def _mean_task(policy: AssignmentPolicy, difficulty: Difficulty, seeds: range) -> float:
    with torch.no_grad():
        scores = [rollout(GEN.generate(s, difficulty), LearnedDispatcher(policy)) for s in seeds]
    return sum(scores) / len(scores)


def evaluate(policy: AssignmentPolicy, difficulties: list[Difficulty]) -> dict[Difficulty, float]:
    return {d: _mean_task(policy, d, range(50)) for d in difficulties}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--difficulty", default="all", choices=["all", *(d.value for d in Difficulty)])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/assignment_policy.pt")
    args = parser.parse_args()

    train_diffs = list(Difficulty) if args.difficulty == "all" else [Difficulty(args.difficulty)]
    rng = random.Random(0)
    policy = AssignmentPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0

    for episode in range(1, args.episodes + 1):
        difficulty = rng.choice(train_diffs)
        seed = rng.randrange(10_000, 1_000_000)
        agent = LearnedDispatcher(policy, training=True)
        reward = rollout(GEN.generate(seed, difficulty), agent)
        baseline = rollout(GEN.generate(seed, difficulty), GreedyDispatcher())
        if agent.log_probs:
            loss = -(reward - baseline) * torch.stack(agent.log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 200 == 0:
            scores = evaluate(policy, train_diffs)
            mean = sum(scores.values()) / len(scores)
            report = "  ".join(f"{d.value} {s:.3f}" for d, s in scores.items())
            marker = ""
            if mean > best:
                best, marker = mean, "  *saved"
                torch.save(policy.state_dict(), out)
            print(f"ep {episode:4d}  learned: {report}  (mean {mean:.3f}){marker}")

    print(f"best mean {best:.3f} saved to {out}")


if __name__ == "__main__":
    main()
