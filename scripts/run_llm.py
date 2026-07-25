"""Evaluate an LLM agent through the tool API, printing per-seed progress.

Point it at any OpenAI-compatible endpoint via environment variables:

    # Groq (free tier)
    export OPTIMBENCH_LLM_BASE_URL=https://api.groq.com/openai/v1
    export OPTIMBENCH_LLM_API_KEY=gsk_...
    export OPTIMBENCH_LLM_MODEL=llama-3.3-70b-versatile

    # Ollama (local, no key)
    export OPTIMBENCH_LLM_BASE_URL=http://localhost:11434/v1
    export OPTIMBENCH_LLM_MODEL=qwen2.5:7b

    python scripts/run_llm.py --difficulty easy --seeds 5

Each turn is one API call, so start small; --max-turns bounds a flailing episode.
"""

from __future__ import annotations

import argparse

from optimbench.agents import openai_compatible_agent
from optimbench.domain import ActionType, Difficulty, is_feasible
from optimbench.evaluation import task_score
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier

GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()


def run_episode(agent, scenario, max_turns: int):
    env = DispatchEnvironment(max_turns_per_wave=max_turns)
    env.reset(scenario)
    agent.reset()
    committed: list[bool] = []
    while not env.done:
        action, args = agent.act(env.observation())
        feasible = is_feasible(env.state) if action is ActionType.DISPATCH else False
        if env.step(action, args)["accepted"] and action is ActionType.DISPATCH:
            committed.append(feasible)
    waves = len(scenario.disruptions) + 1
    return VERIFIER.verify(env.state, env.trajectory, waves, sum(committed[:waves]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=[d.value for d in Difficulty])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=60)
    args = parser.parse_args()

    difficulty = Difficulty(args.difficulty)
    tasks, feasible = [], []
    for seed in range(args.seeds):
        result = run_episode(openai_compatible_agent(), GEN.generate(seed, difficulty), args.max_turns)
        tasks.append(task_score(result))
        feasible.append(result.feasible)
        print(f"  seed {seed}: feasible={result.feasible}  task={task_score(result):.3f}", flush=True)

    print(f"{difficulty.value}: feasibility {sum(feasible) / len(feasible):.0%}  "
          f"task {sum(tasks) / len(tasks):.3f}  ({len(tasks)} seeds)")


if __name__ == "__main__":
    main()
