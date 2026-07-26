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
import logging

from optimbench.agents import openai_compatible_agent
from optimbench.domain import ActionType, Difficulty, Field, is_feasible
from optimbench.evaluation import task_score
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier

GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()
log = logging.getLogger("optimbench")


def run_episode(agent, scenario, max_turns: int):
    env = DispatchEnvironment(max_turns_per_wave=max_turns)
    env.reset(scenario)
    agent.reset()
    committed: list[bool] = []
    while not env.done:
        action, args = agent.act(env.observation())
        feasible = is_feasible(env.state) if action is ActionType.DISPATCH else False
        if env.step(action, args)[Field.ACCEPTED] and action is ActionType.DISPATCH:
            committed.append(feasible)
    waves = len(scenario.disruptions) + 1
    return VERIFIER.verify(env.state, env.trajectory, waves, sum(committed[:waves]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=[d.value for d in Difficulty], default=Difficulty.EASY.value)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=60)
    args = parser.parse_args()

    difficulty = Difficulty(args.difficulty)
    tasks, feasible = [], []
    for seed in range(args.seeds):
        result = run_episode(openai_compatible_agent(), GEN.generate(seed, difficulty), args.max_turns)
        tasks.append(task_score(result))
        feasible.append(result.feasible)
        log.info("  seed %d: feasible=%s  task=%.3f", seed, result.feasible, task_score(result))

    log.info("%s: feasibility %.0f%%  task %.3f  (%d seeds)", difficulty.value,
             100 * sum(feasible) / len(feasible), sum(tasks) / len(tasks), len(tasks))


if __name__ == "__main__":
    main()
