"""Export expert dispatch trajectories as JSONL for supervised warm-start (SFT).

A tool-using baseline (greedy by default) is rolled across a seed range, and every turn of
each successful episode is written as a (prompt, completion) pair in the same format the LLM
agent uses: the rendered observation is the prompt, the expert's tool call is the completion.
Only episodes that finish feasible with clean integrity are kept, so the data is high quality.

    python scripts/export_trajectories.py --agent greedy --difficulty easy --episodes 200

The result plugs into an SFT run to warm-start an LLM before RL on the same environment.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from optimbench.agents import AgentType, GreedyDispatcher, RandomDispatcher
from optimbench.agents.llm import SYSTEM_PROMPT, render_state
from optimbench.domain import ActionType, Arg, Difficulty, Field, ToolCallKey, is_feasible
from optimbench.evaluation import TRAIN_SEEDS, verify_episode
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier

ROOT = Path(__file__).resolve().parent.parent
GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()
log = logging.getLogger("optimbench")

_EXPERTS = {AgentType.GREEDY.value: GreedyDispatcher, AgentType.RANDOM.value: RandomDispatcher}


def _tool_call(action: ActionType, args: dict[Arg, object]) -> str:
    return json.dumps({
        ToolCallKey.ACTION.value: action.value,
        ToolCallKey.ARGS.value: {arg.value: value for arg, value in args.items()},
    })


def _episode(agent, scenario) -> tuple[list[dict[str, str]], bool]:
    env = DispatchEnvironment()
    env.reset(scenario)
    agent.reset()
    pairs: list[dict[str, str]] = []
    committed: list[bool] = []
    while not env.done:
        observation = env.observation()
        action, args = agent.act(observation)
        pairs.append({"prompt": render_state(observation), "completion": _tool_call(action, args)})
        feasible = is_feasible(env.state) if action is ActionType.DISPATCH else False
        if env.step(action, args)[Field.ACCEPTED] and action is ActionType.DISPATCH:
            committed.append(feasible)
    result, _ = verify_episode(VERIFIER, env, committed)
    return pairs, result.feasible and result.integrity_ok


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=list(_EXPERTS), default=AgentType.GREEDY.value)
    parser.add_argument("--difficulty", choices=[d.value for d in Difficulty], default=Difficulty.EASY.value)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", default="data/expert_trajectories.jsonl")
    args = parser.parse_args()

    difficulty = Difficulty(args.difficulty)
    expert = _EXPERTS[args.agent]
    seeds = list(TRAIN_SEEDS)[: args.episodes]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    kept_episodes, pairs_written = 0, 0
    with out.open("w") as sink:
        for seed in seeds:
            pairs, keep = _episode(expert(), GEN.generate(seed, difficulty))
            if not keep:
                continue
            kept_episodes += 1
            for pair in pairs:
                sink.write(json.dumps({**pair, "system": SYSTEM_PROMPT}) + "\n")
                pairs_written += 1

    log.info("%s on %s: kept %d/%d episodes, %d (prompt, completion) pairs -> %s",
             args.agent, difficulty.value, kept_episodes, len(seeds), pairs_written, out)


if __name__ == "__main__":
    main()
