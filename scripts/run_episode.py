"""Roll out one policy on one scenario and export a GIF + MP4 of the episode.

    python scripts/run_episode.py --agent random   --difficulty easy
    python scripts/run_episode.py --agent greedy    --difficulty medium
    python scripts/run_episode.py --agent learned                       # needs ".[rl]" + a trained model
    python scripts/run_episode.py --agent llm                           # needs OPTIMBENCH_LLM_* env vars

Frames land in docs/media/<agent>.gif by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio

from optimbench.domain import Difficulty
from optimbench.generation import DispatchScenarioGenerator
from optimbench.rendering import EpisodeRenderer
from optimbench.simulation import DispatchEnvironment

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ("random", "greedy", "learned", "llm")


def make_agent(name: str, seed: int):
    if name == "random":
        from optimbench.agents import RandomDispatcher
        return RandomDispatcher(seed)
    if name == "greedy":
        from optimbench.agents import GreedyDispatcher
        return GreedyDispatcher()
    if name == "learned":
        import torch

        from optimbench.agents.learned import AssignmentPolicy, LearnedDispatcher
        policy = AssignmentPolicy()
        policy.load_state_dict(torch.load(ROOT / "models/assignment_policy.pt"))
        policy.eval()
        return LearnedDispatcher(policy)
    from optimbench.agents import openai_compatible_agent
    return openai_compatible_agent()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="greedy", choices=AGENTS)
    parser.add_argument("--difficulty", default="medium", choices=[d.value for d in Difficulty])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    scenario = DispatchScenarioGenerator().generate(args.seed, Difficulty(args.difficulty))
    env = DispatchEnvironment()
    agent = make_agent(args.agent, args.seed)
    renderer = EpisodeRenderer()
    waves = len(scenario.disruptions)

    env.reset(scenario)
    agent.reset()
    frames = [renderer.frame(env.state, waves)]
    while not env.done:
        action, tool_args = agent.act(env.observation())
        env.step(action, tool_args)
        frames.append(renderer.frame(env.state, waves))
    frames += [frames[-1]] * args.fps
    renderer.close()

    out = ROOT / (args.out or f"docs/media/{args.agent}")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(f"{out}.gif", [f[::2, ::2] for f in frames], fps=args.fps, loop=0)
    imageio.mimsave(f"{out}.mp4", frames, fps=args.fps, quality=8)
    print(f"{args.agent}: {out}.gif / .mp4  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
