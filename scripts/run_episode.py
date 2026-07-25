"""Run the greedy baseline on one scenario and export a GIF + MP4 of the episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio

from optimbench.agents import GreedyDispatcher
from optimbench.domain import Difficulty
from optimbench.generation import DispatchScenarioGenerator
from optimbench.rendering import EpisodeRenderer
from optimbench.simulation import DispatchEnvironment

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="medium", choices=[d.value for d in Difficulty])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", default="docs/media/demo")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    scenario = DispatchScenarioGenerator().generate(args.seed, Difficulty(args.difficulty))
    env = DispatchEnvironment()
    agent = GreedyDispatcher()
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

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(f"{out}.gif", [f[::2, ::2] for f in frames], fps=args.fps, loop=0)
    imageio.mimsave(f"{out}.mp4", frames, fps=args.fps, quality=8)
    print(f"{out}.gif / .mp4  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
