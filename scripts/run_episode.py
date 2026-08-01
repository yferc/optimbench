"""Roll out one policy on one scenario and export a GIF + MP4 of the episode.

    python scripts/run_episode.py --agent random   --difficulty easy
    python scripts/run_episode.py --agent greedy    --difficulty medium
    python scripts/run_episode.py --agent learned                       # needs ".[rl]" + a trained model
    python scripts/run_episode.py --agent llm                           # needs OPTIMBENCH_LLM_* env vars

Frames land in <out-dir>/<agent>.gif (default docs/media/).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio.v2 as imageio

from optimbench.agents import AgentType, GreedyDispatcher, RandomDispatcher, openai_compatible_agent
from optimbench.domain import Difficulty, DisruptionType, Field, Note, Scenario
from optimbench.generation import DispatchScenarioGenerator
from optimbench.rendering import EpisodeRenderer, FrameContext
from optimbench.simulation import DispatchEnvironment

_BREAKDOWN = (255, 92, 140)
_DISPATCHED = (75, 212, 138)
_HOLD_EVENT = 5  # extra frames on a commit or disruption so the viewer can read what happened

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("optimbench")


def _learned_agent():
    # torch is the optional [rl] extra; import it lazily so random/greedy/llm need no torch.
    import torch

    from optimbench.agents.learned import AssignmentPolicy, LearnedDispatcher
    policy = AssignmentPolicy()
    policy.load_state_dict(torch.load(ROOT / "models" / "assignment_policy.pt"))
    policy.eval()
    return LearnedDispatcher(policy)


def make_agent(agent_type: AgentType, seed: int):
    builders = {
        AgentType.RANDOM: lambda: RandomDispatcher(seed),
        AgentType.GREEDY: GreedyDispatcher,
        AgentType.LEARNED: _learned_agent,
        AgentType.LLM: openai_compatible_agent,
    }
    return builders[agent_type]()


def _wave_advanced(outcome: dict) -> bool:
    result = outcome[Field.RESULT]
    return Field.WAVE_ADVANCED in result and result[Field.WAVE_ADVANCED]


_DISRUPTION_BANNER = {
    DisruptionType.BREAKDOWN: "BREAKDOWN: BUSIEST TRUCK DOWN",
    DisruptionType.RUSH_ORDER: "RUSH ORDER ARRIVED",
    DisruptionType.CANCELLATION: "ORDER CANCELLED",
}


def _event_banner(scenario: Scenario, outcome: dict, wave_index: int) -> tuple[str, tuple[int, int, int]]:
    """The headline for this frame: which disruption just landed, or the final commit."""
    if outcome[Field.NOTE] is Note.DISRUPTION_APPLIED:
        return _DISRUPTION_BANNER[scenario.disruptions[wave_index].type], _BREAKDOWN
    if outcome[Field.NOTE] is Note.FINAL_COMMIT:
        return "FINAL WAVE DISPATCHED", _DISPATCHED
    return "", _BREAKDOWN


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=[k.value for k in AgentType], default=AgentType.GREEDY.value)
    parser.add_argument("--difficulty", choices=[d.value for d in Difficulty], default=Difficulty.MEDIUM.value)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out-dir", default="docs/media")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    agent_type = AgentType(args.agent)
    scenario = DispatchScenarioGenerator().generate(args.seed, Difficulty(args.difficulty))
    env = DispatchEnvironment()
    agent = make_agent(agent_type, args.seed)
    renderer = EpisodeRenderer()
    waves = len(scenario.disruptions)

    env.reset(scenario)
    agent.reset()
    frames = [renderer.frame(env.state, waves)]
    wave_index = 0
    while not env.done:
        action, tool_args = agent.act(env.observation())
        outcome = env.step(action, tool_args)
        banner, color = _event_banner(scenario, outcome, wave_index)
        if _wave_advanced(outcome):
            wave_index += 1
        context = FrameContext(action=action, args=tool_args, accepted=outcome[Field.ACCEPTED],
                               note=outcome[Field.NOTE], banner=banner, banner_color=color)
        frame = renderer.frame(env.state, waves, context)
        frames.append(frame)
        if banner:
            frames += [frame] * _HOLD_EVENT
    frames += [frames[-1]] * args.fps
    renderer.close()

    out = ROOT / args.out_dir / agent_type.value
    out.parent.mkdir(parents=True, exist_ok=True)
    # the GIF keeps full resolution: halving it turned every label into unreadable mush, and the
    # action bar is the point of the animation
    imageio.mimsave(f"{out}.gif", frames, fps=args.fps, loop=0)
    imageio.mimsave(f"{out}.mp4", frames, fps=args.fps, quality=8)
    log.info("%s: %s.gif / .mp4 (%d frames)", agent_type.value, out, len(frames))


if __name__ == "__main__":
    main()
