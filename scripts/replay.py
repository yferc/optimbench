"""Replay one episode as a readable narrative: what the agent did, when a disruption hit, and
why it did or did not hold up. A score of 0.61 says little; a step-through of the decisions,
the disruption, and the constraint it violated says a lot. This is the qualitative companion
to the leaderboard, and the source of the failure traces worth putting in a write-up.

    python scripts/replay.py --agent random --difficulty easy --seed 0

Works with any built-in agent (random, greedy, learned) or an LLM through the tool API when the
OPTIMBENCH_LLM_* env vars are set.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from optimbench.agents import AgentType, GreedyDispatcher, RandomDispatcher, openai_compatible_agent
from optimbench.domain import ActionType, Arg, Difficulty, DisruptionType, Field, Note, is_feasible
from optimbench.evaluation import combined_reward, verify_episode
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier

GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()
log = logging.getLogger("optimbench")

# Actions worth narrating: the mutations and the commit. Reads are collapsed into a count.
_NARRATED = {
    ActionType.ASSIGN_ORDER, ActionType.UNASSIGN_ORDER, ActionType.SET_ROUTE,
    ActionType.REROUTE, ActionType.DISPATCH, ActionType.REFUSE, ActionType.INVALID,
}
_DISRUPTION_TEXT = {
    DisruptionType.BREAKDOWN: "the busiest vehicle breaks down",
    DisruptionType.RUSH_ORDER: "an urgent order arrives",
    DisruptionType.CANCELLATION: "an order is cancelled",
}


def _agent(agent_type: AgentType):
    if agent_type is AgentType.GREEDY:
        return GreedyDispatcher()
    if agent_type is AgentType.RANDOM:
        return RandomDispatcher()
    if agent_type is AgentType.LLM:
        return openai_compatible_agent()
    import torch  # lazy: torch is the optional rl dependency

    from optimbench.agents.learned import AssignmentPolicy, LearnedDispatcher
    policy = AssignmentPolicy()
    policy.load_state_dict(torch.load(Path(__file__).resolve().parent.parent / "models" / "assignment_policy.pt"))
    policy.eval()
    return LearnedDispatcher(policy)


def _arg_summary(action: ActionType, args: dict[Arg, object]) -> str:
    return ", ".join(f"{arg.value}={args[arg]}" for arg in args) if action in _NARRATED else ""


def replay(agent_type: AgentType, difficulty, seed: int) -> None:
    scenario = GEN.generate(seed, difficulty)
    disruptions = " then ".join(_DISRUPTION_TEXT[d.type] for d in scenario.disruptions) or "none"
    log.info("scenario seed=%d difficulty=%s", seed, difficulty.value)
    log.info("  %d vehicles, %d orders, %d disruption waves: %s",
             len(scenario.state.vehicles), len(scenario.state.orders),
             len(scenario.disruptions), disruptions)

    env = DispatchEnvironment()
    env.reset(scenario)
    agent = _agent(agent_type)
    agent.reset()
    committed_feasibility: list[bool] = []
    reads = 0
    wave = 1
    log.info("--- wave %d ---", wave)
    while not env.done:
        action, args = agent.act(env.observation())
        plan_feasible = is_feasible(env.state) if action is ActionType.DISPATCH else False
        outcome = env.step(action, args)
        if action not in _NARRATED:
            reads += 1
            continue
        if reads:
            log.info("  (%d read actions)", reads)
            reads = 0
        mark = "ok" if outcome[Field.ACCEPTED] else "REJECTED"
        note = "" if outcome[Field.NOTE] is Note.NONE else f"  [{outcome[Field.NOTE].value}]"
        log.info("  %s(%s) -> %s%s", action.value, _arg_summary(action, args), mark, note)
        if action is ActionType.DISPATCH and outcome[Field.ACCEPTED]:
            committed_feasibility.append(plan_feasible)
            log.info("  committed wave %d: plan was %s",
                     wave, "feasible" if plan_feasible else "INFEASIBLE")
            if outcome[Field.RESULT][Field.WAVE_ADVANCED]:
                wave += 1
                log.info("--- wave %d (%s) ---", wave, disruptions)

    result, wave_feasibility = verify_episode(VERIFIER, env, committed_feasibility)
    log.info("verdict: feasible=%s  task=%.3f  robustness=%.2f  integrity=%s  reward=%.3f",
             result.feasible, min(1.0, result.reference / result.objective) if result.objective else 0.0,
             sum(wave_feasibility) / len(wave_feasibility) if wave_feasibility else 0.0,
             "ok" if result.integrity_ok else [f.value for f in result.integrity_flags],
             combined_reward(result, wave_feasibility))
    if result.violations:
        log.info("  violated: %s", ", ".join(sorted({v.value for v in result.violations})))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=[a.value for a in AgentType], default=AgentType.RANDOM.value)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    replay(AgentType(args.agent), Difficulty(args.difficulty), args.seed)


if __name__ == "__main__":
    main()
