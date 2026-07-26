"""Adversarial audits of the reward: the integrity gate and flags must actually fire, and
no shortcut may collect credit. A shaped optimization reward invites sim-quirk exploitation,
so the reward is audited directly, not assumed correct.
"""
from __future__ import annotations

from collections.abc import Callable

from optimbench.agents import GreedyDispatcher
from optimbench.domain import ActionType, Arg, Difficulty, Field, IntegrityFlag, is_feasible
from optimbench.evaluation import combined_reward, integrity_score, task_score
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier, VerificationResult

GEN = DispatchScenarioGenerator()
VERIFIER = DispatchVerifier()

Policy = Callable[[DispatchEnvironment], tuple[ActionType, dict]]


def _run(policy: Policy, seed: int = 0, difficulty: Difficulty = Difficulty.EASY,
         max_turns: int = 40) -> tuple[VerificationResult, list[bool]]:
    env = DispatchEnvironment(max_turns_per_wave=max_turns)
    env.reset(GEN.generate(seed, difficulty))
    committed: list[bool] = []
    while not env.done:
        action, args = policy(env)
        feasible = is_feasible(env.state) if action is ActionType.DISPATCH else False
        if env.step(action, args)[Field.ACCEPTED] and action is ActionType.DISPATCH:
            committed.append(feasible)
    waves = len(env.scenario.disruptions) + 1
    resolved = sum(committed[:waves])
    wave_feasibility = (committed + [False] * waves)[:waves]
    result = VERIFIER.verify(env.state, env.trajectory, waves, resolved)
    return result, wave_feasibility


def _greedy_policy() -> Policy:
    agent = GreedyDispatcher()
    agent.reset()
    return lambda env: agent.act(env.observation())


def _never_commit(env: DispatchEnvironment) -> tuple[ActionType, dict]:
    return ActionType.CHECK_FEASIBILITY, {}


def _spam_invalid(env: DispatchEnvironment) -> tuple[ActionType, dict]:
    return ActionType.ASSIGN_ORDER, {Arg.ORDER_ID: "nope", Arg.VEHICLE_ID: "nope"}


def _dispatch_immediately(env: DispatchEnvironment) -> tuple[ActionType, dict]:
    return ActionType.DISPATCH, {}


def test_never_committing_is_flagged_and_scores_zero():
    result, wave_feasibility = _run(_never_commit)
    assert IntegrityFlag.NEVER_COMMITTED in result.integrity_flags
    assert integrity_score(result) == 0.0
    assert combined_reward(result, wave_feasibility) == 0.0


def test_invalid_action_spam_is_flagged():
    result, _ = _run(_spam_invalid)
    assert IntegrityFlag.INVALID_ACTION_SPAM in result.integrity_flags
    assert integrity_score(result) == 0.0


def test_single_shot_dispatch_shortcut_scores_zero():
    # Committing without assigning or routing leaves orders uncovered: the feasibility gate
    # must zero the task score, and leaving disruptions unresolved must fail integrity.
    result, wave_feasibility = _run(_dispatch_immediately)
    assert task_score(result) == 0.0
    assert combined_reward(result, wave_feasibility) == 0.0


def test_greedy_is_honest_and_earns_bounded_positive_reward():
    result, wave_feasibility = _run(_greedy_policy())
    assert result.integrity_flags == []
    assert integrity_score(result) == 1.0
    reward = combined_reward(result, wave_feasibility)
    assert 0.0 < reward <= 1.0


def test_reward_is_bounded_across_seeds():
    for seed in range(8):
        result, wave_feasibility = _run(_greedy_policy(), seed=seed)
        assert 0.0 <= combined_reward(result, wave_feasibility) <= 1.0


def test_integrity_gate_dominates_quality():
    # Whenever integrity fails, the reward is zero regardless of any task or robustness credit.
    for policy in (_never_commit, _spam_invalid, _dispatch_immediately):
        result, wave_feasibility = _run(policy)
        if integrity_score(result) == 0.0:
            assert combined_reward(result, wave_feasibility) == 0.0
