import pytest

pytest.importorskip("torch")

from optimbench.agents.learned import AssignmentPolicy, LearnedDispatcher
from optimbench.domain import Difficulty, is_feasible
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment

GEN = DispatchScenarioGenerator()


def _run(agent, seed: int, difficulty: Difficulty) -> DispatchEnvironment:
    env = DispatchEnvironment()
    env.reset(GEN.generate(seed, difficulty))
    while not env.done:
        env.step(*agent.act(env.observation()))
    return env


def test_untrained_policy_still_produces_feasible_episodes():
    env = _run(LearnedDispatcher(AssignmentPolicy()), 0, Difficulty.EASY)
    assert env.done and is_feasible(env.state)


def test_training_mode_records_differentiable_log_probs():
    agent = LearnedDispatcher(AssignmentPolicy(), training=True)
    _run(agent, 1, Difficulty.MEDIUM)
    assert agent.log_probs
    assert all(logp.requires_grad for logp in agent.log_probs)
