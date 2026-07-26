import pytest

from optimbench.agents import GreedyDispatcher, RandomDispatcher
from optimbench.domain import ActionType, Difficulty, is_feasible
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment

GEN = DispatchScenarioGenerator()
AGENTS = [GreedyDispatcher, lambda: RandomDispatcher(seed=0)]


@pytest.mark.parametrize("make_agent", AGENTS)
def test_agent_emits_valid_actions_and_terminates(make_agent):
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    agent = make_agent()
    agent.reset()
    steps = 0
    while not env.done and steps < 5000:
        action, args = agent.act(env.observation())
        assert isinstance(action, ActionType)
        assert isinstance(args, dict)
        env.step(action, args)
        steps += 1
    assert env.done


def test_greedy_reaches_a_feasible_final_state():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    agent = GreedyDispatcher()
    agent.reset()
    while not env.done:
        env.step(*agent.act(env.observation()))
    assert is_feasible(env.state)
