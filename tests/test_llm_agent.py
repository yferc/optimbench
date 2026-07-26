import pytest

from optimbench.agents import LLMAgent
from optimbench.domain import ActionType, Difficulty, Field
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment

GEN = DispatchScenarioGenerator()


class _ScriptedClient:
    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._turn = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        reply = self._replies[min(self._turn, len(self._replies) - 1)]
        self._turn += 1
        return reply


def test_parses_json_action_from_noisy_reply():
    agent = LLMAgent(_ScriptedClient(['thinking... {"action": "list_orders", "args": {"filter": "live"}}']))
    action, args = agent.act({})
    assert action is ActionType.LIST_ORDERS and args == {"filter": "live"}


def test_unparseable_reply_maps_to_invalid():
    agent = LLMAgent(_ScriptedClient(["no json here"]))
    assert agent.act({})[0] is ActionType.INVALID


def test_unknown_action_maps_to_invalid():
    agent = LLMAgent(_ScriptedClient(['{"action": "teleport", "args": {}}']))
    assert agent.act({})[0] is ActionType.INVALID


_ADVERSARIAL = [
    "no json here",
    "not even close",
    '{"nonsense": true}',                                              # no action key
    '{"action": {"x": 1}}',                                           # non-hashable action value
    '{"action": ["assign_order"]}',                                   # list action value
    '{"action": "teleport", "args": {}}',                             # unknown action
    '{"action": "assign_order", "args": {"order_id": "ord_0"}}',      # missing vehicle_id
    '{"action": "list_orders", "args": {"filter": ["live"]}}',        # non-scalar filter
    '{"action": "get_vehicle", "args": {"vehicle_id": {"a": 1}}}',    # non-scalar id
    '{"action": "assign_order", "args": {"order_id": ["x"], "vehicle_id": "v"}}',
    '{"action": "set_route", "args": {"vehicle_id": "veh_0", "stops": 5}}',  # stops not a list
]


@pytest.mark.parametrize("reply", _ADVERSARIAL)
def test_adversarial_replies_map_to_invalid_and_never_crash_the_env(reply):
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    agent = LLMAgent(_ScriptedClient([reply]))
    action, args = agent.act(env.observation())
    assert action is ActionType.INVALID
    assert env.step(action, args)[Field.ACCEPTED] is False  # rejected, and no exception raised


def test_missing_required_arg_is_rejected_not_crashing():
    # assign_order without vehicle_id used to KeyError inside env.step and abort the whole run.
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    agent = LLMAgent(_ScriptedClient(['{"action": "assign_order", "args": {"order_id": "ord_0"}}']))
    action, args = agent.act(env.observation())
    assert action is ActionType.INVALID
    assert env.step(action, args)[Field.ACCEPTED] is False


def test_llm_agent_drives_an_episode_to_completion():
    env = DispatchEnvironment()
    env.reset(GEN.generate(0, Difficulty.EASY))
    agent = LLMAgent(_ScriptedClient(['{"action": "dispatch", "args": {}}']))
    steps = 0
    while not env.done and steps < 500:
        env.step(*agent.act(env.observation()))
        steps += 1
    assert env.done
