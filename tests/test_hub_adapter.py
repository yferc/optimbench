"""The verifiers hub adapter, exercised through the real verifiers rollout hooks. Skipped
cleanly when verifiers is not installed (it is the optional hub extra).
"""
from __future__ import annotations

import asyncio
import json

import pytest

vf = pytest.importorskip("verifiers")

from optimbench.agents import GreedyDispatcher
from optimbench.domain import ToolCallKey
from optimbench.hub import OptimBenchEnv, dispatch_reward, load_environment


def _tool_call_json(action, args) -> str:
    return json.dumps({
        ToolCallKey.ACTION.value: action.value,
        ToolCallKey.ARGS.value: {arg.value: value for arg, value in args.items()},
    })


def test_load_environment_builds_a_verifiers_environment():
    env = load_environment(difficulty="easy", train_size=4)
    assert isinstance(env, vf.Environment)
    assert isinstance(env, OptimBenchEnv)
    assert len(env.get_eval_dataset()) == 50  # TEST_SEEDS


def test_greedy_rollout_through_the_hooks_scores_bounded_positive():
    env = load_environment(difficulty="easy", train_size=4)

    async def drive() -> float:
        state = {"info": {"seed": 0, "difficulty": "easy", "benchmark_version": "1.0"}}
        await env.setup_state(state)
        agent = GreedyDispatcher()
        agent.reset()
        turns = 0
        while state.get("final_env_response") is None and turns < 500:
            action, args = agent.act(state["optimbench_env"].observation())
            message = {"role": "assistant", "content": _tool_call_json(action, args)}
            await env.env_response([message], state)
            turns += 1
        assert state["optimbench_env"].done
        return await dispatch_reward(state)

    reward = asyncio.run(drive())
    assert 0.0 < reward <= 1.0
