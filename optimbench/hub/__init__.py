"""Prime Intellect Hub adapter: expose OptimBench as a verifiers Environment.

This wraps the standalone OptimBench package behind the verifiers load_environment()
contract so it installs and runs from the Prime Intellect hub (prime eval run, prime-rl)
with no glue, without changing any core OptimBench code. verifiers is an optional
dependency (the hub extra), and the core package never imports this module, so a plain
install stays verifiers-free.

The mapping: each scenario seed is one dataset row; the deterministic verifier becomes the
single weighted reward function (combined_reward, with integrity as a hard gate); the three
individual scores are registered as zero-weight observability metrics; the DispatchEnvironment
runs inside env_response, one per rollout, kept in the rollout state (never a module global).
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from enum import Enum
from typing import Any

import verifiers as vf
from datasets import Dataset

from optimbench.agents.llm import SYSTEM_PROMPT, parse_tool_call, render_state, tool_action_names
from optimbench.domain import BENCHMARK_VERSION, ActionType, Difficulty, Field, is_feasible
from optimbench.evaluation import (
    TEST_SEEDS,
    TRAIN_SEEDS,
    combined_reward,
    integrity_score,
    robustness_score,
    task_score,
    verify_episode,
)
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier

_GEN = DispatchScenarioGenerator()
_VERIFIER = DispatchVerifier()
_NAMES = tool_action_names()

# A loose upper bound on model turns per episode: the per-wave cap times more waves than any
# difficulty generates, so a well-behaved agent is never cut off mid-episode.
_WAVE_HEADROOM = 6


class InfoKey(str, Enum):
    """Keys of the per-row info payload the adapter round-trips through JSON."""

    SEED = "seed"
    DIFFICULTY = "difficulty"
    BENCHMARK_VERSION = "benchmark_version"


# Rollout-state keys the adapter owns. Kept out of the model-visible message stream.
_ENV = "optimbench_env"
_COMMITTED = "optimbench_committed"
_RESULT = "optimbench_result"


def _text(message: Any) -> str:
    content = message.content if hasattr(message, "content") else message["content"]
    return content if isinstance(content, str) else ""


def _dataset(seeds: Iterable[int], difficulty: Difficulty) -> Dataset:
    rows = []
    for seed in seeds:
        env = DispatchEnvironment()
        first_observation = env.reset(_GEN.generate(seed, difficulty))
        rows.append({
            "question": render_state(first_observation),
            "info": json.dumps({
                InfoKey.SEED: int(seed),
                InfoKey.DIFFICULTY: difficulty.value,
                InfoKey.BENCHMARK_VERSION: BENCHMARK_VERSION,
            }),
        })
    return Dataset.from_list(rows)


def _new_env(info: dict[str, Any], max_turns_per_wave: int) -> DispatchEnvironment:
    env = DispatchEnvironment(max_turns_per_wave=max_turns_per_wave)
    env.reset(_GEN.generate(info[InfoKey.SEED], Difficulty(info[InfoKey.DIFFICULTY])))
    return env


def _verified(state: Any) -> tuple[Any, list[bool]]:
    if _RESULT not in state:
        state[_RESULT] = verify_episode(_VERIFIER, state[_ENV], state[_COMMITTED])
    return state[_RESULT]


async def dispatch_reward(state: Any) -> float:
    result, wave_feasibility = _verified(state)
    return combined_reward(result, wave_feasibility)


async def metric_task(state: Any) -> float:
    result, _ = _verified(state)
    return task_score(result)


async def metric_robustness(state: Any) -> float:
    _, wave_feasibility = _verified(state)
    return robustness_score(wave_feasibility)


async def metric_integrity(state: Any) -> float:
    result, _ = _verified(state)
    return integrity_score(result)


class OptimBenchEnv(vf.MultiTurnEnv):
    """MultiTurnEnv over the OptimBench dispatch tool API.

    One DispatchEnvironment per rollout lives in the state. Each model turn is parsed as a
    single JSON tool call, applied to the environment (off the event loop), and answered with
    the rendered new observation. The rollout ends when the environment terminates or truncates.
    """

    def __init__(self, max_turns_per_wave: int = 80, **kwargs: Any) -> None:
        self._max_turns_per_wave = max_turns_per_wave
        super().__init__(**kwargs)

    async def setup_state(self, state: Any) -> None:
        state[_ENV] = _new_env(state["info"], self._max_turns_per_wave)
        state[_COMMITTED] = []
        await super().setup_state(state)

    async def env_response(self, messages: Any, state: Any, **kwargs: Any) -> Any:
        env: DispatchEnvironment = state[_ENV]
        action, args = parse_tool_call(_text(messages[-1]), _NAMES)
        committing = action is ActionType.DISPATCH
        feasible = is_feasible(env.state) if committing else False
        outcome = await asyncio.to_thread(env.step, action, args)
        if committing and outcome[Field.ACCEPTED]:
            state[_COMMITTED].append(feasible)
        reply = [{"role": "user", "content": render_state(outcome[Field.OBSERVATION])}]
        if env.done:
            state["final_env_response"] = reply
        return reply


def load_environment(
    difficulty: str = "medium",
    max_turns_per_wave: int = 80,
    train_size: int = 128,
    **kwargs: Any,
) -> vf.Environment:
    """Build the verifiers Environment for the Prime Intellect hub.

    difficulty selects the scenario tier (easy, medium, hard). Training rows are the first
    train_size TRAIN_SEEDS; evaluation rows are the held-out TEST_SEEDS. The reward is
    combined_reward (integrity gate times weighted task and robustness); the three scores are
    also logged as zero-weight metrics.
    """
    tier = Difficulty(difficulty)
    rubric = vf.Rubric(funcs=[dispatch_reward], weights=[1.0])
    rubric.add_metric(metric_task)
    rubric.add_metric(metric_robustness)
    rubric.add_metric(metric_integrity)
    return OptimBenchEnv(
        max_turns_per_wave=max_turns_per_wave,
        dataset=_dataset(list(TRAIN_SEEDS)[:train_size], tier),
        eval_dataset=_dataset(list(TEST_SEEDS), tier),
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns_per_wave * _WAVE_HEADROOM,
    )
