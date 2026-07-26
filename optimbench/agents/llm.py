from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from enum import Enum
from typing import Any, Protocol
from urllib.error import HTTPError, URLError

from optimbench.domain import TOOLSET, ActionType, Arg, Field, ToolCallKey, ToolSpec

_RETRIES = 5
_BACKOFF = 2.0
_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_NO_KEY = "none"  # local servers (Ollama) accept any placeholder
_ARG_VALUES = frozenset(arg.value for arg in Arg)


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# The example call is derived from the enums, not hardcoded, so the schema the LLM is
# told to emit cannot drift from the keys the parser reads.
_EXAMPLE_CALL = json.dumps({
    ToolCallKey.ACTION.value: ActionType.ASSIGN_ORDER.value,
    ToolCallKey.ARGS.value: {Arg.ORDER_ID.value: "ord_3", Arg.VEHICLE_ID.value: "veh_1"},
})

_SYSTEM = (
    """You are a vehicle-dispatch controller working one tool call per turn.

Each wave, follow this procedure:
  1. Assign every unassigned order to an in-service vehicle that still has spare
     capacity (prefer the vehicle whose centroid is nearest the order to keep travel low).
  2. When no orders are unassigned, REROUTE each vehicle that has orders. This builds
     its depot-anchored route. A vehicle with orders but an empty route is infeasible.
  3. Once every loaded vehicle has been rerouted, DISPATCH to submit the plan.
Repeat for each disruption wave, including the final one (you must dispatch it too).

The latest STATE is authoritative; your recent actions are shown for context.
Reply with ONLY a JSON object naming one tool call, e.g.
""" + _EXAMPLE_CALL + ". No prose."
)

_MEMORY_TURNS = 6


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatibleClient:
    """Works with any OpenAI-compatible /chat/completions endpoint:
    Groq, Gemini, Ollama, OpenRouter, x.ai, OpenAI."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._temperature = temperature

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self._model, "temperature": self._temperature, "messages": messages,
        }).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        for attempt in range(_RETRIES):
            try:
                request = urllib.request.Request(self._url, data=payload, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.loads(response.read())["choices"][0]["message"]["content"]
            except HTTPError as error:
                if error.code not in _RETRYABLE or attempt == _RETRIES - 1:
                    raise
            except URLError:
                if attempt == _RETRIES - 1:
                    raise
            time.sleep(_BACKOFF * 2**attempt)
        raise RuntimeError("unreachable")


class LLMAgent:
    def __init__(self, client: ChatClient, tools: tuple[ToolSpec, ...] = TOOLSET) -> None:
        self._client = client
        self._tools = tools
        self._names = {tool.action.value: tool.action for tool in tools}
        self.reset()

    def reset(self) -> None:
        self._history: list[dict[str, str]] = []

    def act(self, observation: dict[Field, Any]) -> tuple[ActionType, dict[Arg, Any]]:
        user = _message(Role.USER, self._render(observation))
        reply = self._client.chat([_message(Role.SYSTEM, _SYSTEM), *self._history, user])
        self._remember(user, reply)
        return self._parse(reply)

    def _remember(self, user: dict[str, str], reply: str) -> None:
        self._history = [*self._history, user, _message(Role.ASSISTANT, reply)]
        self._history = self._history[-2 * _MEMORY_TURNS :]

    def _render(self, observation: dict[Field, Any]) -> str:
        lines = "\n".join(
            f"  {tool.action.value}({', '.join(a.value for a in tool.args)}): {tool.summary}"
            for tool in self._tools
        )
        return f"TOOLS:\n{lines}\n\nSTATE:\n{json.dumps(observation, separators=(',', ':'))}"

    def _parse(self, reply: str) -> tuple[ActionType, dict[Arg, Any]]:
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        if match is None:
            return ActionType.CHECK_FEASIBILITY, {}
        try:
            call = json.loads(match.group())
        except json.JSONDecodeError:
            return ActionType.CHECK_FEASIBILITY, {}
        if ToolCallKey.ACTION not in call or call[ToolCallKey.ACTION] not in self._names:
            return ActionType.CHECK_FEASIBILITY, {}
        action = self._names[call[ToolCallKey.ACTION]]
        raw = call[ToolCallKey.ARGS] if ToolCallKey.ARGS in call else {}
        return action, _to_args(raw)


def _message(role: Role, content: str) -> dict[str, str]:
    # "role"/"content" are the OpenAI chat wire-format keys.
    return {"role": role.value, "content": content}


def _to_args(raw: Any) -> dict[Arg, Any]:
    # The reply is untrusted, so keep only keys that name a real tool argument and
    # re-key them to the Arg enum the environment and every other agent use.
    if not isinstance(raw, dict):
        return {}
    return {Arg(key): value for key, value in raw.items() if key in _ARG_VALUES}


def openai_compatible_agent() -> LLMAgent:
    """Build an LLMAgent from OPTIMBENCH_LLM_{BASE_URL,API_KEY,MODEL} env vars."""
    env = os.environ
    client = OpenAICompatibleClient(
        base_url=env["OPTIMBENCH_LLM_BASE_URL"],
        api_key=env["OPTIMBENCH_LLM_API_KEY"] if "OPTIMBENCH_LLM_API_KEY" in env else _NO_KEY,
        model=env["OPTIMBENCH_LLM_MODEL"],
    )
    return LLMAgent(client)
