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
_USER_AGENT = "optimbench"
_ARG_VALUES = frozenset(arg.value for arg in Arg)


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MessageKey(str, Enum):
    """The OpenAI chat wire-format keys of a single message."""

    ROLE = "role"
    CONTENT = "content"


# The example call is derived from the enums, not hardcoded, so the schema the LLM is
# told to emit cannot drift from the keys the parser reads.
_EXAMPLE_CALL = json.dumps({
    ToolCallKey.ACTION.value: ActionType.ASSIGN_ORDER.value,
    ToolCallKey.ARGS.value: {Arg.ORDER_ID.value: "ord_3", Arg.VEHICLE_ID.value: "veh_1"},
})

SYSTEM_PROMPT = (
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
        # Send an explicit User-Agent: the default urllib one is blocked by the Cloudflare
        # front on providers like Groq (403, error 1010), which would fail the whole run.
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
        }
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


def tool_action_names(tools: tuple[ToolSpec, ...] = TOOLSET) -> dict[str, ActionType]:
    return {tool.action.value: tool.action for tool in tools}


def render_state(observation: dict[Field, Any], tools: tuple[ToolSpec, ...] = TOOLSET) -> str:
    """Render the tool list and the current observation into the one user message a model sees."""
    lines = "\n".join(
        f"  {tool.action.value}({', '.join(a.value for a in tool.args)}): {tool.summary}"
        for tool in tools
    )
    return f"TOOLS:\n{lines}\n\nSTATE:\n{json.dumps(observation, separators=(',', ':'))}"


_REQUIRED_ARGS: dict[ActionType, tuple[Arg, ...]] = {tool.action: tool.args for tool in TOOLSET}


def parse_tool_call(reply: str, action_by_name: dict[str, ActionType]) -> tuple[ActionType, dict[Arg, Any]]:
    """Parse one JSON tool call out of an untrusted model reply, re-keying args to the Arg enum.

    This is the sanitizer boundary for untrusted model output. Anything that is not a complete,
    known tool call (no JSON, unparseable JSON, unknown action, or a required argument missing or
    misshapen) maps to ActionType.INVALID, which the environment rejects and records as a rejected
    decision. It is never crashed on downstream, nor silently rewritten to a benign accepted read.
    """
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match is None:
        return ActionType.INVALID, {}
    try:
        call = json.loads(match.group())
    except json.JSONDecodeError:
        return ActionType.INVALID, {}
    if not isinstance(call, dict) or ToolCallKey.ACTION not in call:
        return ActionType.INVALID, {}
    action_name = call[ToolCallKey.ACTION]
    # isinstance before the membership test: a non-scalar action value would otherwise be
    # hashed here and crash the run on an untrusted reply
    if not isinstance(action_name, str) or action_name not in action_by_name:
        return ActionType.INVALID, {}
    action = action_by_name[action_name]
    args = _to_args(call[ToolCallKey.ARGS] if ToolCallKey.ARGS in call else {})
    if not _complete_call(action, args):
        return ActionType.INVALID, {}
    return action, args


def _complete_call(action: ActionType, args: dict[Arg, Any]) -> bool:
    # Every required arg must be present and well-typed. Ids and the filter are scalars that the
    # environment later hashes against its dicts, so a list or object here must be rejected now
    # rather than crash the handler. STOPS is a list; its elements are range-checked downstream.
    for required in _REQUIRED_ARGS[action]:
        if required not in args:
            return False
        value = args[required]
        if required is Arg.STOPS:
            if not isinstance(value, list):
                return False
        elif isinstance(value, bool) or not isinstance(value, (str, int)):
            return False
    return True


class LLMAgent:
    def __init__(self, client: ChatClient, tools: tuple[ToolSpec, ...] = TOOLSET) -> None:
        self._client = client
        self._tools = tools
        self._names = tool_action_names(tools)
        self.reset()

    def reset(self) -> None:
        self._history: list[dict[str, str]] = []

    def act(self, observation: dict[Field, Any]) -> tuple[ActionType, dict[Arg, Any]]:
        user = chat_message(Role.USER, render_state(observation, self._tools))
        reply = self._client.chat([chat_message(Role.SYSTEM, SYSTEM_PROMPT), *self._history, user])
        self._remember(user, reply)
        return parse_tool_call(reply, self._names)

    def _remember(self, user: dict[str, str], reply: str) -> None:
        self._history = [*self._history, user, chat_message(Role.ASSISTANT, reply)]
        self._history = self._history[-2 * _MEMORY_TURNS :]


def chat_message(role: Role, content: str) -> dict[str, str]:
    return {MessageKey.ROLE.value: role.value, MessageKey.CONTENT.value: content}


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
