from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Protocol

from ..domain import ActionType
from ..simulation import TOOLSET, ToolSpec

_SYSTEM = """You are a vehicle-dispatch controller. Each turn you see the live orders,
the fleet, and which disruption wave you are on. Assign every live order to an
in-service vehicle, sequence each vehicle's route from the depot (node 0) and back,
then DISPATCH to submit the plan and advance. You MUST dispatch every wave, including
the final one. Keep total travel time low.

Reply with ONLY a JSON object naming one tool call, e.g. {"action": "assign_order",
"args": {"order_id": "ord_3", "vehicle_id": "veh_1"}}. No prose."""


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleClient:
    """Works with any OpenAI-compatible /chat/completions endpoint —
    Groq, Gemini, Ollama, OpenRouter, x.ai, OpenAI."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        request = urllib.request.Request(
            self._url, data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read())
        return body["choices"][0]["message"]["content"]


class LLMAgent:
    def __init__(self, client: ChatClient, tools: tuple[ToolSpec, ...] = TOOLSET) -> None:
        self._client = client
        self._tools = tools
        self._names = {tool.action.value: tool.action for tool in tools}

    def reset(self) -> None:
        pass

    def act(self, observation: dict[str, Any]) -> tuple[ActionType, dict[str, Any]]:
        reply = self._client.complete(_SYSTEM, self._render(observation))
        return self._parse(reply)

    def _render(self, observation: dict[str, Any]) -> str:
        return "\n".join([
            "TOOLS:",
            *(f"  {t.action.value}({', '.join(t.args)}): {t.summary}" for t in self._tools),
            "",
            "STATE:",
            json.dumps(observation, separators=(",", ":")),
        ])

    def _parse(self, reply: str) -> tuple[ActionType, dict[str, Any]]:
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        if match is None:
            return ActionType.CHECK_FEASIBILITY, {}
        try:
            call = json.loads(match.group())
        except json.JSONDecodeError:
            return ActionType.CHECK_FEASIBILITY, {}
        action = self._names.get(call.get("action"))
        if action is None:
            return ActionType.CHECK_FEASIBILITY, {}
        args = call.get("args")
        return action, args if isinstance(args, dict) else {}


def openai_compatible_agent() -> LLMAgent:
    """Build an LLMAgent from OPTIMBENCH_LLM_{BASE_URL,API_KEY,MODEL} env vars."""
    client = OpenAICompatibleClient(
        base_url=os.environ["OPTIMBENCH_LLM_BASE_URL"],
        api_key=os.environ.get("OPTIMBENCH_LLM_API_KEY", "none"),
        model=os.environ["OPTIMBENCH_LLM_MODEL"],
    )
    return LLMAgent(client)
