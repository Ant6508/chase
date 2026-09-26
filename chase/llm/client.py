"""Client LLM contre un serveur LM Studio (API compatible OpenAI).

Chaque appel à `decide` est un tour isolé : aucun historique de conversation
n'est conservé entre les pas, pour garder un budget de tokens borné quelle
que soit la longueur de l'épisode (contexte chargé : 13056 tokens).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from ..moves import Move
from .config import LLMConfig

_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move",
        "description": "Choisit le déplacement du poursuivant pour ce pas.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": [m.name for m in Move],
                },
                "reasoning": {
                    "type": "string",
                    "description": "Justification brève du choix, une ou deux phrases.",
                },
            },
            "required": ["direction", "reasoning"],
        },
    },
}


@dataclass
class LLMCallResult:
    move: Move
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    retries: int
    fallback: bool
    thinking: str = ""  # chaîne de pensée émise avant l'appel d'outil (`reasoning_content`)


class LLMClient(Protocol):
    def decide(self, system_prompt: str, user_prompt: str) -> LLMCallResult: ...


def _parse_tool_call(response) -> tuple[str, str] | None:
    try:
        call = response.choices[0].message.tool_calls[0]
        args = json.loads(call.function.arguments)
        direction = args["direction"]
    except (IndexError, AttributeError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if direction not in Move.__members__:
        return None
    return direction, str(args.get("reasoning", ""))


class LMStudioClient:
    """Un appel = un tour isolé : pas d'historique de conversation entre les pas."""

    def __init__(self, cfg: LLMConfig, client=None):
        self.cfg = cfg
        self._client = client or OpenAI(base_url=cfg.base_url, api_key="lm-studio")

    def decide(self, system_prompt: str, user_prompt: str) -> LLMCallResult:
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.monotonic()
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=self.cfg.timeout_s,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tools=[_MOVE_TOOL],
                    tool_choice="required",
                )
            except Exception as exc:
                last_error = exc
                continue
            latency_ms = (time.monotonic() - t0) * 1000
            parsed = _parse_tool_call(response)
            if parsed is None:
                last_error = ValueError("réponse sans appel d'outil valide")
                continue
            direction, reasoning = parsed
            usage = response.usage
            thinking = getattr(response.choices[0].message, "reasoning_content", None) or ""
            return LLMCallResult(
                move=Move[direction],
                reasoning=reasoning,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
                retries=attempt,
                fallback=False,
                thinking=thinking,
            )
        return LLMCallResult(
            move=Move.STAY,
            reasoning=f"repli après échec : {last_error}",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
            retries=self.cfg.max_retries,
            fallback=True,
        )
