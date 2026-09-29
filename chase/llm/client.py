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
from .message import MESSAGE_SCHEMA, validate

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


def _move_tool(with_message: bool) -> dict:
    """L'outil `move`, avec le paramètre `message` obligatoire en A2 seulement."""
    if not with_message:
        return _MOVE_TOOL
    fn = _MOVE_TOOL["function"]
    params = fn["parameters"]
    return {"type": "function", "function": {**fn, "parameters": {
        **params,
        "properties": {**params["properties"], "message": MESSAGE_SCHEMA},
        "required": [*params["required"], "message"],
    }}}


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
    message: dict | None = None  # message validé (A2) ; None hors canal ou en repli
    raw_arguments: str = ""      # arguments bruts de l'appel d'outil (A4 y situera le message)


class LLMClient(Protocol):
    def decide(self, system_prompt: str, user_prompt: str,
               with_message: bool = False) -> LLMCallResult: ...


def _parse_tool_call(response) -> tuple[str, str, dict, str] | None:
    """(direction, justification, arguments décodés, arguments bruts), ou None."""
    try:
        raw = response.choices[0].message.tool_calls[0].function.arguments
        args = json.loads(raw)
        direction = args["direction"]
    except (IndexError, AttributeError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(direction, str) or direction not in Move.__members__:
        return None
    return direction, str(args.get("reasoning", "")), args, raw


class LMStudioClient:
    """Un appel = un tour isolé : pas d'historique de conversation entre les pas."""

    def __init__(self, cfg: LLMConfig, client=None):
        self.cfg = cfg
        self._client = client or OpenAI(base_url=cfg.base_url, api_key="lm-studio")

    def decide(self, system_prompt: str, user_prompt: str,
               with_message: bool = False) -> LLMCallResult:
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
                    tools=[_move_tool(with_message)],
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
            direction, reasoning, args, raw = parsed
            message = None
            if with_message:
                message = args.get("message")
                problem = validate(message)
                if problem is not None:
                    last_error = ValueError(f"message invalide : {problem}")
                    continue
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
                message=message,
                raw_arguments=raw,
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
