"""Client LLM contre un serveur LM Studio (API compatible OpenAI).

Chaque appel à `decide` est un tour isolé : aucun historique de conversation
n'est conservé entre les pas, pour garder un budget de tokens borné quelle
que soit la longueur de l'épisode (contexte chargé : 13056 tokens).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
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


def move_tool(with_message: bool) -> dict:
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
    finish_reason: str = ""      # raison d'arrêt de la réponse (« length » : réponse coupée)
    # raison de chaque tentative échouée, dans l'ordre ; en repli, elles y sont toutes
    attempt_errors: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    def decide(self, system_prompt: str, user_prompt: str,
               with_message: bool = False) -> LLMCallResult: ...


def _clean(text: str) -> str:
    """Texte privé (pensée, justification) rendu encodable en UTF-8 : un demi-caractère
    de substitution isolé ferait échouer `count_tokens` et l'écriture de la trace."""
    return text.encode("utf-8", "replace").decode("utf-8")


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
        errors: list[str] = []
        # pensée, arguments bruts et finish_reason de la dernière réponse reçue, gardés
        # dans le repli pour le diagnostic
        last_thinking, last_raw, last_finish = "", "", ""
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
                    tools=[move_tool(with_message)],
                    tool_choice="required",
                )
            except Exception as exc:
                errors.append(str(exc))
                continue
            latency_ms = (time.monotonic() - t0) * 1000
            # Aucune exception ne sort de `decide` : une réponse illisible est une
            # tentative échouée comme une autre.
            try:
                choice = response.choices[0]
                finish = getattr(choice, "finish_reason", None) or ""
                last_finish = finish
                last_thinking = _clean(getattr(choice.message, "reasoning_content", None) or "")
                parsed = _parse_tool_call(response)
                if parsed is None:
                    errors.append("réponse sans appel d'outil valide "
                                  f"(finish_reason={finish or 'inconnu'})")
                    continue
                direction, reasoning, args, raw = parsed
                last_raw = raw
                message = None
                if with_message:
                    message = args.get("message")
                    problem = validate(message)
                    if problem is not None:
                        errors.append(f"message invalide : {problem}")
                        continue
                usage = response.usage
                return LLMCallResult(
                    move=Move[direction],
                    reasoning=_clean(reasoning),
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency_ms,
                    retries=attempt,
                    fallback=False,
                    thinking=last_thinking,
                    message=message,
                    raw_arguments=raw,
                    finish_reason=finish,
                    attempt_errors=errors,
                )
            except Exception as exc:
                errors.append(f"réponse illisible : {exc!r}")
        return LLMCallResult(
            move=Move.STAY,
            reasoning=f"repli après échec : {errors[-1] if errors else ''}",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
            retries=self.cfg.max_retries,
            fallback=True,
            thinking=last_thinking,
            raw_arguments=last_raw,
            finish_reason=last_finish,
            attempt_errors=errors,
        )
