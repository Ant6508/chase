"""Structures de journalisation pour les bras LLM (stables entre A1/A2/A3).

message_tokens vaut toujours 0 pour A1 (pas de canal de communication) ; le
champ existe déjà pour que A2/A3 réutilisent le même schéma sans le changer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..moves import Move

REASONING_LOG_CHARS = 200  # troncature du texte de raisonnement dans les logs


@dataclass
class StepLog:
    pursuer: int
    move: Move
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    retries: int
    fallback: bool
    message_tokens: int


@dataclass
class EpisodeLLMStats:
    total_prompt_tokens: int
    total_completion_tokens: int
    total_message_tokens: int
    mean_latency_ms: float
    fallback_count: int
    wall_time_s: float

    @classmethod
    def from_steps(cls, steps: list[StepLog], wall_time_s: float) -> "EpisodeLLMStats":
        if not steps:
            return cls(0, 0, 0, 0.0, 0, wall_time_s)
        return cls(
            total_prompt_tokens=sum(s.prompt_tokens for s in steps),
            total_completion_tokens=sum(s.completion_tokens for s in steps),
            total_message_tokens=sum(s.message_tokens for s in steps),
            mean_latency_ms=sum(s.latency_ms for s in steps) / len(steps),
            fallback_count=sum(1 for s in steps if s.fallback),
            wall_time_s=wall_time_s,
        )
