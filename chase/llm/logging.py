"""Structures de journalisation des bras LLM (A1, A1bis, A2).

message_tokens compte les positions du message émis, telles que le récepteur les
lira : 0 pour A1 et A1bis. Les champs ajoutés pour A2 ont des valeurs par défaut,
pour relire les journaux d'A1v3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    # Diagnostic (traces pas à pas de run_llm.py --trace) : ce que le modèle a vu
    # et pensé. La pensée n'est pas tronquée : c'est elle qu'on vient lire.
    pos: tuple[int, int] | None = None
    perception: str = ""
    thinking: str = ""
    # Bras A1bis et A2 : prompt complet, messages et arguments bruts, qui forment le
    # jeu de données d'entraînement d'A4.
    user_prompt: str = ""                # perception, plus le message reçu en A2
    message_in: str | None = None        # message reçu, réécrit (A2)
    message_out: str | None = None       # message émis, réécrit (A2) ; None en repli
    raw_arguments: str = ""
    thinking_tokens: int = 0
    unknown_names: list[str] = field(default_factory=list)


@dataclass
class EpisodeLLMStats:
    total_prompt_tokens: int
    total_completion_tokens: int
    total_message_tokens: int
    mean_latency_ms: float
    fallback_count: int
    wall_time_s: float
    total_thinking_tokens: int = 0
    messages_sent: int = 0
    mean_message_tokens: float = 0.0  # sur les messages effectivement émis
    unknown_name_count: int = 0

    @classmethod
    def from_steps(cls, steps: list[StepLog], wall_time_s: float) -> "EpisodeLLMStats":
        if not steps:
            return cls(0, 0, 0, 0.0, 0, wall_time_s)
        sent = [s for s in steps if s.message_out is not None]
        total_message = sum(s.message_tokens for s in steps)
        return cls(
            total_prompt_tokens=sum(s.prompt_tokens for s in steps),
            total_completion_tokens=sum(s.completion_tokens for s in steps),
            total_message_tokens=total_message,
            mean_latency_ms=sum(s.latency_ms for s in steps) / len(steps),
            fallback_count=sum(1 for s in steps if s.fallback),
            wall_time_s=wall_time_s,
            total_thinking_tokens=sum(s.thinking_tokens for s in steps),
            messages_sent=len(sent),
            mean_message_tokens=total_message / len(sent) if sent else 0.0,
            unknown_name_count=sum(len(s.unknown_names) for s in steps),
        )
