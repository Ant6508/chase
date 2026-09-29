"""Tests unitaires de l'agrégation des logs LLM (chase/llm/logging.py)."""

from __future__ import annotations

from chase.llm.logging import EpisodeLLMStats, StepLog
from chase.moves import Move


def _step(prompt_tokens, completion_tokens, latency_ms, fallback, message_tokens=0):
    return StepLog(pursuer=0, move=Move.STAY, reasoning="r", prompt_tokens=prompt_tokens,
                   completion_tokens=completion_tokens, latency_ms=latency_ms,
                   retries=0, fallback=fallback, message_tokens=message_tokens)


def test_from_steps_aggregates_tokens_and_latency():
    steps = [_step(10, 5, 100.0, False), _step(20, 8, 200.0, False)]
    stats = EpisodeLLMStats.from_steps(steps, wall_time_s=12.5)
    assert stats.total_prompt_tokens == 30
    assert stats.total_completion_tokens == 13
    assert stats.total_message_tokens == 0
    assert stats.mean_latency_ms == 150.0
    assert stats.fallback_count == 0
    assert stats.wall_time_s == 12.5


def test_from_steps_counts_fallbacks():
    steps = [_step(10, 5, 100.0, True), _step(20, 8, 200.0, False), _step(5, 2, 50.0, True)]
    stats = EpisodeLLMStats.from_steps(steps, wall_time_s=1.0)
    assert stats.fallback_count == 2


def test_from_steps_handles_empty_list():
    stats = EpisodeLLMStats.from_steps([], wall_time_s=3.0)
    assert stats.total_prompt_tokens == 0
    assert stats.mean_latency_ms == 0.0
    assert stats.wall_time_s == 3.0


def _log(**changes) -> StepLog:
    fields = dict(pursuer=0, move=Move.STAY, reasoning="r", prompt_tokens=1, completion_tokens=1,
                  latency_ms=1.0, retries=0, fallback=False, message_tokens=0)
    fields.update(changes)
    return StepLog(**fields)


def test_from_steps_aggregates_messages_thinking_and_unknown_names():
    steps = [
        _log(message_tokens=30, message_out="{…}", thinking_tokens=100, unknown_names=["C99"]),
        _log(pursuer=1, message_tokens=10, message_out="{}", thinking_tokens=50),
        _log(fallback=True),  # repli : aucun message émis
    ]
    stats = EpisodeLLMStats.from_steps(steps, wall_time_s=1.0)
    assert stats.total_message_tokens == 40
    assert stats.messages_sent == 2
    assert stats.mean_message_tokens == 20.0
    assert stats.total_thinking_tokens == 150
    assert stats.unknown_name_count == 1


def test_stats_written_before_a2_are_still_readable():
    """Les journaux A1v3 n'ont que les six premiers champs."""
    stats = EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0)
    assert (stats.messages_sent, stats.mean_message_tokens, stats.total_thinking_tokens) == (0, 0.0, 0)
