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
