"""Configuration du client LLM local (jalon 2, bras A1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = "google/gemma-4-12b"
    temperature: float = 0.0
    max_tokens: int = 150      # budget de raisonnement, identique pour A1/A2/A3
    timeout_s: float = 30.0
    max_retries: int = 2       # tentatives supplémentaires après le premier essai
