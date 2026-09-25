"""Configuration du client LLM local (jalon 2, bras A1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = "google/gemma-4-12b"
    temperature: float = 0.0
    max_tokens: int = 700      # budget de raisonnement, identique pour A1/A2/A3 ;
                               # calibré empiriquement (Task 7) contre google/gemma-4-12b,
                               # dont le raisonnement interne consomme la majorité du budget
                               # avant l'appel d'outil (150 était insuffisant : finish_reason="length")
    timeout_s: float = 180.0   # calibré empiriquement (Task 7) : un appel réussi contre
                               # google/gemma-4-12b prend ~40-90 s ; un timeout trop court
                               # (30 s d'origine) fait retenter côté client sans annuler la
                               # requête côté serveur (mono-GPU, sérialisé), ce qui empile des
                               # requêtes fantômes et fait s'effondrer la latence de tous les
                               # appels suivants
    max_retries: int = 2       # tentatives supplémentaires après le premier essai
