"""Configuration du client LLM (jalon 2, bras A1) — LM Studio headless sur un pod RunPod."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    # Pod RunPod jhbk30ligdrozo (RTX 4090, LM Studio headless, --parallel 32) ;
    # remplace le LM Studio local du pilote initial — même client, seule l'adresse
    # change (décision du 2026-09-25, voir design doc § Modèle).
    base_url: str = "https://jhbk30ligdrozo-1234.proxy.runpod.net/v1"
    model: str = "gemma-4-12b-a1"
    temperature: float = 0.0
    max_tokens: int = 700      # budget de raisonnement, identique pour A1/A2/A3 ;
                               # calibré empiriquement (Task 7, LM Studio local) contre
                               # google/gemma-4-12b, dont le raisonnement interne consomme la
                               # majorité du budget avant l'appel d'outil (150 était insuffisant :
                               # finish_reason="length"). Toujours valide sur RunPod : même
                               # modèle, même température, distribution de tokens inchangée.
    timeout_s: float = 180.0   # calibré empiriquement (Task 7, LM Studio local) : un timeout
                               # trop court fait retenter côté client sans annuler la requête
                               # côté serveur, ce qui empile des requêtes fantômes et effondre la
                               # latence des appels suivants. Sur RunPod (RTX 4090 dédié, appel
                               # mesuré ~2,5 s) cette marge est large mais reste un plafond de
                               # sécurité peu coûteux à garder.
    max_retries: int = 2       # tentatives supplémentaires après le premier essai
