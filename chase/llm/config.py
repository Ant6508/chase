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
    max_tokens: int = 1300     # budget de raisonnement, identique pour A1/A2/A3 ;
                               # recalibré empiriquement contre le pod RunPod (700 laissait 53,6 %
                               # des appels du pilote finir en finish_reason="length",
                               # reasoning_content seul consommant 697-700/700 tokens sans jamais
                               # atteindre l'appel d'outil — confirmé sur les logs serveur
                               # /root/.lmstudio/server-logs/). Rejouer le prompt exact d'un échec
                               # avec max_tokens=3000 converge à 875 tokens : ce n'est pas une
                               # boucle infinie, juste un budget trop court pour les cas les plus
                               # chargés. 1300 laisse une marge confortable au-dessus de ce
                               # maximum observé.
    timeout_s: float = 180.0   # calibré empiriquement (Task 7, LM Studio local) : un timeout
                               # trop court fait retenter côté client sans annuler la requête
                               # côté serveur, ce qui empile des requêtes fantômes et effondre la
                               # latence des appels suivants. Sur RunPod (RTX 4090 dédié, appel
                               # mesuré ~2,5 s) cette marge est large mais reste un plafond de
                               # sécurité peu coûteux à garder.
    max_retries: int = 2       # tentatives supplémentaires après le premier essai
