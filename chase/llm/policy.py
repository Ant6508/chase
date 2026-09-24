"""Politique de poursuite pilotée par un LLM, sans communication (bras A1).

Croyance individuelle identique à GreedyPursuers(fused=False) (R1) :
chaque poursuivant tient sa propre croyance et ignore tout de son
coéquipier, position comprise. Seule la décision (act) change : elle vient
d'un appel LLM au lieu de l'heuristique gloutonne.
"""

from __future__ import annotations

import numpy as np

from ..belief import observe, propagate
from ..config import ChaseConfig
from ..graph import MazeGraph
from ..moves import Move
from ..policies import Percept, PursuerPolicy
from .client import LLMClient, LMStudioClient
from .config import LLMConfig
from .logging import REASONING_LOG_CHARS, StepLog
from .prompts import SYSTEM_PROMPT, build_perception


class LLMPursuers(PursuerPolicy):
    name = "A1"

    def __init__(self, cfg: ChaseConfig, llm_cfg: LLMConfig, client: LLMClient | None = None):
        super().__init__(cfg)
        self.llm_cfg = llm_cfg
        self.client = client or LMStudioClient(llm_cfg)
        self.step_logs: list[StepLog] = []

    def reset(self, graph: MazeGraph, rng: np.random.Generator):
        super().reset(graph, rng)
        self._beliefs: list[np.ndarray | None] = [None] * self.cfg.n_pursuers
        self.step_logs = []

    def update(self, percepts: list[Percept]):
        free = self.g.free
        for i, p in enumerate(percepts):
            first = self._beliefs[i] is None
            prior = free if first else propagate(self._beliefs[i], free)
            self._beliefs[i] = observe(prior, p.visible, p.target_seen)

    def beliefs(self) -> list[np.ndarray]:
        return list(self._beliefs)

    def act(self, percepts: list[Percept]) -> list[Move]:
        moves = []
        for i, p in enumerate(percepts):
            perception = build_perception(
                p.pos, self.g, self._beliefs[i], p.target_seen, self.cfg.track_threshold)
            result = self.client.decide(SYSTEM_PROMPT, perception)
            self.step_logs.append(StepLog(
                pursuer=i,
                move=result.move,
                reasoning=result.reasoning[:REASONING_LOG_CHARS],
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                latency_ms=result.latency_ms,
                retries=result.retries,
                fallback=result.fallback,
                message_tokens=0,
            ))
            moves.append(result.move)
        return moves
