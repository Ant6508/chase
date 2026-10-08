"""Politique de poursuite pilotée par un LLM : bras A1, A1bis et A2.

Dans les trois bras, la croyance est individuelle et identique à celle de
GreedyPursuers(fused=False) (R1) : le harnais ne fusionne jamais rien. Seule la
décision (act) change : elle vient d'un appel LLM au lieu de l'heuristique.

- A1 : perception de la campagne A1v3, aucun contact avec le coéquipier.
- A1bis : même perception, plus les lieux nommés (chase/llm/places.py).
- A2 : perception d'A1bis, plus le message écrit par le coéquipier au pas
  précédent, en fin de prompt. Chacun écrit le sien dans le même appel. Le
  harnais ne lit jamais le contenu d'un message pour décider ou calculer quoi
  que ce soit : il le valide, le réécrit, le compte, relève les noms inconnus
  et le transmet (docs/superpowers/specs/2026-09-29-jalon2-a2-design.md).
"""

from __future__ import annotations

import numpy as np

from ..belief import diffuse, observe, observe_prob, propagate
from ..config import ChaseConfig
from ..graph import MazeGraph
from ..moves import Move
from ..policies import Percept, PursuerPolicy
from . import message as msg
from .client import LLMClient, LMStudioClient
from .config import LLMConfig
from .logging import REASONING_LOG_CHARS, StepLog
from .places import Places
from .prompts import ARMS, CHANNEL_ARMS, MESSAGE_SPECS, build_perception, message_block, system_prompt


class LLMPursuers(PursuerPolicy):
    """Politique gloutonne (croyance de R1), décidée à chaque pas par un appel LLM."""

    def __init__(self, cfg: ChaseConfig, llm_cfg: LLMConfig, client: LLMClient | None = None,
                 arm: str = "A1"):
        if arm not in ARMS:
            raise ValueError(f"bras inconnu : {arm}")
        if arm in CHANNEL_ARMS and cfg.n_pursuers != 2:
            raise ValueError("A2 suppose 2 poursuivants : chaque message va à l'autre")
        super().__init__(cfg)
        self.arm = arm
        self.name = arm
        self.llm_cfg = llm_cfg
        self.client = client or LMStudioClient(llm_cfg)
        self.system_prompt = system_prompt(arm)
        self.message_spec = MESSAGE_SPECS.get(arm)  # None hors canal
        self.step_logs: list[StepLog] = []

    def reset(self, graph: MazeGraph, rng: np.random.Generator):
        super().reset(graph, rng)
        n = self.cfg.n_pursuers
        self._beliefs: list[np.ndarray | None] = [None] * n
        self._probs: list[np.ndarray | None] = [None] * n
        self.places = None if self.arm == "A1" else Places.from_graph(graph)
        self._inbox: list[str | None] = [None] * n  # message reçu, déjà réécrit
        self.step_logs = []

    def update(self, percepts: list[Percept]):
        free = self.g.free
        uniform = free / free.sum()
        for i, p in enumerate(percepts):
            first = self._beliefs[i] is None
            prior = free if first else propagate(self._beliefs[i], free)
            prior_p = uniform if first else diffuse(self._probs[i], free)
            self._beliefs[i] = observe(prior, p.visible, p.target_seen)
            self._probs[i] = observe_prob(prior_p, p.visible, p.target_seen)

    def beliefs(self) -> list[np.ndarray]:
        return list(self._beliefs)

    def probs(self) -> list[np.ndarray]:
        return list(self._probs)

    def act(self, percepts: list[Percept]) -> list[Move]:
        channel = self.arm in CHANNEL_ARMS
        outbox: list[str | None] = [None] * len(percepts)
        moves = []
        for i, p in enumerate(percepts):
            perception = build_perception(p.pos, self.g, self._beliefs[i], self._probs[i],
                                          p.target_seen, self.cfg.track_threshold, self.places)
            received = self._inbox[i] if channel else None
            if channel:
                user = f"{perception}\n\n{message_block(received)}"
                result = self.client.decide(self.system_prompt, user, message_spec=self.message_spec)
            else:
                # A1 et A1bis appellent le client exactement comme la campagne A1v3
                user = perception
                result = self.client.decide(self.system_prompt, user)
            # hors A2, on ignore le message même si un client de test en renvoie un
            message = result.message if channel else None
            sent = msg.render(message) if message is not None else None
            if channel:
                outbox[1 - i] = sent  # deux poursuivants : le message va à l'autre
            self.step_logs.append(StepLog(
                pursuer=i,
                move=result.move,
                reasoning=result.reasoning[:REASONING_LOG_CHARS],
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                latency_ms=result.latency_ms,
                retries=result.retries,
                fallback=result.fallback,
                message_tokens=msg.count_tokens(sent) if sent is not None else 0,
                pos=p.pos,
                perception=perception,
                thinking=result.thinking,
                user_prompt=user,
                message_in=received,
                message_out=sent,
                raw_arguments=result.raw_arguments,
                finish_reason=result.finish_reason,
                attempt_errors=list(result.attempt_errors),
                thinking_tokens=msg.count_tokens(result.thinking),
                unknown_names=msg.unknown_names(message, self.places) if message is not None else [],
            ))
            moves.append(result.move)
        if channel:
            # après les deux appels seulement : un message écrit au pas t n'est lu qu'au pas t+1
            self._inbox = outbox
        return moves
