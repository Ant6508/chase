"""Politiques scriptées des poursuivants pour le jalon 1.

- R0 : aléatoire uniforme.
- R1 : gloutonne locale sans communication. Chaque poursuivant tient sa propre
  croyance et ignore tout de son coéquipier, position comprise.
- R2 : gloutonne à croyance fusionnée. Les croyances sont fusionnées avec
  `comm_delay` pas de retard, comme le canal des bras LLM. Les poursuivants
  se répartissent le travail : en exploration, chacun ne vise que les cases
  plus proches de lui que des autres (partition de Voronoï) ; quand la cible
  est localisée (croyance de taille <= `track_threshold`), chacun joue le pas
  qui réduit le plus le territoire de la cible (cases qu'elle atteint avant
  tout poursuivant connu), ce qui produit la prise en tenaille. Seul (R1), on
  la poursuit directement : pour un poursuivant isolé, cette règle fait mieux
  que celle du territoire, et R1 ne doit pas être affaibli artificiellement.

En exploration, l'objectif est choisi sur une carte de probabilité posée sur
l'ensemble candidat (marche aléatoire de la cible, même support que
l'ensemble) : on vise la case qui maximise la masse de probabilité à moins de
`EXPLORE_RADIUS` cases, divisée par la distance à parcourir. C'est une
heuristique de décision, pas une perception : l'ensemble candidat reste
l'objet exact partagé et mesuré.

R1 et R2 partagent le même code de décision : seuls changent la source de la
croyance et la connaissance du coéquipier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belief import diffuse, observe, observe_prob, propagate
from .config import ChaseConfig
from .graph import UNREACHABLE, MazeGraph
from .moves import Move

EXPLORE_RADIUS = 3    # rayon (en cases de chemin) de la masse visée en exploration
EXPLORE_OFFSET = 4    # score = masse / (distance + EXPLORE_OFFSET)
GOAL_HYSTERESIS = 0.8  # on garde l'objectif courant tant qu'il vaut >= 80 % du meilleur


@dataclass
class Percept:
    """Ce qu'un poursuivant perçoit à un pas donné."""
    pos: tuple[int, int]
    visible: np.ndarray
    target_seen: tuple[int, int] | None


class PursuerPolicy:
    name = "?"

    def __init__(self, cfg: ChaseConfig):
        self.cfg = cfg

    def reset(self, graph: MazeGraph, rng: np.random.Generator):
        self.g = graph
        self.rng = rng

    def update(self, percepts: list[Percept]):
        """Appelée après la réinitialisation puis après chaque pas."""

    def act(self, percepts: list[Percept]) -> list[Move]:
        raise NotImplementedError

    def beliefs(self) -> list[np.ndarray]:
        """Croyance courante de chaque poursuivant (pour les tests)."""
        return []


class RandomPursuers(PursuerPolicy):
    name = "R0"

    def act(self, percepts):
        return [Move(int(self.rng.integers(len(Move)))) for _ in percepts]


class GreedyPursuers(PursuerPolicy):

    def __init__(self, cfg: ChaseConfig, fused: bool):
        super().__init__(cfg)
        self.fused = fused
        self.name = "R2" if fused else "R1"

    def reset(self, graph, rng):
        super().reset(graph, rng)
        n = self.cfg.n_pursuers
        self._beliefs: list[np.ndarray | None] = [None] * n
        self._shared: np.ndarray | None = None
        self._probs: list[np.ndarray | None] = [None] * n
        self._shared_prob: np.ndarray | None = None
        self._near = (graph.dist <= EXPLORE_RADIUS).astype(float)
        self._goals: list[int | None] = [None] * n
        self._known_pos: list[tuple[int, int] | None] = [None] * n

    # -------------------------------------------------------------- croyances

    def update(self, percepts):
        free = self.g.free
        uniform = free / free.sum()
        if not self.fused:
            for i, p in enumerate(percepts):
                first = self._beliefs[i] is None
                prior = free if first else propagate(self._beliefs[i], free)
                prior_p = uniform if first else diffuse(self._probs[i], free)
                self._beliefs[i] = observe(prior, p.visible, p.target_seen)
                self._probs[i] = observe_prob(prior_p, p.visible, p.target_seen)
            return

        # Fusion exacte avec retard : la croyance partagée intègre toutes les
        # observations jusqu'au pas t - delay ; chacun y ajoute les siennes.
        first = self._shared is None
        prior = free if first else propagate(self._shared, free)
        prior_p = uniform if first else diffuse(self._shared_prob, free)
        shared, shared_p = prior, prior_p
        for p in percepts:
            shared = observe(shared, p.visible, p.target_seen)
            shared_p = observe_prob(shared_p, p.visible, p.target_seen)
        if self.cfg.comm_delay == 0:
            self._beliefs = [shared.copy() for _ in percepts]
            self._probs = [shared_p.copy() for _ in percepts]
        else:
            self._beliefs = [observe(prior, p.visible, p.target_seen) for p in percepts]
            self._probs = [observe_prob(prior_p, p.visible, p.target_seen) for p in percepts]
        # Positions des coéquipiers telles que reçues : avec retard d'un pas,
        # c'est la position de l'étape précédente (au départ, la position initiale).
        if self.cfg.comm_delay == 0 or self._shared is None:
            self._known_pos = [p.pos for p in percepts]
        self._next_known = [p.pos for p in percepts]
        self._shared = shared
        self._shared_prob = shared_p

    def beliefs(self):
        return list(self._beliefs)

    # ---------------------------------------------------------------- décision

    def act(self, percepts):
        moves = [self._act_one(i, p) for i, p in enumerate(percepts)]
        if self.fused and self.cfg.comm_delay > 0:
            self._known_pos = self._next_known
        return moves

    def _team_positions(self, i: int, me: tuple[int, int]) -> list[int]:
        """Cases des poursuivants connues de `i` (lui-même en position réelle)."""
        if not self.fused:
            return [int(self.g.index[me])]
        return [int(self.g.index[me if k == i else pos]) for k, pos in enumerate(self._known_pos)]

    def _act_one(self, i: int, p: Percept) -> Move:
        g = self.g
        here = int(g.index[p.pos])
        cand = g.index[self._beliefs[i]]
        if len(cand) == 0:
            return Move.STAY
        team = self._team_positions(i, p.pos)
        me = team.index(here) if not self.fused else i

        if len(cand) <= self.cfg.track_threshold:
            self._goals[i] = None
            d_target = g.dist[cand].min(axis=0)
            others = [c for k, c in enumerate(team) if k != me]
            if not others:
                # Seul : poursuite directe par le plus court chemin.
                return self._descend(here, d_target)
            # À plusieurs : on joue le pas qui minimise le territoire de la
            # cible, c'est-à-dire les cases qu'elle atteint strictement avant
            # tout poursuivant connu. C'est ce qui produit la tenaille.
            d_others = g.dist[others].min(axis=0)
            options = [here, *g.neighbors[here]]
            keys = [(int((d_target < np.minimum(g.dist[o], d_others)).sum()), int(d_target[o]))
                    for o in options]
            best = min(keys)
            ties = [o for o, k in zip(options, keys) if k == best]
            nxt = ties[self.rng.integers(len(ties))]
            return Move.STAY if nxt == here else g.move_between(here, nxt)

        # Exploration : on ne vise que les cases dont on est le poursuivant le plus proche.
        prob = self._probs[i][g.xy[:, 0], g.xy[:, 1]]
        mass = self._near @ prob
        owner = np.argmin(g.dist[team], axis=0)
        allowed = (owner == me) & (mass > 0)
        if not allowed.any():
            allowed = mass > 0
        score = np.where(allowed, mass / (g.dist[here] + EXPLORE_OFFSET), -1.0)
        best = float(score.max())
        goal = self._goals[i]
        if goal is None or score[goal] < GOAL_HYSTERESIS * best or goal == here:
            ties = np.flatnonzero(score == best)
            goal = int(ties[self.rng.integers(len(ties))])
            self._goals[i] = goal
        return self._descend(here, g.dist[:, goal])

    def _descend(self, here: int, field: np.ndarray) -> Move:
        g = self.g
        if field[here] == 0 or field[here] == UNREACHABLE:
            return Move.STAY
        steps = [n for n in g.neighbors[here] if field[n] == field[here] - 1]
        nxt = steps[self.rng.integers(len(steps))]
        return g.move_between(here, nxt)


def make_policy(name: str, cfg: ChaseConfig) -> PursuerPolicy:
    if name == "R0":
        return RandomPursuers(cfg)
    if name == "R1":
        return GreedyPursuers(cfg, fused=False)
    if name == "R2":
        return GreedyPursuers(cfg, fused=True)
    raise ValueError(name)
