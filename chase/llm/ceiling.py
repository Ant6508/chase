"""Plafond mécanique du protocole A2 (P2), sans LLM.

P2 prend les décisions de R2 (répartition de Voronoï en exploration, tenaille
quand la cible est localisée), mais il ne connaît de son coéquipier que ce que
transporte le message A2 du pas précédent, au grain des lieux :
- sa position : la case `moi` ;
- sa croyance : l'union des cases de ses lieux `candidates`, ou la case `cible`
  quand il la voyait, propagée d'un pas puis intersectée avec la croyance
  individuelle du récepteur.

Il n'y a pas d'accumulation. Chaque poursuivant garde sa croyance individuelle
(R1) et ne refait l'intersection que pour décider, comme un LLM sans mémoire qui
relit le message à chaque pas. Sans message (premier pas), il décide comme R1.

Si P2 perd plus de la moitié de l'écart de captures R1 -> R2, c'est le protocole
qui est en cause, pas le LLM (docs/superpowers/specs/2026-09-29-jalon2-a2-design.md,
§ Validation, étape 1).
"""

from __future__ import annotations

import numpy as np

from ..belief import diffuse, observe, observe_prob, propagate
from ..config import ChaseConfig
from ..graph import MazeGraph
from ..moves import Move
from ..policies import GreedyPursuers, Percept
from .places import Places

INTENTION_STEPS = 5  # cases du chemin prévu résumées dans `intention`


def write_message(g: MazeGraph, places: Places, pos: tuple[int, int], belief: np.ndarray,
                  prob: np.ndarray, target_seen: tuple[int, int] | None,
                  path_ahead: list[int]) -> dict:
    """Message A2 rédigé mécaniquement depuis la croyance individuelle d'un
    poursuivant : tous les lieux qui contiennent une candidate, avec leur part de
    probabilité arrondie, et les lieux successifs de son chemin prévu."""
    candidates = {}
    for place in places.places:
        cells = np.array(place.cells, dtype=np.int64)
        xs, ys = g.xy[cells, 0], g.xy[cells, 1]
        if belief[xs, ys].any():
            # au moins 1 : à 0, un lieu candidat se confondrait avec un lieu vu vide,
            # alors que la perception dit seulement « moins de 1 % ».
            candidates[place.name] = max(1, int(round(100 * float(prob[xs, ys].sum()))))
    intention: list[str] = []
    for c in path_ahead:
        name = places.places[places.place_of[c]].name
        if not intention or intention[-1] != name:
            intention.append(name)
    return {"moi": places.cell_name[g.index[pos]],
            "cible": None if target_seen is None else places.cell_name[g.index[target_seen]],
            "candidates": candidates, "intention": intention, "je_couvre": None}


class ProtocolPursuers(GreedyPursuers):

    def __init__(self, cfg: ChaseConfig):
        if cfg.n_pursuers != 2:
            raise ValueError("P2 suppose 2 poursuivants : chaque message va à l'autre")
        super().__init__(cfg, fused=True)
        self.name = "P2"

    def reset(self, graph, rng):
        super().reset(graph, rng)
        self.places = Places.from_graph(graph)
        n = self.cfg.n_pursuers
        self._own: list[np.ndarray | None] = [None] * n    # croyances individuelles (R1)
        self._own_p: list[np.ndarray | None] = [None] * n
        self._inbox: list[dict | None] = [None] * n        # message reçu du coéquipier
        self.sent: list[dict | None] = [None] * n          # dernier message écrit par chacun

    def update(self, percepts: list[Percept]):
        free = self.g.free
        uniform = free / free.sum()
        for i, p in enumerate(percepts):
            first = self._own[i] is None
            prior = free if first else propagate(self._own[i], free)
            prior_p = uniform if first else diffuse(self._own_p[i], free)
            self._own[i] = observe(prior, p.visible, p.target_seen)
            self._own_p[i] = observe_prob(prior_p, p.visible, p.target_seen)
        # croyance de décision : la sienne, intersectée avec ce que dit le message reçu
        self._beliefs, self._probs = [], []
        for i in range(len(percepts)):
            belief, prob = self._own[i], self._own_p[i]
            if self._inbox[i] is not None:
                belief = belief & propagate(self._support(self._inbox[i]), free)
                prob = np.where(belief, prob, 0.0)
                total = prob.sum()
                prob = prob / total if total > 0 else belief / belief.sum()
            self._beliefs.append(belief)
            self._probs.append(prob)

    def _support(self, message: dict) -> np.ndarray:
        """Cases où l'auteur du message situait la cible quand il l'a écrit."""
        places = self.places
        if message["cible"] is not None:
            cells = [places.cell_by_name[message["cible"]]]
        else:
            cells = [c for name in message["candidates"] for c in places.by_name[name].cells]
        out = np.zeros(self.g.free.shape, dtype=bool)
        for c in cells:
            out[self.g.cells[c]] = True
        return out

    def act(self, percepts: list[Percept]):
        moves = []
        for i, p in enumerate(percepts):
            # sans message reçu (premier pas), le coéquipier est inconnu : décision de R1
            self.fused = self._inbox[i] is not None
            moves.append(self._act_one(i, p))
        # P2 est fondamentalement un poursuivant fusionné (hérité de
        # GreedyPursuers(fused=True)) : la bascule à False ci-dessus n'est qu'un
        # artifice local, pas à pas, pour le tout premier message (aucun reçu).
        # _act_one lit self.fused (chase/policies.py) pour calculer `me`, et
        # _team_positions (ci-dessous) en dépend aussi : avec fused=False, elle
        # renvoie une équipe réduite à [here], sans coéquipier. Si on laissait
        # self.fused à False après la boucle, le poursuivant perdrait son
        # coéquipier de vue (team = [here]) pour tout ce qui suit — écriture du
        # chemin prévu, prochains appels — alors qu'un message vient justement
        # d'être échangé.
        self.fused = True
        # Décider avant d'écrire, pour toute l'équipe avant de rien stocker : sinon
        # le pursuivant traité en second lirait, via self._inbox, le message que
        # son coéquipier vient d'écrire à cette même itération (message « au même
        # pas », zéro délai) au lieu de celui reçu à l'itération précédente. Voir
        # test_decision_loop_never_sees_a_message_written_this_same_act_call.
        outbox: list[dict | None] = [None] * len(percepts)
        for i, p in enumerate(percepts):
            here = int(self.g.index[p.pos])
            outbox[1 - i] = write_message(self.g, self.places, p.pos, self._own[i],
                                          self._own_p[i], p.target_seen,
                                          self._path_ahead(i, here, moves[i]))
        self.sent = [outbox[1], outbox[0]]
        self._inbox = outbox
        return moves

    def _team_positions(self, i: int, me: tuple[int, int]) -> list[int]:
        here = int(self.g.index[me])
        if not self.fused:
            return [here]
        mate = self.places.cell_by_name[self._inbox[i]["moi"]]
        return [here if k == i else mate for k in range(self.cfg.n_pursuers)]

    def _path_ahead(self, i: int, here: int, move: Move,
                    steps: int = INTENTION_STEPS) -> list[int]:
        """Début du chemin réellement prévu : la case où mène le coup joué (rien si ce
        coup est STAY ou bute contre un mur), puis, au plus court, vers l'objectif
        courant en exploration ou vers la candidate la plus proche en poursuite,
        jusqu'à `steps` cases en tout. La première case suit le coup effectivement
        joué (la tenaille, à plusieurs, diffère du plus court chemin vers une
        candidate) ; le reste départage déterministement, sans consommer le
        générateur aléatoire des décisions."""
        g = self.g
        path: list[int] = []
        cur = here
        if move != Move.STAY:
            nxt = next((n for n in g.neighbors[cur] if g.move_between(cur, n) == move), None)
            if nxt is not None:
                path.append(nxt)
                cur = nxt
        goal = self._goals[i]
        if goal is None:
            cand = g.index[self._beliefs[i]]
            if len(cand) == 0:
                return path
            goal = int(cand[np.argmin(g.dist[cur, cand])])
        field = g.dist[:, goal]
        while len(path) < steps and field[cur] > 0:
            cur = min(n for n in g.neighbors[cur] if field[n] == field[cur] - 1)
            path.append(cur)
        return path
