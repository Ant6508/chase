"""Politique scriptée de la cible. Figée après le jalon 1 (SPEC §2).

- Poursuivant visible, ou vu il y a moins de `target_memory` pas :
  * si la cible n'est pas sur un cycle et qu'une case de cycle est atteignable
    strictement avant tout poursuivant connu, elle y court (plus court chemin
    vers la plus proche de ces cases) : sur un cycle, un poursuivant seul ne
    peut pas la prendre à vitesse égale ;
  * sinon fuite gloutonne : on maximise la distance de chemin au poursuivant
    connu le plus proche ; à égalité on préfère se rapprocher d'un cycle, puis
    on tire au sort.
- Sinon : marche aléatoire sans demi-tour (sauf cul-de-sac), pondérée par
  `exp(-bias * distance_au_cycle)` de la case suivante.

La cible perçoit avec le même champ de vision orienté que les poursuivants.
"""

from __future__ import annotations

import numpy as np

from .graph import MazeGraph
from .moves import MOVE_DELTAS, Move

_OPPOSITE = {Move.NORTH: Move.SOUTH, Move.SOUTH: Move.NORTH,
             Move.EAST: Move.WEST, Move.WEST: Move.EAST}


class ScriptedTarget:

    def __init__(self, graph: MazeGraph, rng: np.random.Generator,
                 memory: int, cycle_bias: float):
        self.g = graph
        self.rng = rng
        self.memory = memory
        self.bias = cycle_bias
        self.threats: dict[int, tuple[int, int]] = {}  # poursuivant -> (case, âge)
        self.last_move = Move.STAY

    def act(self, pos: tuple[int, int], visible: np.ndarray,
            pursuers: list[tuple[int, int]]) -> Move:
        g = self.g
        for p, ppos in enumerate(pursuers):
            if visible[ppos]:
                self.threats[p] = (int(g.index[ppos]), 0)
        self.threats = {p: (c, age + 1) for p, (c, age) in self.threats.items()
                        if age < self.memory}

        here = int(g.index[pos])
        options = self._options(pos)
        if self.threats:
            threat_cells = [c for c, _ in self.threats.values()]
            d_threat = g.dist[threat_cells].min(axis=0)
            refuge = np.flatnonzero(g.on_cycle & (g.dist[here] < d_threat))
            if not g.on_cycle[here] and len(refuge):
                goal = refuge[np.argmin(g.dist[here, refuge])]
                moves = [m for m, n in options if g.dist[n, goal] == g.dist[here, goal] - 1]
                move = moves[self.rng.integers(len(moves))]
                self.last_move = move
                return move
            scored = [(min(g.dist[n, threat_cells]), -g.dist_to_cycle[n], m)
                      for m, n in options]
            best = max(s[:2] for s in scored)
            moves = [m for *s, m in scored if tuple(s) == best]
        else:
            moves = [m for m, n in options if n != here]
            back = _OPPOSITE.get(self.last_move)
            if len(moves) > 1 and back in moves:
                moves.remove(back)
            cells = [dict(options)[m] for m in moves]
            w = np.exp(-self.bias * g.dist_to_cycle[cells].astype(float))
            moves = [moves[self.rng.choice(len(moves), p=w / w.sum())]]
        move = moves[self.rng.integers(len(moves))]
        self.last_move = move
        return move

    def _options(self, pos: tuple[int, int]) -> list[tuple[Move, int]]:
        out = []
        for m, (dx, dy) in MOVE_DELTAS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if self.g.free[nxt]:
                out.append((m, int(self.g.index[nxt])))
        return out
