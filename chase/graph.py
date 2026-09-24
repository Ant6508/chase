"""Graphe des cases libres : distances de chemin et cases sur cycle."""

from __future__ import annotations

from collections import deque

import numpy as np

from .moves import MOVE_DELTAS, Move

UNREACHABLE = np.iinfo(np.int32).max


class MazeGraph:
    """Précalcule tout ce dont les politiques scriptées ont besoin sur une carte."""

    def __init__(self, free: np.ndarray):
        self.free = free
        self.cells: list[tuple[int, int]] = [tuple(map(int, c)) for c in np.argwhere(free)]
        self.index = np.full(free.shape, -1, dtype=np.int32)
        for i, c in enumerate(self.cells):
            self.index[c] = i
        self.n = len(self.cells)
        self.xy = np.array(self.cells, dtype=np.int32).reshape(-1, 2)
        self.neighbors: list[list[int]] = [
            [int(self.index[c[0] + dx, c[1] + dy])
             for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if free[c[0] + dx, c[1] + dy]]
            for c in self.cells
        ]
        self.dist = np.stack([self._bfs(i) for i in range(self.n)])
        self.on_cycle = self._two_core()
        cyc = np.flatnonzero(self.on_cycle)
        self.dist_to_cycle = (self.dist[:, cyc].min(axis=1) if len(cyc)
                              else np.zeros(self.n, dtype=np.int32))

    def _bfs(self, src: int, blocked: set[int] | frozenset = frozenset()) -> np.ndarray:
        d = np.full(self.n, UNREACHABLE, dtype=np.int32)
        d[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in self.neighbors[u]:
                if d[v] == UNREACHABLE and v not in blocked:
                    d[v] = d[u] + 1
                    queue.append(v)
        return d

    def _two_core(self) -> np.ndarray:
        """Cases appartenant à un cycle : ce qui survit à l'élagage des feuilles."""
        degree = np.array([len(nb) for nb in self.neighbors])
        alive = np.ones(self.n, dtype=bool)
        queue = deque(np.flatnonzero(degree <= 1).tolist())
        while queue:
            u = queue.popleft()
            if not alive[u]:
                continue
            alive[u] = False
            for v in self.neighbors[u]:
                if alive[v]:
                    degree[v] -= 1
                    if degree[v] == 1:
                        queue.append(v)
        return alive

    def bfs_avoiding(self, src: int, blocked: set[int]) -> np.ndarray:
        return self._bfs(src, frozenset(blocked))

    def move_between(self, a: int, b: int) -> Move:
        """Déplacement qui mène de la case `a` à la case voisine (ou identique) `b`."""
        ax, ay = self.cells[a]
        bx, by = self.cells[b]
        for move, (dx, dy) in MOVE_DELTAS.items():
            if (ax + dx, ay + dy) == (bx, by):
                return move
        raise ValueError("cases non adjacentes")
