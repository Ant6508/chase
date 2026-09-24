"""Génération du labyrinthe : arbre couvrant parfait + quelques cycles.

Convention : les tableaux sont indexés `[x, y]`, comme `Grid.state` de MultiGrid.
`free[x, y]` vaut True pour une case praticable.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _perfect_maze(size: int, rng: np.random.Generator) -> np.ndarray:
    """Backtracker récursif : arbre couvrant aléatoire, couloirs de largeur 1."""
    free = np.zeros((size, size), dtype=bool)
    cells = range(1, size - 1, 2)
    start = (int(rng.choice(cells)), int(rng.choice(cells)))
    free[start] = True
    stack = [start]
    while stack:
        x, y = stack[-1]
        options = [
            (x + 2 * dx, y + 2 * dy, dx, dy)
            for dx, dy in _STEPS
            if 0 < x + 2 * dx < size - 1 and 0 < y + 2 * dy < size - 1
            and not free[x + 2 * dx, y + 2 * dy]
        ]
        if not options:
            stack.pop()
            continue
        nx, ny, dx, dy = options[rng.integers(len(options))]
        free[x + dx, y + dy] = True
        free[nx, ny] = True
        stack.append((nx, ny))
    return free


def _path_length(free: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> int:
    dist = {a: 0}
    queue = deque([a])
    while queue:
        cur = queue.popleft()
        if cur == b:
            return dist[cur]
        for dx, dy in _STEPS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if free[nxt] and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    raise ValueError("cases non connectées")


def generate_maze(size: int, n_loops: int, min_loop_len: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Labyrinthe majoritairement arborescent avec `n_loops` cycles ajoutés.

    Chaque cycle est créé en ouvrant un mur séparant deux cellules voisines du
    labyrinthe dont la distance de chemin est d'au moins `min_loop_len - 2`,
    ce qui garantit un cycle d'au moins `min_loop_len` cases (pas de boucle
    triviale autour d'un seul pilier).
    """
    if size % 2 == 0 or size < 5:
        raise ValueError("size doit être impair et >= 5")
    free = _perfect_maze(size, rng)

    for _ in range(n_loops):
        candidates = []
        for x in range(1, size - 1):
            for y in range(1, size - 1):
                if free[x, y] or (x % 2) == (y % 2):
                    continue  # seuls les murs entre deux cellules sont ouvrables
                a, b = ((x - 1, y), (x + 1, y)) if x % 2 == 0 else ((x, y - 1), (x, y + 1))
                if _path_length(free, a, b) + 2 >= min_loop_len:
                    candidates.append((x, y))
        if not candidates:
            raise ValueError("impossible d'ajouter un cycle : réduire min_loop_len")
        x, y = candidates[rng.integers(len(candidates))]
        free[x, y] = True
    return free
