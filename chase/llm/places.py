"""Référentiel de lieux commun aux deux poursuivants (bras A1bis et A2).

Les noms ne dépendent que de la carte : ils sont identiques pour les deux
poursuivants et dans tous les bras.

- Carrefours : cases libres à au moins 3 voisines libres, nommées K1, K2… dans
  l'ordre de lecture (du nord au sud, puis de l'ouest à l'est).
- Couloirs : composantes connexes des autres cases, nommées C1, C2… dans
  l'ordre de lecture de leur première case. Chacun est orienté depuis son
  extrémité voisine du carrefour de plus petit nom ; un cul-de-sac vient en
  dernier. Au-delà de MAX_SEGMENT cases, un couloir est coupé en tronçons
  consécutifs et équilibrés, suffixés a, b, c… dans ce sens.
- Case précise : lieu et rang dans le sens de lecture (« C6b.3 ») ; une case de
  carrefour porte le nom du carrefour.

Voir docs/superpowers/specs/2026-09-29-jalon2-a2-design.md, § Référentiel de lieux.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np

from ..graph import MazeGraph

MAX_SEGMENT = 5  # cases par tronçon au plus : à peu près la portée du champ de vision


@dataclass(frozen=True)
class Place:
    name: str
    cells: tuple[int, ...]  # indices MazeGraph, dans le sens de lecture


class Places:
    def __init__(self, places: list[Place], n_cells: int):
        self.places = places
        self.cell_name = [""] * n_cells                # nom de chaque case
        self.place_of = np.full(n_cells, -1, dtype=np.int64)  # lieu de chaque case
        for k, place in enumerate(places):
            for rank, c in enumerate(place.cells):
                self.place_of[c] = k
                self.cell_name[c] = (place.name if place.name.startswith("K")
                                     else f"{place.name}.{rank + 1}")
        self.by_name = {p.name: p for p in places}
        self.cell_by_name = {name: c for c, name in enumerate(self.cell_name)}
        # noms valides dans un message : lieux et cases
        self.names = frozenset(self.by_name) | frozenset(self.cell_by_name)

    @classmethod
    def from_graph(cls, g: MazeGraph) -> "Places":
        degree = [len(nb) for nb in g.neighbors]
        order = sorted(range(g.n), key=lambda i: (g.cells[i][1], g.cells[i][0]))
        position = {c: k for k, c in enumerate(order)}
        junctions = [i for i in order if degree[i] >= 3]
        rank = {j: k for k, j in enumerate(junctions)}
        places = [Place(f"K{k + 1}", (j,)) for k, j in enumerate(junctions)]
        seen: set[int] = set()
        corridor = 0
        for i in order:
            if degree[i] >= 3 or i in seen:
                continue
            chain = _chain(g, i, degree, rank, position)
            seen.update(chain)
            corridor += 1
            parts = -(-len(chain) // MAX_SEGMENT)
            if parts > len(string.ascii_lowercase):
                raise ValueError(f"couloir de {len(chain)} cases : trop de tronçons à nommer")
            for q, part in enumerate(np.array_split(np.array(chain), parts)):
                suffix = string.ascii_lowercase[q] if parts > 1 else ""
                places.append(Place(f"C{corridor}{suffix}", tuple(int(c) for c in part)))
        return cls(places, g.n)


def _chain(g: MazeGraph, start: int, degree: list[int], rank: dict[int, int],
           position: dict[int, int]) -> list[int]:
    """Cases du couloir qui contient `start`, dans le sens de lecture."""
    comp: set[int] = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in comp:
            continue
        comp.add(u)
        stack += [v for v in g.neighbors[u] if degree[v] < 3 and v not in comp]

    def end_key(u: int):
        near = [rank[v] for v in g.neighbors[u] if v in rank]
        # le bout voisin du plus petit carrefour d'abord, un cul-de-sac en dernier ;
        # l'ordre de lecture départage (couloir qui revient sur son carrefour)
        return (0, min(near), position[u]) if near else (1, 0, position[u])

    ends = [u for u in comp if sum(v in comp for v in g.neighbors[u]) <= 1] or list(comp)
    chain = [min(ends, key=end_key)]
    while True:
        nxt = [v for v in g.neighbors[chain[-1]] if v in comp and v not in chain]
        if not nxt:
            return chain
        chain.append(nxt[0])
