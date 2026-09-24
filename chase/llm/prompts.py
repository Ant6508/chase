"""Prompt système et résumé de perception par direction, envoyés au LLM à
chaque pas. Pas de grille ASCII : pour un modèle de 12B, un résumé structuré
par direction évite le piège classique de confusion lignes/colonnes ou
nord/sud sur une grille texte."""

from __future__ import annotations

import numpy as np

from ..graph import MazeGraph
from ..moves import MOVE_DELTAS, Move

SYSTEM_PROMPT = """Tu es un poursuivant dans un labyrinthe, en coopération avec \
un coéquipier que tu ne peux PAS contacter : tu ne connais ni sa position ni ses \
intentions. La cible se déplace à la même vitesse que toi, une case par pas.

À chaque pas, choisis une direction parmi STAY, NORTH, SOUTH, EAST, WEST. Une \
direction praticable te fait avancer d'une case ; une direction impraticable \
(mur) te fait seulement te tourner sans avancer. La capture a lieu quand tu \
occupes la même case que la cible, ou que vous échangez vos cases dans le même \
pas.

L'« ensemble candidat » est l'ensemble des cases où la cible peut encore se \
trouver, compte tenu de tout ce que tu as observé jusqu'ici. Il se réduit \
quand tu observes une zone sans y voir la cible, et s'étend d'une case dans \
toutes les directions praticables à chaque pas où tu ne l'observes pas.

Réponds uniquement en appelant l'outil `move` avec la direction choisie et une \
justification brève."""

_DIRS = (Move.NORTH, Move.SOUTH, Move.EAST, Move.WEST)
_LABELS = {Move.NORTH: "NORD", Move.SOUTH: "SUD", Move.EAST: "EST", Move.WEST: "OUEST"}


def _bucket(dx: int, dy: int) -> Move:
    """Direction cardinale dominante d'un décalage relatif (dx, dy).

    En cas d'égalité stricte (abs(dy) == abs(dx), décalage diagonal), la
    convention retenue est de trancher vers NORD/SUD plutôt que EST/OUEST."""
    if abs(dy) >= abs(dx):
        return Move.SOUTH if dy > 0 else Move.NORTH
    return Move.EAST if dx > 0 else Move.WEST


def build_perception(pos: tuple[int, int], g: MazeGraph, belief: np.ndarray,
                      target_seen: tuple[int, int] | None,
                      track_threshold: int) -> str:
    """Résumé structuré par direction de ce que perçoit un poursuivant."""
    lines: list[str] = []

    if target_seen is not None:
        dx, dy = target_seen[0] - pos[0], target_seen[1] - pos[1]
        lines.append(f"Cible visible, position relative (dx={dx}, dy={dy}).")
    else:
        lines.append("Cible non visible.")

    cand_cells = [(int(x), int(y)) for x, y in np.argwhere(belief)]
    offsets = [(x - pos[0], y - pos[1]) for x, y in cand_cells if (x, y) != pos]
    counts = {d: 0 for d in _DIRS}
    for dx, dy in offsets:
        counts[_bucket(dx, dy)] += 1

    lines.append("Directions :")
    for d in _DIRS:
        nx, ny = pos[0] + MOVE_DELTAS[d][0], pos[1] + MOVE_DELTAS[d][1]
        passable = bool(g.free[nx, ny])
        lines.append(
            f"- {_LABELS[d]} : {'praticable' if passable else 'mur'} ; "
            f"cases candidates de ce côté : {counts[d]}")

    lines.append(f"Ensemble candidat total : {len(offsets)} case(s).")
    if offsets and len(offsets) <= track_threshold:
        listed = ", ".join(f"(dx={dx}, dy={dy})" for dx, dy in offsets)
        lines.append(f"Cases candidates : {listed}.")

    return "\n".join(lines)
