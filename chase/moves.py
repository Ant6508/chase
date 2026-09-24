"""Espace d'action commun à tous les agents (poursuivants et cible).

Déplacements absolus nommés : l'agent s'oriente dans la direction demandée
puis avance d'une case si elle est libre. Contre un mur, il se contente de
s'orienter, ce qui lui permet de regarder sans bouger. Le champ de vision
MiniGrid étant orienté, l'orientation fait partie de la décision.
"""

from __future__ import annotations

from enum import IntEnum


class Move(IntEnum):
    STAY = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


# (dx, dy) en coordonnées de grille ; y croît vers le sud.
MOVE_DELTAS = {
    Move.STAY: (0, 0),
    Move.NORTH: (0, -1),
    Move.SOUTH: (0, 1),
    Move.EAST: (1, 0),
    Move.WEST: (-1, 0),
}

# Direction MultiGrid (0 droite, 1 bas, 2 gauche, 3 haut).
MOVE_TO_DIR = {Move.EAST: 0, Move.SOUTH: 1, Move.WEST: 2, Move.NORTH: 3}
