"""Prompt système et résumé de perception par direction, envoyés au LLM à
chaque pas. Pas de grille ASCII : pour un modèle de 12B, un résumé structuré
par direction évite le piège classique de confusion lignes/colonnes ou
nord/sud sur une grille texte.

Deux choix viennent de la campagne A1 du 2026-09-25, restée au niveau du
hasard (results/jalon2_a1.md) :

- aucune position en (dx, dy). Le modèle lisait dy < 0 comme « au sud »
  (convention mathématique), alors que NORTH vaut dy = -1 ici : cible au nord,
  il partait au sud. Tout est dit en mots cardinaux, le vocabulaire des actions ;
- les cases candidates sont comptées par issue, en distance de chemin, et non
  plus par direction à vol d'oiseau. Dans un labyrinthe fait de couloirs, la
  direction la plus chargée à vol d'oiseau était un mur une fois sur deux.
"""

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

Les distances sont comptées en pas le long des couloirs, pas à vol d'oiseau. \
Pour chaque direction praticable, la perception indique combien de cases \
candidates tu atteins au plus court en partant par là, et à combien de pas se \
trouve la plus proche. Une case aussi proche par deux directions compte pour \
les deux.

Réponds uniquement en appelant l'outil `move` avec la direction choisie et une \
justification brève."""

_DIRS = (Move.NORTH, Move.SOUTH, Move.EAST, Move.WEST)
_LABELS = {Move.NORTH: "NORD", Move.SOUTH: "SUD", Move.EAST: "EST", Move.WEST: "OUEST"}


def _cases(n: int) -> str:
    return f"{n} case{'s' if n > 1 else ''}"


def _cardinal(dx: int, dy: int) -> str:
    """Décalage en mots cardinaux. Sens de MOVE_DELTAS : dy < 0 au nord, dx > 0 à l'est."""
    parts = []
    if dy:
        parts.append(f"{_cases(abs(dy))} {'au nord' if dy < 0 else 'au sud'}")
    if dx:
        side = "à l'est" if dx > 0 else "à l'ouest"
        parts.append(f"{_cases(abs(dx))} {side}")
    return " et ".join(parts)


def build_perception(pos: tuple[int, int], g: MazeGraph, belief: np.ndarray,
                      target_seen: tuple[int, int] | None,
                      track_threshold: int) -> str:
    """Résumé structuré par direction de ce que perçoit un poursuivant."""
    here = g.index[pos]
    exits = {}  # direction praticable -> case voisine
    for d in _DIRS:
        nxt = (pos[0] + MOVE_DELTAS[d][0], pos[1] + MOVE_DELTAS[d][1])
        if g.free[nxt]:
            exits[d] = g.index[nxt]

    cand = np.array([g.index[tuple(map(int, c))] for c in np.argwhere(belief)
                     if tuple(map(int, c)) != pos], dtype=np.int64)
    dist = g.dist[here, cand]
    # une direction « mène au plus court » à une case si sa voisine en est plus proche d'un pas
    first = {d: g.dist[n, cand] == dist - 1 for d, n in exits.items()}

    lines: list[str] = []
    if target_seen is not None:
        steps = int(g.dist[here, g.index[target_seen]])
        offset = _cardinal(target_seen[0] - pos[0], target_seen[1] - pos[1])
        lines.append(f"Cible visible, à {steps} pas par les couloirs (à vol d'oiseau : {offset}).")
    else:
        lines.append("Cible non visible.")

    lines.append("Directions :")
    for d in _DIRS:
        if d not in exits:
            lines.append(f"- {_LABELS[d]} : mur")
        elif first[d].any():
            n = int(first[d].sum())
            what = "1 case candidate" if n == 1 else f"{n} cases candidates"
            lines.append(f"- {_LABELS[d]} : praticable ; {what} au plus court par là, "
                         f"la plus proche à {int(dist[first[d]].min())} pas")
        else:
            lines.append(f"- {_LABELS[d]} : praticable ; aucune case candidate au plus court par là")

    lines.append(f"Ensemble candidat total : {len(cand)} case(s).")
    if len(cand) and len(cand) <= track_threshold:
        listed = " ; ".join(
            f"à {int(dist[k])} pas par {', '.join(_LABELS[d] for d in exits if first[d][k])}"
            for k in np.argsort(dist, kind="stable"))
        lines.append(f"Cases candidates : {listed}.")

    return "\n".join(lines)
