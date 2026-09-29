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

La deuxième campagne (results/jalon2_a1v2.md) a ajouté la part de probabilité
par issue. Sans elle, la case qu'on vient de quitter, sortie du champ de vision
orienté, redevenait « la candidate la plus proche, à 1 pas », et le poursuivant
faisait l'aller-retour une fois sur deux. La carte est celle de R1
(belief.diffuse / observe_prob) : le harnais fait le calcul probabiliste, le
LLM décide.
"""

from __future__ import annotations

import numpy as np

from ..graph import MazeGraph
from ..moves import MOVE_DELTAS, Move
from .places import Places

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

Chaque direction indique aussi la part de probabilité que la cible soit de ce \
côté, en supposant qu'elle se déplace au hasard depuis tes dernières \
observations. Une case que tu viens de voir vide redevient candidate dès que \
tu ne la vois plus, mais avec une probabilité presque nulle. Une case aussi \
proche par deux directions partage sa probabilité entre elles.

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


def _pct(p: float) -> str:
    v = round(100 * p)
    return "moins de 1 %" if v == 0 and p > 0 else f"{v} %"


def build_perception(pos: tuple[int, int], g: MazeGraph, belief: np.ndarray,
                      prob: np.ndarray, target_seen: tuple[int, int] | None,
                      track_threshold: int, places: Places | None = None) -> str:
    """Résumé structuré par direction de ce que perçoit un poursuivant.
    `prob` est la carte de probabilité posée sur l'ensemble candidat `belief`.

    Avec `places` (bras A1bis et A2), la perception nomme la position, la case de la
    cible, les lieux desservis par chaque issue et les cases candidates. Sans
    `places`, elle est identique au caractère près au format de la campagne A1v3."""
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
    mass = prob[g.xy[cand, 0], g.xy[cand, 1]]
    ties = np.maximum(sum(first.values(), np.zeros(len(cand))), 1)  # issues à égalité par case

    lines: list[str] = []
    if places is not None:
        lines.append(_position_line(here, g, belief, prob, places))
    if target_seen is not None:
        steps = int(g.dist[here, g.index[target_seen]])
        offset = _cardinal(target_seen[0] - pos[0], target_seen[1] - pos[1])
        where = f" en {places.cell_name[g.index[target_seen]]}" if places is not None else ""
        lines.append(f"Cible visible{where}, à {steps} pas par les couloirs "
                     f"(à vol d'oiseau : {offset}).")
    else:
        lines.append("Cible non visible.")

    served = _places_by_exit(here, exits, g, belief, prob, places) if places is not None else {}
    lines.append("Directions :")
    for d in _DIRS:
        if d not in exits:
            lines.append(f"- {_LABELS[d]} : mur")
        elif first[d].any():
            n = int(first[d].sum())
            what = "1 case candidate" if n == 1 else f"{n} cases candidates"
            share = float((mass / ties)[first[d]].sum())
            lines.append(f"- {_LABELS[d]} : praticable ; {what} au plus court par là "
                         f"({_pct(share)} de la probabilité), "
                         f"la plus proche à {int(dist[first[d]].min())} pas")
        else:
            lines.append(f"- {_LABELS[d]} : praticable ; aucune case candidate au plus court par là")
        if d in served:
            lines.append(f"  lieux par là : {served[d]}")

    lines.append(f"Ensemble candidat total : {len(cand)} case(s).")
    if len(cand) and len(cand) <= track_threshold:
        listed = " ; ".join(
            f"{places.cell_name[cand[k]] + ' ' if places is not None else ''}"
            f"à {int(dist[k])} pas par {', '.join(_LABELS[d] for d in exits if first[d][k])}"
            f" ({_pct(float(mass[k]))})"
            for k in np.argsort(dist, kind="stable"))
        lines.append(f"Cases candidates : {listed}.")

    return "\n".join(lines)


def _position_line(here: int, g: MazeGraph, belief: np.ndarray, prob: np.ndarray,
                   places: Places) -> str:
    """« Tu es en C2a.3. », avec la part de probabilité des autres cases candidates de
    son lieu s'il en reste (sa propre case, toujours vue, n'est jamais candidate)."""
    place = places.places[places.place_of[here]]
    others = np.array([c for c in place.cells if c != here], dtype=np.int64)
    xs, ys = g.xy[others, 0], g.xy[others, 1]
    if len(others) and belief[xs, ys].any():
        share = float(prob[xs, ys].sum())
        return (f"Tu es en {places.cell_name[here]} "
                f"(ton lieu {place.name} contient des candidates : {_pct(share)}).")
    return f"Tu es en {places.cell_name[here]}."


def _places_by_exit(here: int, exits: dict, g: MazeGraph, belief: np.ndarray,
                    prob: np.ndarray, places: Places) -> dict[Move, str]:
    """Pour chaque issue praticable, les lieux dont la case la plus proche s'atteint au
    plus court par là, triés par distance puis dans l'ordre de la liste des lieux
    (K1…, puis C1…). Un lieu aussi proche par deux issues figure sous les deux. Le
    pourcentage n'est donné que pour un lieu qui contient une case candidate."""
    served: dict[Move, list] = {d: [] for d in exits}
    mine = places.place_of[here]
    for k, place in enumerate(places.places):
        if k == mine:
            continue
        cells = np.array(place.cells, dtype=np.int64)
        d = g.dist[here, cells]
        dmin = int(d.min())
        near = cells[d == dmin]
        xs, ys = g.xy[cells, 0], g.xy[cells, 1]
        label = f"{place.name} à {dmin}"
        if belief[xs, ys].any():
            label += f" ({_pct(float(prob[xs, ys].sum()))})"
        for move, n in exits.items():
            if (g.dist[n, near] == dmin - 1).any():
                served[move].append((dmin, k, label))
    return {move: ", ".join(label for _, _, label in sorted(items)) or "aucun"
            for move, items in served.items()}
