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

# Bras A1bis et A2 (docs/superpowers/specs/2026-09-29-jalon2-a2-design.md). SYSTEM_PROMPT
# ci-dessus reste celui de la campagne A1v3, inchangé.
ARMS = ("A1", "A1bis", "A2")

_INTRO_A1 = ("en coopération avec un coéquipier que tu ne peux PAS contacter : tu ne "
             "connais ni sa position ni ses intentions.")
_INTRO_A2 = "en coopération avec un coéquipier avec qui tu échanges un message à chaque pas."
_ANSWER_A1 = ("Réponds uniquement en appelant l'outil `move` avec la direction choisie et une "
              "justification brève.")
_ANSWER_A2 = ("Réponds uniquement en appelant l'outil `move` avec la direction choisie, une "
              "justification brève et ton message.")

PLACES_PARAGRAPH = """Les lieux du labyrinthe portent des noms, les mêmes pour toi et \
pour ton coéquipier. Un carrefour (case à au moins trois issues) s'appelle K1, K2…, et un \
couloir entre deux carrefours, ou entre un carrefour et un cul-de-sac, s'appelle C1, C2… \
Un couloir de plus de 5 cases est coupé en tronçons C7a, C7b… Une case précise se désigne \
par son lieu et son rang, compté depuis le bout du couloir qui touche le carrefour de plus \
petit numéro (un cul-de-sac en dernier) : C6b.3 est la 3e case du tronçon C6b. Une case de \
carrefour porte le nom du carrefour. Sous chaque direction praticable, la perception liste \
les lieux que tu atteins au plus court en partant par là, avec leur distance en pas ; un \
pourcentage donne la probabilité que la cible y soit, et un lieu sans pourcentage ne \
contient aucune case candidate. Un lieu aussi proche par deux directions figure sous les \
deux avec sa probabilité entière : contrairement à la part d'une direction, elle ne se \
partage pas."""

CHANNEL_PARAGRAPH = """À chaque pas, tu écris un message à ton coéquipier dans le champ \
`message` de l'outil `move`. Il le lira au pas suivant ; de même, le message qu'il t'a \
écrit au pas précédent figure à la fin de ta perception. Ta pensée et ta justification \
restent privées : seul le message lui parvient. Le message a cinq champs, tous \
obligatoires :
- `moi` : ta case, pour vous répartir la recherche et préparer une prise en tenaille ;
- `cible` : la case de la cible si tu la vois, sinon null ;
- `candidates` : les lieux où la cible peut être d'après ta perception, avec leur \
probabilité en pourcentage entier ;
- `intention` : les prochains lieux que tu comptes traverser, dans l'ordre ;
- `je_couvre` : le lieu que tu bloques ou gardes, sinon null.
Un lieu absent des `candidates` d'un message a été vu vide par son auteur au pas \
précédent : la cible n'y est presque sûrement pas.
Exemple : {"moi":"C2a.3","cible":null,"candidates":{"C7a":7,"C9":6,"K3":2},\
"intention":["K1","C4a"],"je_couvre":null}"""

MESSAGE_HEADER = "Message de ton coéquipier (écrit au pas précédent) :"
NO_MESSAGE = "Aucun message reçu."


def system_prompt(arm: str) -> str:
    """A1 : le prompt de la campagne A1v3. A1bis : le même, plus les lieux. A2 : les
    lieux et le canal de message, avec la phrase sur le coéquipier remplacée."""
    if arm not in ARMS:
        raise ValueError(f"bras inconnu : {arm}")
    if arm == "A1":
        return SYSTEM_PROMPT
    if _INTRO_A1 not in SYSTEM_PROMPT or not SYSTEM_PROMPT.endswith(_ANSWER_A1):
        raise AssertionError("SYSTEM_PROMPT a changé : revoir system_prompt()")
    body = SYSTEM_PROMPT.removesuffix(_ANSWER_A1) + PLACES_PARAGRAPH + "\n\n"
    if arm == "A1bis":
        return body + _ANSWER_A1
    return body.replace(_INTRO_A1, _INTRO_A2) + CHANNEL_PARAGRAPH + "\n\n" + _ANSWER_A2


def message_block(rendered: str | None) -> str:
    """Fin du prompt utilisateur en A2 : l'en-tête fixe, puis le message reçu tel que
    réécrit par `message.render`, ou son absence (premier pas, repli du coéquipier).
    C'est l'emplacement où A4 injectera ses vecteurs."""
    return f"{MESSAGE_HEADER}\n{NO_MESSAGE if rendered is None else rendered}"


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
