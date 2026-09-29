"""Tests unitaires de la construction du prompt de perception (chase/llm/prompts.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from chase.graph import MazeGraph
from chase.llm.message import validate
from chase.llm.prompts import (CHANNEL_PARAGRAPH, MESSAGE_HEADER, NO_MESSAGE, PLACES_PARAGRAPH,
                               SYSTEM_PROMPT, build_perception, message_block, system_prompt)
from chase.llm.places import Places


def _open_room() -> MazeGraph:
    """Pièce ouverte 5x5 (indices 1..5) entourée de murs, sur une grille 7x7."""
    free = np.zeros((7, 7), dtype=bool)
    free[1:6, 1:6] = True
    return MazeGraph(free)


def _u_corridor() -> MazeGraph:
    """Couloir en U sur une grille 7x7 : de (1,1) au sud jusqu'à (1,5), à l'est
    jusqu'à (5,5), puis au nord jusqu'à (5,1). (1,1) et (5,1) sont à 4 cases
    à vol d'oiseau, mais à 12 pas par le couloir."""
    free = np.zeros((7, 7), dtype=bool)
    free[1, 1:6] = True
    free[1:6, 5] = True
    free[5, 1:6] = True
    return MazeGraph(free)


def _belief(g: MazeGraph, *cells) -> np.ndarray:
    belief = np.zeros(g.free.shape, dtype=bool)
    for c in cells:
        belief[c] = True
    return belief


def _prob(g: MazeGraph, masses: dict) -> np.ndarray:
    prob = np.zeros(g.free.shape)
    for c, m in masses.items():
        prob[c] = m
    return prob


def _perceive(pos, g, belief, target_seen=None, track_threshold=8, prob=None) -> str:
    """Par défaut, probabilité uniforme sur l'ensemble candidat."""
    if prob is None:
        prob = belief / belief.sum() if belief.any() else np.zeros(belief.shape)
    return build_perception(pos, g, belief, prob, target_seen, track_threshold)


def test_visible_target_is_located_in_cardinal_words_never_in_dx_dy():
    g = _open_room()
    text = _perceive((3, 3), g, _belief(g, (5, 2)), target_seen=(5, 2))
    assert "Cible visible, à 3 pas par les couloirs" in text
    assert "à vol d'oiseau : 1 case au nord et 2 cases à l'est" in text
    assert "dx" not in text and "dy" not in text


def test_target_north_is_called_north():
    """Régression : avec (dx=0, dy=-2), le modèle lisait « au sud » (convention
    mathématique) alors que NORTH vaut dy=-1 dans le code."""
    g = _open_room()
    text = _perceive((3, 3), g, _belief(g, (3, 1)), target_seen=(3, 1))
    assert "à vol d'oiseau : 2 cases au nord)" in text


def test_target_not_visible_reports_so():
    g = _open_room()
    text = _perceive((3, 3), g, _belief(g))
    assert "Cible non visible." in text


def test_directions_report_passable_and_walls():
    g = _open_room()
    # (1, 3) est collé au mur ouest de la pièce : OUEST doit être un mur,
    # les trois autres directions praticables.
    text = _perceive((1, 3), g, _belief(g))
    assert "OUEST : mur" in text
    assert "EST : praticable" in text
    assert "NORD : praticable" in text
    assert "SUD : praticable" in text


def test_candidates_are_counted_by_the_exit_that_starts_the_shortest_path():
    g = _u_corridor()
    # (1,1) au nord à 2 pas ; (5,1) « à l'est » à vol d'oiseau, mais à 10 pas en partant au sud
    text = _perceive((1, 3), g, _belief(g, (1, 1), (5, 1)))
    assert ("- NORD : praticable ; 1 case candidate au plus court par là "
            "(50 % de la probabilité), la plus proche à 2 pas") in text
    assert ("- SUD : praticable ; 1 case candidate au plus court par là "
            "(50 % de la probabilité), la plus proche à 10 pas") in text
    assert "- EST : mur" in text
    assert "- OUEST : mur" in text
    assert "Ensemble candidat total : 2 case(s)." in text


def test_each_exit_reports_the_probability_mass_it_leads_to():
    """Une case qu'on vient de vider redevient candidate aussitôt, mais presque
    sans masse : c'est ce qui la distingue d'une vraie piste."""
    g = _u_corridor()
    belief = _belief(g, (1, 1), (5, 1))
    text = _perceive((1, 3), g, belief, prob=_prob(g, {(1, 1): 0.25, (5, 1): 0.75}))
    assert "1 case candidate au plus court par là (25 % de la probabilité)" in text
    assert "1 case candidate au plus court par là (75 % de la probabilité)" in text


def test_tiny_probability_is_not_rounded_to_zero():
    g = _u_corridor()
    belief = _belief(g, (1, 1), (5, 1))
    text = _perceive((1, 3), g, belief, prob=_prob(g, {(1, 1): 0.002, (5, 1): 0.998}))
    assert "(moins de 1 % de la probabilité)" in text
    assert "(100 % de la probabilité)" in text


def test_exit_without_candidates_says_so():
    g = _u_corridor()
    text = _perceive((1, 3), g, _belief(g, (1, 1)))
    assert "- SUD : praticable ; aucune case candidate au plus court par là" in text


def test_candidate_equidistant_by_two_exits_counts_for_both_and_splits_its_probability():
    g = _open_room()
    # (5,5) est à 4 pas de (3,3), aussi bien en partant au sud qu'à l'est
    text = _perceive((3, 3), g, _belief(g, (5, 5)))
    assert ("- SUD : praticable ; 1 case candidate au plus court par là "
            "(50 % de la probabilité), la plus proche à 4 pas") in text
    assert ("- EST : praticable ; 1 case candidate au plus court par là "
            "(50 % de la probabilité), la plus proche à 4 pas") in text
    assert "- NORD : praticable ; aucune case candidate au plus court par là" in text


def test_candidate_list_gives_path_distance_first_exits_and_probability_nearest_first():
    g = _u_corridor()
    belief = _belief(g, (5, 1), (1, 1))
    text = _perceive((1, 3), g, belief, prob=_prob(g, {(1, 1): 0.25, (5, 1): 0.75}))
    assert "Cases candidates : à 2 pas par NORD (25 %) ; à 10 pas par SUD (75 %)." in text


def test_candidate_list_shown_only_under_threshold():
    g = _open_room()
    text_below = _perceive((3, 3), g, _belief(g, (3, 1), (3, 5)), track_threshold=2)
    assert "Cases candidates :" in text_below

    text_above = _perceive((3, 3), g, _belief(g, (3, 1), (3, 5), (5, 3)), track_threshold=2)
    assert "Cases candidates :" not in text_above


def _t_shape() -> MazeGraph:
    """Couloir est-ouest de (1,1) à (5,1) et branche sud de (3,2) à (3,5) :
    K1 = (3,1) ; C1 = (2,1), (1,1) ; C2 = (4,1), (5,1) ; C3 = (3,2) … (3,5)."""
    free = np.zeros((9, 9), dtype=bool)
    free[1:6, 1] = True
    free[3, 2:6] = True
    return MazeGraph(free)


def _ring() -> MazeGraph:
    """Anneau de (2,2) à (4,4) autour d'un mur, avec un cul-de-sac au nord en (3,1)
    et un au sud en (3,5). K1 = (3,2), K2 = (3,4), à 4 pas l'un de l'autre par l'ouest
    comme par l'est ; C1 = (3,1), C2 = arc ouest, C3 = arc est, C4 = (3,5)."""
    free = np.zeros((9, 9), dtype=bool)
    for c in [(3, 1), (2, 2), (3, 2), (4, 2), (2, 3), (4, 3), (2, 4), (3, 4), (4, 4), (3, 5)]:
        free[c] = True
    return MazeGraph(free)


def _perceive_places(pos, g, belief, target_seen=None, track_threshold=8) -> str:
    prob = belief / belief.sum() if belief.any() else np.zeros(belief.shape)
    return build_perception(pos, g, belief, prob, target_seen, track_threshold,
                            Places.from_graph(g))


def test_places_name_the_position_the_exits_and_the_candidates():
    g = _t_shape()
    text = _perceive_places((3, 3), g, _belief(g, (1, 1), (3, 5)))
    assert text.splitlines() == [
        "Tu es en C3.2 (ton lieu C3 contient des candidates : 50 %).",
        "Cible non visible.",
        "Directions :",
        "- NORD : praticable ; 1 case candidate au plus court par là "
        "(50 % de la probabilité), la plus proche à 4 pas",
        "  lieux par là : K1 à 2, C1 à 3 (50 %), C2 à 3",
        "- SUD : praticable ; 1 case candidate au plus court par là "
        "(50 % de la probabilité), la plus proche à 2 pas",
        "  lieux par là : aucun",
        "- EST : mur",
        "- OUEST : mur",
        "Ensemble candidat total : 2 case(s).",
        "Cases candidates : C3.4 à 2 pas par SUD (50 %) ; C1.2 à 4 pas par NORD (50 %).",
    ]


def test_visible_target_is_named_and_a_junction_has_no_candidate_note():
    g = _t_shape()
    text = _perceive_places((3, 1), g, _belief(g, (1, 1)), target_seen=(1, 1))
    assert text.splitlines()[:2] == [
        "Tu es en K1.",
        "Cible visible en C1.2, à 2 pas par les couloirs (à vol d'oiseau : 2 cases à l'ouest).",
    ]


def test_a_place_as_close_by_two_exits_is_listed_under_both():
    g = _ring()
    text = _perceive_places((3, 2), g, _belief(g))
    lines = text.splitlines()
    north = lines.index("- NORD : praticable ; aucune case candidate au plus court par là")
    assert lines[north + 1] == "  lieux par là : C1 à 1"
    assert "  lieux par là : C3 à 1, K2 à 4, C4 à 5" in lines   # sous EST
    assert "  lieux par là : C2 à 1, K2 à 4, C4 à 5" in lines   # sous OUEST
    assert text.count("K2 à 4") == 2


def test_a_place_percentage_is_not_split_between_the_two_exits_that_reach_it():
    """Deux pourcentages coexistent et ne se calculent pas pareil. Celui de la ligne
    d'issue est une part : une case candidate à égalité entre EST et OUEST partage sa
    probabilité entre les deux (50 % chacun). Celui d'un lieu, dans « lieux par là »,
    est la probabilité que la cible soit dans ce lieu : elle ne se partage pas, et
    figure entière (100 %) sous chacune des deux issues qui l'atteignent au plus court."""
    g = _ring()
    text = _perceive_places((3, 2), g, _belief(g, (3, 4)))  # une seule candidate : K2
    lines = text.splitlines()
    assert ("- EST : praticable ; 1 case candidate au plus court par là "
            "(50 % de la probabilité), la plus proche à 4 pas") in lines
    assert ("- OUEST : praticable ; 1 case candidate au plus court par là "
            "(50 % de la probabilité), la plus proche à 4 pas") in lines
    assert text.count("K2 à 4 (100 %)") == 2


def test_without_places_the_perception_is_the_a1v3_one():
    g = _t_shape()
    belief = _belief(g, (1, 1), (3, 5))
    with_places = _perceive_places((3, 3), g, belief, track_threshold=1)
    without = _perceive((3, 3), g, belief, track_threshold=1)
    stripped = [line for line in with_places.splitlines()
                if not line.startswith("Tu es en") and not line.startswith("  lieux par là")]
    assert stripped == without.splitlines()
    assert "Tu es en" not in without and "lieux par là" not in without


def test_a1_system_prompt_is_the_a1v3_one():
    assert system_prompt("A1") == SYSTEM_PROMPT


def test_a1bis_adds_the_places_and_keeps_the_teammate_out_of_reach():
    text = system_prompt("A1bis")
    assert PLACES_PARAGRAPH in text
    assert "que tu ne peux PAS contacter" in text
    assert "message" not in text
    assert text.endswith("avec la direction choisie et une justification brève.")


def test_a2_adds_places_and_channel_and_asks_for_the_message():
    text = system_prompt("A2")
    assert PLACES_PARAGRAPH in text and CHANNEL_PARAGRAPH in text
    assert "ne peux PAS contacter" not in text
    assert "avec qui tu échanges un message à chaque pas" in text
    assert text.endswith("une justification brève et ton message.")


def test_channel_example_is_a_valid_message():
    example = json.loads(CHANNEL_PARAGRAPH.split("Exemple : ", 1)[1])
    assert validate(example) is None


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="bras inconnu"):
        system_prompt("A3")


def test_message_block_shows_the_message_or_its_absence():
    assert message_block('{"moi":"K1"}') == f'{MESSAGE_HEADER}\n{{"moi":"K1"}}'
    assert message_block(None) == f"{MESSAGE_HEADER}\n{NO_MESSAGE}"
