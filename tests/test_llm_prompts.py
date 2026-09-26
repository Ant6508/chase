"""Tests unitaires de la construction du prompt de perception (chase/llm/prompts.py)."""

from __future__ import annotations

import numpy as np

from chase.graph import MazeGraph
from chase.llm.prompts import build_perception


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
