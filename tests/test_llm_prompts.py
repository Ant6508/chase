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


def test_visible_target_is_located_in_cardinal_words_never_in_dx_dy():
    g = _open_room()
    text = build_perception((3, 3), g, _belief(g, (5, 2)), target_seen=(5, 2), track_threshold=8)
    assert "Cible visible, à 3 pas par les couloirs" in text
    assert "à vol d'oiseau : 1 case au nord et 2 cases à l'est" in text
    assert "dx" not in text and "dy" not in text


def test_target_north_is_called_north():
    """Régression : avec (dx=0, dy=-2), le modèle lisait « au sud » (convention
    mathématique) alors que NORTH vaut dy=-1 dans le code."""
    g = _open_room()
    text = build_perception((3, 3), g, _belief(g, (3, 1)), target_seen=(3, 1), track_threshold=8)
    assert "à vol d'oiseau : 2 cases au nord)" in text


def test_target_not_visible_reports_so():
    g = _open_room()
    belief = np.zeros(g.free.shape, dtype=bool)
    text = build_perception((3, 3), g, belief, target_seen=None, track_threshold=8)
    assert "Cible non visible." in text


def test_directions_report_passable_and_walls():
    g = _open_room()
    belief = np.zeros(g.free.shape, dtype=bool)
    # (1, 3) est collé au mur ouest de la pièce : OUEST doit être un mur,
    # les trois autres directions praticables.
    text = build_perception((1, 3), g, belief, target_seen=None, track_threshold=8)
    assert "OUEST : mur" in text
    assert "EST : praticable" in text
    assert "NORD : praticable" in text
    assert "SUD : praticable" in text


def test_candidates_are_counted_by_the_exit_that_starts_the_shortest_path():
    g = _u_corridor()
    # (1,1) au nord à 2 pas ; (5,1) « à l'est » à vol d'oiseau, mais à 10 pas en partant au sud
    text = build_perception((1, 3), g, _belief(g, (1, 1), (5, 1)), target_seen=None,
                            track_threshold=8)
    assert "- NORD : praticable ; 1 case candidate au plus court par là, la plus proche à 2 pas" in text
    assert "- SUD : praticable ; 1 case candidate au plus court par là, la plus proche à 10 pas" in text
    assert "- EST : mur" in text
    assert "- OUEST : mur" in text
    assert "Ensemble candidat total : 2 case(s)." in text


def test_exit_without_candidates_says_so():
    g = _u_corridor()
    text = build_perception((1, 3), g, _belief(g, (1, 1)), target_seen=None, track_threshold=8)
    assert "- SUD : praticable ; aucune case candidate au plus court par là" in text


def test_candidate_equidistant_by_two_exits_counts_for_both():
    g = _open_room()
    # (5,5) est à 4 pas de (3,3), aussi bien en partant au sud qu'à l'est
    text = build_perception((3, 3), g, _belief(g, (5, 5)), target_seen=None, track_threshold=8)
    assert "- SUD : praticable ; 1 case candidate au plus court par là, la plus proche à 4 pas" in text
    assert "- EST : praticable ; 1 case candidate au plus court par là, la plus proche à 4 pas" in text
    assert "- NORD : praticable ; aucune case candidate au plus court par là" in text


def test_candidate_list_gives_path_distance_and_first_exits_nearest_first():
    g = _u_corridor()
    text = build_perception((1, 3), g, _belief(g, (5, 1), (1, 1)), target_seen=None,
                            track_threshold=8)
    assert "Cases candidates : à 2 pas par NORD ; à 10 pas par SUD." in text


def test_candidate_list_shown_only_under_threshold():
    g = _open_room()
    below = np.zeros(g.free.shape, dtype=bool)
    below[3, 1] = True
    below[3, 5] = True
    text_below = build_perception((3, 3), g, below, target_seen=None, track_threshold=2)
    assert "Cases candidates :" in text_below

    above = np.zeros(g.free.shape, dtype=bool)
    above[3, 1] = True
    above[3, 5] = True
    above[5, 3] = True
    text_above = build_perception((3, 3), g, above, target_seen=None, track_threshold=2)
    assert "Cases candidates :" not in text_above
