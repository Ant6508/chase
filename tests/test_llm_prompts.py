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


def test_target_visible_reports_relative_offset():
    g = _open_room()
    belief = np.zeros(g.free.shape, dtype=bool)
    text = build_perception((3, 3), g, belief, target_seen=(5, 2), track_threshold=8)
    assert "Cible visible, position relative (dx=2, dy=-1)." in text


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


def test_candidate_cells_bucketed_by_dominant_direction():
    g = _open_room()
    belief = np.zeros(g.free.shape, dtype=bool)
    belief[3, 1] = True   # au nord de (3,3) : dx=0, dy=-2
    belief[3, 5] = True   # au sud : dx=0, dy=2
    belief[5, 3] = True   # à l'est : dx=2, dy=0
    belief[1, 3] = True   # à l'ouest : dx=-2, dy=0
    text = build_perception((3, 3), g, belief, target_seen=None, track_threshold=8)
    assert "NORD : praticable ; cases candidates de ce côté : 1" in text
    assert "SUD : praticable ; cases candidates de ce côté : 1" in text
    assert "EST : praticable ; cases candidates de ce côté : 1" in text
    assert "OUEST : praticable ; cases candidates de ce côté : 1" in text
    assert "Ensemble candidat total : 4 case(s)." in text


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
