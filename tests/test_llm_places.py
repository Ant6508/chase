"""Tests du référentiel de lieux (chase/llm/places.py)."""

from __future__ import annotations

import string

import numpy as np
import pytest

from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.graph import MazeGraph
from chase.llm.places import MAX_SEGMENT, Places

PROFILE = ChaseConfig().replace(max_steps=60, size=15, n_loops=1, min_loop_len=6,
                                min_spawn_dist=6)


def _graph(seed: int, cfg: ChaseConfig = PROFILE) -> MazeGraph:
    env = ChaseEnv(cfg)
    env.reset(seed=seed)
    return env.graph


def _t_shape(south_len: int = 4) -> MazeGraph:
    """Couloir est-ouest de (1,1) à (5,1) et branche sud depuis (3,1) : un carrefour
    en (3,1), trois culs-de-sac (ouest, est, sud)."""
    size = max(9, south_len + 4)
    free = np.zeros((size, size), dtype=bool)
    free[1:6, 1] = True
    free[3, 2:2 + south_len] = True
    return MazeGraph(free)


def _names(places: Places, g: MazeGraph, cells) -> list[str]:
    return [places.cell_name[g.index[c]] for c in cells]


def _corridors(places: Places) -> dict[str, list]:
    """Tronçons regroupés par couloir, dans l'ordre : « C7a », « C7b » -> « C7 »."""
    out: dict[str, list] = {}
    for p in places.places:
        if p.name.startswith("C"):
            out.setdefault(p.name.rstrip(string.ascii_lowercase), []).append(p)
    return out


def test_t_shape_names_follow_reading_order_and_start_at_the_junction():
    g = _t_shape()
    places = Places.from_graph(g)
    assert [p.name for p in places.places] == ["K1", "C1", "C2", "C3"]
    assert _names(places, g, [(3, 1)]) == ["K1"]
    # chaque couloir part du carrefour et finit au cul-de-sac
    assert _names(places, g, [(2, 1), (1, 1)]) == ["C1.1", "C1.2"]
    assert _names(places, g, [(4, 1), (5, 1)]) == ["C2.1", "C2.2"]
    assert _names(places, g, [(3, 2), (3, 3), (3, 4), (3, 5)]) == ["C3.1", "C3.2", "C3.3", "C3.4"]


def test_long_corridor_is_cut_into_balanced_segments():
    g = _t_shape(south_len=7)
    places = Places.from_graph(g)
    assert [p.name for p in places.places] == ["K1", "C1", "C2", "C3a", "C3b"]
    assert [len(places.by_name[n].cells) for n in ("C3a", "C3b")] == [4, 3]
    assert _names(places, g, [(3, 5), (3, 6), (3, 8)]) == ["C3a.4", "C3b.1", "C3b.3"]


def test_valid_names_cover_places_and_cells():
    g = _t_shape(south_len=7)
    places = Places.from_graph(g)
    assert {"K1", "C1", "C3a", "C3b.2"} <= places.names
    assert "C3" not in places.names  # un couloir coupé n'a plus de nom à lui
    assert places.cell_by_name["C3b.1"] == g.index[3, 6]


@pytest.mark.parametrize("seed", range(10))
def test_every_free_cell_has_exactly_one_unique_name(seed):
    g = _graph(seed)
    places = Places.from_graph(g)
    assert all(places.cell_name)
    assert len(set(places.cell_name)) == g.n
    assert sorted(c for p in places.places for c in p.cells) == list(range(g.n))


@pytest.mark.parametrize("seed", range(10))
def test_places_are_junctions_or_short_chains(seed):
    g = _graph(seed)
    places = Places.from_graph(g)
    for p in places.places:
        if p.name.startswith("K"):
            assert len(p.cells) == 1 and len(g.neighbors[p.cells[0]]) >= 3
        else:
            assert 1 <= len(p.cells) <= MAX_SEGMENT
            assert all(len(g.neighbors[c]) <= 2 for c in p.cells)
            assert all(g.dist[a, b] == 1 for a, b in zip(p.cells, p.cells[1:]))


@pytest.mark.parametrize("seed", range(10))
def test_segments_follow_each_other_from_the_smallest_junction(seed):
    g = _graph(seed)
    places = Places.from_graph(g)
    junction = {p.cells[0]: int(p.name[1:]) for p in places.places if p.name.startswith("K")}

    def touching(c):
        return [junction[v] for v in g.neighbors[c] if v in junction]

    for base, segments in _corridors(places).items():
        sizes = [len(s.cells) for s in segments]
        assert max(sizes) - min(sizes) <= 1 and sizes == sorted(sizes, reverse=True)
        if len(segments) > 1:
            assert [s.name for s in segments] == [base + string.ascii_lowercase[q]
                                                  for q in range(len(segments))]
        chain = [c for s in segments for c in s.cells]
        assert all(g.dist[a, b] == 1 for a, b in zip(chain, chain[1:]))
        ends = touching(chain[0]) + touching(chain[-1])
        if ends:
            assert touching(chain[0]) and min(touching(chain[0])) == min(ends)


@pytest.mark.parametrize("seed", range(10))
def test_numbers_follow_reading_order(seed):
    g = _graph(seed)
    places = Places.from_graph(g)

    def reading(c):
        return g.cells[c][1], g.cells[c][0]

    ks = [p.cells[0] for p in places.places if p.name.startswith("K")]
    assert ks == sorted(ks, key=reading)
    firsts = [min((c for s in segs for c in s.cells), key=reading)
              for segs in _corridors(places).values()]
    assert firsts == sorted(firsts, key=reading)


def test_names_depend_only_on_the_seed():
    a, b = Places.from_graph(_graph(4)), Places.from_graph(_graph(4))
    assert [(p.name, p.cells) for p in a.places] == [(p.name, p.cells) for p in b.places]


def test_default_profile_maps_are_named_too():
    """Les tests de la politique jouent sur la carte par défaut (27 x 27), où le plus
    long couloir mesuré sur 60 seeds fait 68 cases, soit 14 tronçons."""
    for seed in range(3):
        g = _graph(seed, ChaseConfig())
        assert len(set(Places.from_graph(g).cell_name)) == g.n
