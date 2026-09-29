"""Tests du plafond mécanique du protocole A2 (chase/llm/ceiling.py), sans LLM."""

from __future__ import annotations

import numpy as np
import pytest

from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.llm.ceiling import ProtocolPursuers, write_message
from chase.llm.message import unknown_names, validate
from chase.llm.places import Places
from chase.policies import GreedyPursuers, Percept
from chase.runner import run_episode

PROFILE = ChaseConfig().replace(max_steps=60, size=15, n_loops=1, min_loop_len=6,
                                min_spawn_dist=6)


def _map(seed: int):
    env = ChaseEnv(PROFILE)
    env.reset(seed=seed)
    return env, env.graph, Places.from_graph(env.graph)


@pytest.mark.parametrize("seed", range(5))
def test_beliefs_always_contain_the_target_and_messages_only_shrink_them(seed):
    def check(env, policy):
        if env.captured:
            return
        t = env.target_pos
        for own, decided in zip(policy._own, policy.beliefs()):
            assert own[t] and decided[t]
            assert not (decided & ~own).any()

    run_episode(PROFILE, ProtocolPursuers(PROFILE), seed, on_step=check)


def test_messages_are_valid_and_use_only_names_of_the_map():
    seen = []

    def check(env, policy):
        for m in policy.sent:
            if m is not None:
                seen.append(m)
                assert validate(m) is None
                assert unknown_names(m, policy.places) == []

    run_episode(PROFILE, ProtocolPursuers(PROFILE), 2, on_step=check)
    assert seen


def test_message_lists_exactly_the_places_holding_candidates():
    _, g, places = _map(1)
    last = places.places[-1]
    belief = np.zeros(g.free.shape, dtype=bool)
    belief[g.cells[last.cells[0]]] = True
    prob = belief / belief.sum()
    first = places.places[0]  # K1
    m = write_message(g, places, g.cells[first.cells[0]], belief, prob, None, [])
    assert m == {"moi": first.name, "cible": None, "candidates": {last.name: 100},
                 "intention": [], "je_couvre": None}


def test_message_names_the_target_cell_when_seen():
    _, g, places = _map(1)
    cell = places.places[-1].cells[0]
    belief = np.zeros(g.free.shape, dtype=bool)
    belief[g.cells[cell]] = True
    m = write_message(g, places, g.cells[places.places[0].cells[0]], belief, belief / 1.0,
                      g.cells[cell], [])
    assert m["cible"] == places.cell_name[cell]


def test_intention_names_the_places_along_the_path_without_repeats():
    _, g, places = _map(1)
    corridor = next(p for p in places.places if len(p.cells) >= 3)
    empty = np.zeros(g.free.shape, dtype=bool)
    m = write_message(g, places, g.cells[corridor.cells[0]], empty, np.zeros(g.free.shape), None,
                      list(corridor.cells[:3]))
    assert m["intention"] == [corridor.name]


def test_without_message_p2_decides_like_r1():
    env, _, _ = _map(4)
    vis = env.visibility()
    percepts = [Percept(env.pursuer_pos(p), vis[p], env.target_seen_by(vis[p]))
                for p in range(PROFILE.n_pursuers)]
    r1 = GreedyPursuers(PROFILE, fused=False)
    r1.reset(env.graph, np.random.default_rng(0))
    r1.update(percepts)
    p2 = ProtocolPursuers(PROFILE)
    p2.reset(env.graph, np.random.default_rng(0))
    p2.update(percepts)
    assert p2.act(percepts) == r1.act(percepts)


def test_p2_requires_two_pursuers():
    with pytest.raises(ValueError, match="2 poursuivants"):
        ProtocolPursuers(PROFILE.replace(n_pursuers=3))
