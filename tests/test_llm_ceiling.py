"""Tests du plafond mécanique du protocole A2 (chase/llm/ceiling.py), sans LLM."""

from __future__ import annotations

import numpy as np
import pytest

from chase.belief import propagate
from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.llm.ceiling import ProtocolPursuers, write_message
from chase.llm.message import A2_SPEC, unknown_names, validate
from chase.llm.places import Places
from chase.moves import Move
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


# ------------------------------------------------------------------ mutation kills


@pytest.mark.parametrize("seed", range(5))
def test_delay_and_exact_formula(seed):
    """`_inbox` est exactement le dernier message écrit (pas de relais, pas
    d'accumulation), et la croyance de décision suit exactement la formule de
    `update` : croyance propre intersectée avec la propagation d'un pas du
    support du message reçu. Sur l'épisode, le message sert vraiment : la
    croyance de décision est au moins une fois strictement plus petite que la
    croyance individuelle."""
    calls = {"n": 0}
    shrunk = {"seen": False}

    def check(env, policy):
        calls["n"] += 1
        if calls["n"] < 2:  # tout premier update, avant le moindre act : rien reçu encore
            return
        free = env.graph.free
        prev_sent = policy.sent  # écrit par l'act qui vient de précéder cet update
        for i in range(PROFILE.n_pursuers):
            assert policy._inbox[i] is prev_sent[1 - i]
            expected = policy._own[i] & propagate(policy._support(prev_sent[1 - i]), free)
            assert (policy.beliefs()[i] == expected).all()
            if policy.beliefs()[i].sum() < policy._own[i].sum():
                shrunk["seen"] = True

    run_episode(PROFILE, ProtocolPursuers(PROFILE), seed, on_step=check)
    assert shrunk["seen"]


def test_decision_loop_never_sees_a_message_written_this_same_act_call():
    """Les décisions de `act()` lisent `self._inbox` tel qu'il était avant l'appel :
    aucune décision ne doit voir un message que le coéquipier vient d'écrire au
    même pas (pas de lecture « au même pas », pas de relais intra-pas). `_inbox`
    ne doit changer qu'une fois toute l'équipe décidée."""
    env, g, places = _map(3)
    policy = ProtocolPursuers(PROFILE)
    policy.reset(env.graph, np.random.default_rng(0))
    vis = env.visibility()
    percepts = [Percept(env.pursuer_pos(p), vis[p], env.target_seen_by(vis[p]))
                for p in range(PROFILE.n_pursuers)]
    policy.update(percepts)
    policy.act(percepts)  # premier pas : établit un message reçu réel pour la suite

    expected_inbox = list(policy._inbox)
    calls = []
    orig = policy._team_positions

    def spy(i, me):
        calls.append(policy._inbox[i] is expected_inbox[i])
        return orig(i, me)

    policy._team_positions = spy
    policy.act(percepts)
    assert calls  # la boucle de décision a bien appelé _team_positions
    assert all(calls)


@pytest.mark.parametrize("seed", range(5))
def test_teammate_position_lags_by_one_step(seed):
    """`_team_positions` ne connaît la case du coéquipier que par le message du
    pas précédent (retard d'un pas) ; sa propre case, elle, est exacte."""
    state = {"prev_pos": None, "n": 0}

    def check(env, policy):
        state["n"] += 1
        cur_pos = [env.pursuer_pos(p) for p in range(PROFILE.n_pursuers)]
        if state["n"] >= 2 and not env.captured:
            g = env.graph
            for i in range(PROFILE.n_pursuers):
                team = policy._team_positions(i, cur_pos[i])
                assert team[i] == int(g.index[cur_pos[i]])
                assert team[1 - i] == int(g.index[state["prev_pos"][1 - i]])
        state["prev_pos"] = cur_pos

    run_episode(PROFILE, ProtocolPursuers(PROFILE), seed, on_step=check)


def test_target_cell_support_when_received():
    """Quand le message reçu voyait la cible, la croyance de décision est incluse
    dans la propagation d'un pas du masque réduit à cette seule case."""
    checked = {"seen": False}

    def check(env, policy):
        if env.captured:
            return
        free = env.graph.free
        g = env.graph
        for i in range(PROFILE.n_pursuers):
            m = policy._inbox[i]
            if m is not None and m["cible"] is not None:
                checked["seen"] = True
                cell = policy.places.cell_by_name[m["cible"]]
                mask = np.zeros(free.shape, dtype=bool)
                mask[g.cells[cell]] = True
                assert not (policy.beliefs()[i] & ~propagate(mask, free)).any()

    for seed in range(10):
        run_episode(PROFILE, ProtocolPursuers(PROFILE), seed, on_step=check)
    assert checked["seen"]


def test_path_ahead_does_not_consume_the_rng():
    env, g, places = _map(1)
    policy = ProtocolPursuers(PROFILE)
    policy.reset(env.graph, np.random.default_rng(0))
    vis = env.visibility()
    percepts = [Percept(env.pursuer_pos(p), vis[p], env.target_seen_by(vis[p]))
                for p in range(PROFILE.n_pursuers)]
    policy.update(percepts)
    policy.act(percepts)  # établit _goals/_beliefs pour un appel direct à _path_ahead
    here = int(g.index[percepts[0].pos])
    state_before = policy.rng.bit_generator.state
    policy._path_ahead(0, here, Move.NORTH)
    assert policy.rng.bit_generator.state == state_before


def test_write_message_rounds_tiny_probability_up_to_one():
    _, g, places = _map(1)
    big, tiny = places.places[0], places.places[-1]
    belief = np.zeros(g.free.shape, dtype=bool)
    belief[g.cells[big.cells[0]]] = True
    belief[g.cells[tiny.cells[0]]] = True
    prob = np.zeros(g.free.shape)
    prob[g.cells[big.cells[0]]] = 0.999
    prob[g.cells[tiny.cells[0]]] = 0.001
    m = write_message(g, places, g.cells[big.cells[0]], belief, prob, None, [])
    assert m["candidates"][tiny.name] == 1


@pytest.mark.parametrize("seed", range(5))
def test_intention_first_cell_matches_the_move_played(seed):
    """Le premier lieu d'`intention` est le lieu de la case où mène le coup
    réellement joué, quand ce coup déplace le poursuivant."""
    state = {"prev_pos": None, "n": 0, "checked": False}

    def check(env, policy):
        state["n"] += 1
        cur_pos = [env.pursuer_pos(p) for p in range(PROFILE.n_pursuers)]
        if state["n"] >= 2 and state["prev_pos"] is not None:
            g = env.graph
            for i in range(PROFILE.n_pursuers):
                if cur_pos[i] != state["prev_pos"][i]:
                    state["checked"] = True
                    dest = int(g.index[cur_pos[i]])
                    place_name = policy.places.places[policy.places.place_of[dest]].name
                    assert policy.sent[i]["intention"][0] == place_name
        state["prev_pos"] = cur_pos

    run_episode(PROFILE, ProtocolPursuers(PROFILE), seed, on_step=check)
    assert state["checked"]


# --- P3 : décisions de R2, message A3 ---------------------------------------------------

from chase.llm.message import A3_SPEC


def test_a3_message_keeps_moi_cible_and_the_first_three_places_of_the_path():
    _, g, places = _map(1)
    first, last = places.places[0], places.places[-1]
    belief = np.zeros(g.free.shape, dtype=bool)
    belief[g.cells[last.cells[0]]] = True
    path = [p.cells[0] for p in places.places[:5]]
    m = write_message(g, places, g.cells[first.cells[0]], belief, belief / 1.0, None, path,
                      spec=A3_SPEC)
    assert m == {"moi": first.name, "cible": None,
                 "intention": [p.name for p in places.places[:3]]}
    assert validate(m, A3_SPEC) is None


def test_a2_message_is_unchanged_by_the_spec_argument():
    _, g, places = _map(1)
    belief = np.zeros(g.free.shape, dtype=bool)
    belief[g.cells[places.places[-1].cells[0]]] = True
    pos = g.cells[places.places[0].cells[0]]
    assert (write_message(g, places, pos, belief, belief / 1.0, None, [])
            == write_message(g, places, pos, belief, belief / 1.0, None, [], spec=A2_SPEC))


def test_protocol_name_follows_the_spec():
    assert ProtocolPursuers(PROFILE).name == "P2"
    assert ProtocolPursuers(PROFILE, spec=A3_SPEC).name == "P3"


def test_p3_messages_are_valid_a3_messages_with_map_names():
    seen = []

    def check(env, policy):
        for m in policy.sent:
            if m is not None:
                seen.append(m)
                assert validate(m, A3_SPEC) is None
                assert unknown_names(m, policy.places) == []

    run_episode(PROFILE, ProtocolPursuers(PROFILE, spec=A3_SPEC), 2, on_step=check)
    assert seen


@pytest.mark.parametrize("seed", range(5))
def test_p3_beliefs_contain_the_target_and_shrink_only_on_a_received_target(seed):
    def check(env, policy):
        if env.captured:
            return
        t = env.target_pos
        for i, (own, decided) in enumerate(zip(policy._own, policy.beliefs())):
            assert own[t] and decided[t]
            m = policy._inbox[i]
            if m is None or m["cible"] is None:
                assert (decided == own).all()
            else:
                assert not (decided & ~own).any()

    run_episode(PROFILE, ProtocolPursuers(PROFILE, spec=A3_SPEC), seed, on_step=check)


def test_p3_uses_a_received_target():
    shrunk = {"seen": False}

    def check(env, policy):
        if env.captured:
            return
        for i in range(PROFILE.n_pursuers):
            m = policy._inbox[i]
            if m is not None and m["cible"] is not None:
                shrunk["seen"] |= bool(policy.beliefs()[i].sum() < policy._own[i].sum())

    for seed in range(10):
        run_episode(PROFILE, ProtocolPursuers(PROFILE, spec=A3_SPEC), seed, on_step=check)
    assert shrunk["seen"]
