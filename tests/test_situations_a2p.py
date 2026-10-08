"""Tests de scripts/situations_a2p.py : classement des issues sur une carte en croix,
rejeu entrelacé avec un faux client, bilan sur des lignes écrites à la main."""

from __future__ import annotations

import numpy as np

from chase.graph import MazeGraph
from chase.llm.client import LLMCallResult
from chase.llm.message import A2_SPEC
from chase.llm.prompts import system_prompt
from chase.moves import Move
from scripts.situations_a2p import approach_moves, exit_moves, play, summarize


def _cross() -> MazeGraph:
    """Croix centrée en (3, 3), bras de 2 cases ; NORTH = y décroissant."""
    free = np.zeros((7, 7), dtype=bool)
    free[3, 1:6] = True
    free[1:6, 3] = True
    return MazeGraph(free)


def test_an_exit_is_dead_when_the_teammate_message_leaves_no_candidate_there():
    g = _cross()
    belief = np.zeros(g.free.shape, dtype=bool)
    prob = np.zeros(g.free.shape)
    for cell, p in (((3, 1), 0.6), ((3, 5), 0.3), ((5, 3), 0.1)):  # nord, sud, est
        belief[cell], prob[cell] = True, p
    support = np.zeros(g.free.shape, dtype=bool)
    support[3, 5] = True  # le coéquipier ne laisse que le bout du bras sud
    out = exit_moves(g, int(g.index[3, 3]), belief, prob, support)
    assert sorted(out["dead"]) == ["EAST", "NORTH"]
    assert out["alive"] == ["SOUTH"]
    assert out["best"] == "NORTH"  # la plus probable pour le récepteur seul, morte


def test_approach_moves_lead_closer_to_the_cell():
    g = _cross()
    assert approach_moves(g, int(g.index[3, 3]), (5, 3)) == ["EAST"]
    assert approach_moves(g, int(g.index[3, 1]), (3, 5)) == ["SOUTH"]


class _Client:
    def __init__(self, moves):
        self.moves = list(moves)
        self.calls = []

    def decide(self, system_prompt, user_prompt, message_spec=None):
        self.calls.append((system_prompt, user_prompt, message_spec))
        return LLMCallResult(move=self.moves.pop(0), reasoning="r", prompt_tokens=5,
                             completion_tokens=9, latency_ms=1.0, retries=1, fallback=False,
                             thinking="pense", finish_reason="tool_calls",
                             attempt_errors=["réponse sans appel d'outil valide (finish_reason=length)"])


def test_play_interleaves_both_arms_on_the_exact_user_prompt():
    situations = [{"id": 0, "user_prompt": "p0"}, {"id": 1, "user_prompt": "p1"}]
    client = _Client([Move.NORTH, Move.SOUTH, Move.EAST, Move.WEST])
    rows = play(client, situations, concurrency=1)
    assert [(r["id"], r["arm"], r["move"]) for r in rows] == [
        (0, "A2", "NORTH"), (0, "A2p", "SOUTH"), (1, "A2", "EAST"), (1, "A2p", "WEST")]
    assert client.calls == [(system_prompt("A2"), "p0", A2_SPEC), (system_prompt("A2p"), "p0", A2_SPEC),
                            (system_prompt("A2"), "p1", A2_SPEC), (system_prompt("A2p"), "p1", A2_SPEC)]
    row = rows[0]
    assert (row["thinking_tokens"], row["retries"], row["cut_attempts"]) == (5, 1, 1)
    assert row["thinking"] == "pense" and row["fallback"] is False


def test_summarize_counts_dead_exits_and_approaches_per_arm():
    situations = [
        {"id": 0, "family": "vide_morte", "dead": ["NORTH"], "alive": ["SOUTH"], "best": "NORTH"},
        {"id": 1, "family": "cible_vue", "approach": ["EAST"]},
        {"id": 2, "family": "vide_morte", "dead": ["NORTH"], "alive": ["SOUTH"], "best": "NORTH"},
    ]
    base = {"fallback": False, "thinking_tokens": 10, "cut_attempts": 0, "retries": 0}
    rows = [{**base, "id": 0, "arm": "A2", "move": "NORTH"},
            {**base, "id": 0, "arm": "A2p", "move": "SOUTH"},
            {**base, "id": 1, "arm": "A2", "move": "EAST"},
            {**base, "id": 1, "arm": "A2p", "move": "WEST"},
            {**base, "id": 2, "arm": "A2", "move": "NORTH"},
            {**base, "id": 2, "arm": "A2p", "move": "STAY", "fallback": True}]
    s = summarize(situations, rows)
    assert s["vide_morte"]["A2"]["to_dead"] == (2, 2)
    assert s["vide_morte"]["A2p"]["to_dead"] == (0, 1)  # le repli est compté à part
    assert s["vide_morte"]["A2p"]["fallbacks"] == 1
    assert s["vide_morte"]["paired"] == {"only_A2": 1, "only_A2p": 0, "both": 0, "neither": 0}
    assert s["cible_vue"]["A2"]["approach"] == (1, 1)
    assert s["cible_vue"]["A2p"]["approach"] == (0, 1)
