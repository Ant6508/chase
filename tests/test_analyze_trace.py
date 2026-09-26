"""Tests de scripts/analyze_trace.py sur des traces produites par run_llm_episode
avec un faux client (pas de réseau, pas de LLM)."""

from __future__ import annotations

import json

import scripts.run_llm as run_llm
from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.llm.client import LLMCallResult
from chase.llm.config import LLMConfig
from chase.moves import Move
from scripts.analyze_trace import summarize

CFG = ChaseConfig(max_steps=6)


class _Client:
    def __init__(self, move: Move):
        self.move = move

    def decide(self, system_prompt, user_prompt):
        return LLMCallResult(move=self.move, reasoning="r", prompt_tokens=1, completion_tokens=4,
                             latency_ms=1.0, retries=0, fallback=False, thinking="t")


def _trace(tmp_path, move: Move, seed: int = 3) -> list[dict]:
    path = tmp_path / f"seed_{seed}.jsonl"
    run_llm.run_llm_episode(CFG, LLMConfig(), seed, trace_path=str(path), client=_Client(move))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _graph(seed: int = 3):
    env = ChaseEnv(CFG)
    env.reset(seed=seed)
    return env.graph


def test_standing_still_is_counted_as_such_and_never_as_a_wall_bump(tmp_path):
    c = summarize(_trace(tmp_path, Move.STAY), _graph())
    assert c["décisions"] == CFG.max_steps * CFG.n_pursuers
    assert c["immobile"] == c["décisions"]
    assert c["contre_un_mur"] == 0
    assert c["allers_retours"] == 0
    assert c["cases_visitées"] == CFG.n_pursuers
    assert c["tokens_complétion"] == 4 * c["décisions"]


def test_wall_bumps_are_detected_from_the_true_map(tmp_path):
    records = _trace(tmp_path, Move.NORTH)
    g = _graph()
    # un poursuivant qui insiste vers le nord finit contre un mur et y reste
    expected = sum(1 for r in records if not g.free[r["pos"][0], r["pos"][1] - 1])
    c = summarize(records, g)
    assert expected > 0
    assert c["contre_un_mur"] == expected


def test_hidden_target_decisions_are_scored_against_the_exits_announced_in_the_perception(tmp_path):
    c = summarize(_trace(tmp_path, Move.STAY), _graph())
    # l'immobilité ne part vers aucune issue, qu'elle ait des candidates ou non
    assert c["cible_cachée"] > 0
    assert c["cachée_vers_issue_avec_candidates"] == 0
