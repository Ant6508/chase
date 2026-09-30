"""Tests de scripts/situations_a2.py : sélection sans LLM, interrogation avec un faux client."""

from __future__ import annotations

from chase.config import ChaseConfig
from chase.llm.client import LLMCallResult
from chase.llm.message import validate
from chase.llm.prompts import MESSAGE_HEADER, system_prompt
from chase.moves import Move
from scripts.situations_a2 import FAMILIES, ask, collect

PROFILE = ChaseConfig().replace(max_steps=60, size=15, n_loops=1, min_loop_len=6,
                                min_spawn_dist=6)


def test_collect_finds_each_family_with_ready_prompts():
    situations = collect(PROFILE, range(30, 130), quotas={f: 1 for f in FAMILIES})
    assert sorted(s["family"] for s in situations) == sorted(FAMILIES)
    for s in situations:
        assert s["a2_prompt"].startswith(s["a1bis_prompt"])
        assert MESSAGE_HEADER in s["a2_prompt"] and MESSAGE_HEADER not in s["a1bis_prompt"]
        assert validate(s["own_message"]) is None
        assert s["p2_move"] in Move.__members__
        assert 30 <= s["seed"] < 130


class _Fake:
    def __init__(self):
        self.calls = []

    def decide(self, system_prompt, user_prompt, with_message=False):
        self.calls.append((system_prompt, with_message))
        message = ({"moi": "K1", "cible": None, "candidates": {"C1": 60, "Z9": 40},
                    "intention": [], "je_couvre": None} if with_message else None)
        return LLMCallResult(move=Move.NORTH, reasoning="r", prompt_tokens=1, completion_tokens=1,
                             latency_ms=1.0, retries=0, fallback=False, message=message)


def test_ask_compares_both_prompts_to_p2_and_scores_the_emitted_message():
    s = {"seed": 30, "step": 4, "pursuer": 1, "family": "tenaille", "p2_move": "NORTH",
         "a1bis_prompt": "perception", "a2_prompt": "perception\n\nmessage",
         "own_message": {"moi": "K1", "cible": None, "candidates": {"C1": 50, "C2": 50},
                         "intention": [], "je_couvre": None}}
    fake = _Fake()
    row = ask(fake, s, frozenset({"K1", "C1", "C2"}))
    assert fake.calls == [(system_prompt("A1bis"), False), (system_prompt("A2"), True)]
    assert row["a1bis_move"] == row["a2_move"] == "NORTH"
    assert row["coverage"] == 0.5
    assert row["unknown_names"] == ["Z9"]
    assert row["a2_message_valid"] is True


def test_ask_keeps_the_diagnostics_of_both_calls():
    from chase.llm.client import LLMCallResult
    from chase.moves import Move

    class Diag:
        def decide(self, system_prompt, user_prompt, with_message=False):
            return LLMCallResult(move=Move.STAY, reasoning="repli après échec : x", prompt_tokens=0,
                                 completion_tokens=9, latency_ms=3.0, retries=2, fallback=True,
                                 finish_reason="length", attempt_errors=["e1"])

    s = {"seed": 1, "step": 0, "pursuer": 0, "family": "f", "p2_move": "STAY",
         "a1bis_prompt": "p", "a2_prompt": "q",
         "own_message": {"candidates": {"C1": 1}}}
    row = ask(Diag(), s, frozenset())
    for arm in ("a1bis", "a2"):
        assert row[f"{arm}_reasoning"] == "repli après échec : x"
        assert row[f"{arm}_prompt_tokens"] == 0 and row[f"{arm}_completion_tokens"] == 9
        assert row[f"{arm}_finish_reason"] == "length" and row[f"{arm}_attempt_errors"] == ["e1"]
        assert row[f"{arm}_latency_ms"] == 3.0
