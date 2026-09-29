"""Tests de scripts/replay_trace.py avec des traces écrites à la main et un faux client."""

from __future__ import annotations

import json

from chase.llm.client import LLMCallResult
from chase.llm.prompts import SYSTEM_PROMPT
from chase.moves import Move
from scripts.replay_trace import draw, parse_seeds, replay

_OPEN = "- {} : praticable ; 3 cases candidates au plus court par là (50 % de la probabilité), " \
        "la plus proche à 2 pas"


def _rec(seed, step, move="EAST", exits=("EST", "OUEST"), fallback=False):
    lines = ["Cible non visible.", "Directions :"] + [_OPEN.format(d) for d in exits]
    return {"seed": seed, "step": step, "pursuer": 0, "move": move, "fallback": fallback,
            "completion_tokens": 10, "perception": "\n".join(lines),
            "reasoning": "vers l'est", "thinking": "je pense"}


def _write(tmp_path, seed, records):
    (tmp_path / f"seed_{seed}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def test_seed_ranges_and_single_seeds_are_expanded():
    assert parse_seeds("4-6,9") == [4, 5, 6, 9]
    assert parse_seeds("17") == [17]


def test_only_real_choices_are_drawn_and_groups_are_interleaved(tmp_path):
    _write(tmp_path, 1, [_rec(1, 0), _rec(1, 1, fallback=True), _rec(1, 2, exits=("EST",))])
    _write(tmp_path, 2, [_rec(2, 0), _rec(2, 1)])
    _write(tmp_path, 3, [_rec(3, 5)])
    sample = draw(str(tmp_path), {"a": [1], "b": [2, 3]}, n=1)
    assert [name for name, _ in sample] == ["a", "b"]
    # seul le pas 0 de la seed 1 est un vrai choix : ni repli, ni issue unique
    assert (sample[0][1]["seed"], sample[0][1]["step"]) == (1, 0)
    assert sample[1][1]["seed"] in (2, 3)


class _Client:
    def __init__(self, move, thinking):
        self.move, self.thinking, self.prompts = move, thinking, []

    def decide(self, system_prompt, user_prompt):
        self.prompts.append((system_prompt, user_prompt))
        return LLMCallResult(move=self.move, reasoning="vers l'est", prompt_tokens=1,
                             completion_tokens=12, latency_ms=1.0, retries=0, fallback=False,
                             thinking=self.thinking)


def test_replay_sends_the_recorded_perception_and_compares_move_and_thinking():
    rec = _rec(4, 7)
    client = _Client(Move.EAST, "je pense")
    row = replay(client, "avant", rec)
    assert client.prompts == [(SYSTEM_PROMPT, rec["perception"])]
    assert row["same_move"] and row["same_thinking"] and row["same_reasoning"]
    assert (row["recorded_tokens"], row["replayed_tokens"]) == (10, 12)

    row = replay(_Client(Move.WEST, "autre chose"), "avant", rec)
    assert (row["recorded"], row["replayed"]) == ("EAST", "WEST")
    assert not row["same_move"] and not row["same_thinking"]
