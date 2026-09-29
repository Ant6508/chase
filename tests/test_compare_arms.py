"""Tests de scripts/compare_arms.py (sans LLM)."""

from __future__ import annotations

import json
import sys

import pytest

import scripts.compare_arms as compare_arms
from scripts.compare_arms import captures_from_journal, compare, mcnemar_p


def test_mcnemar_matches_the_a1v3_report():
    assert round(mcnemar_p(8, 1), 2) == 0.04   # A1v3 contre A1v2
    assert round(mcnemar_p(3, 6), 2) == 0.51   # A1v3 contre R1
    assert round(mcnemar_p(2, 11), 2) == 0.02  # A1v3 contre R2
    assert mcnemar_p(0, 0) == 1.0


def test_compare_lists_discordant_seeds():
    a = {0: True, 1: True, 2: False, 3: False}
    b = {0: True, 1: False, 2: True, 3: False}
    r = compare(a, b)
    assert (r["a"], r["b"], r["only_a"], r["only_b"]) == (2, 2, [1], [2])


def test_journal_captures_are_read_by_seed_and_missing_seeds_refused(tmp_path):
    path = tmp_path / "j.jsonl"
    path.write_text('{"seed": 0, "captured": true}\n{"seed": 1, "captured": false}\n{"seed": 2, "capt',
                    encoding="utf-8")
    assert captures_from_journal(str(path), [0, 1]) == {0: True, 1: False}
    with pytest.raises(ValueError, match="absentes"):
        captures_from_journal(str(path), [0, 1, 2])


def test_main_prints_one_row_per_pair(tmp_path, monkeypatch, capsys):
    j = tmp_path / "a.jsonl"
    j.write_text("".join(json.dumps({"seed": s, "captured": s == 0}) + "\n" for s in range(2)),
                 encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "compare_arms", "--episodes", "2", "--set", "max_steps=10", "size=15", "n_loops=1",
        "min_loop_len=6", "min_spawn_dist=6", "--journal", f"A={j}", "--pairs", "A:R1"])
    compare_arms.main()
    assert "| A contre R1 |" in capsys.readouterr().out
