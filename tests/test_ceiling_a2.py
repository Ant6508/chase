"""Tests de scripts/ceiling_a2.py (sans LLM)."""

from __future__ import annotations

import math
import sys

import pytest

import scripts.ceiling_a2 as ceiling_a2
from scripts.ceiling_a2 import gate, kept_share


def test_gate_asks_p2_to_keep_half_of_the_r1_r2_gap():
    kept, ok = gate({"R1": 0.6, "R2": 0.8, "P2": 0.7})
    assert kept == pytest.approx(0.5) and ok
    kept, ok = gate({"R1": 0.6, "R2": 0.8, "P2": 0.65})
    assert kept == pytest.approx(0.25) and not ok


def test_gate_cannot_be_passed_without_a_gap():
    kept, ok = gate({"R1": 0.8, "R2": 0.8, "P2": 0.9})
    assert math.isnan(kept) and not ok


def test_table_lists_the_policies_the_gate_and_the_share_kept_by_p3(tmp_path, monkeypatch):
    out = tmp_path / "plafond.md"
    monkeypatch.setattr(sys, "argv", [
        "ceiling_a2", "--episodes", "2", "--set", "max_steps=10", "size=15", "n_loops=1",
        "min_loop_len=6", "min_spawn_dist=6", "--out", str(out)])
    ceiling_a2.main()
    text = out.read_text(encoding="utf-8")
    for name in ("R1", "R2", "P2", "P3"):
        assert f"| {name} |" in text
    assert "Porte" in text
    assert "P3 (message A3 : moi, cible, intention) en garde" in text


def test_kept_share_is_measured_against_the_r1_r2_gap():
    assert kept_share({"R1": 0.6, "R2": 0.8, "P3": 0.65}, "P3") == pytest.approx(0.25)
    assert math.isnan(kept_share({"R1": 0.8, "R2": 0.8, "P3": 0.9}, "P3"))


def test_p3_is_the_protocol_policy_with_the_a3_spec():
    from chase.config import ChaseConfig
    from chase.llm.message import A3_SPEC
    policy = ceiling_a2.POLICIES["P3"](ChaseConfig())
    assert policy.name == "P3" and policy.spec is A3_SPEC
