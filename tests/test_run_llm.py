"""Tests de scripts/run_llm.py avec un épisode factice (pas de réseau, pas de
LLM) : journal écrit au fil de l'eau, reprise d'un run interrompu, adresse du
serveur surchargeable."""

from __future__ import annotations

import sys

import scripts.run_llm as run_llm
from chase.llm.logging import EpisodeLLMStats


def _fake_episode(calls: list):
    def fake(cfg, llm_cfg, seed, on_step=None):
        calls.append((seed, llm_cfg.base_url))
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0)
    return fake


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_llm", *args])
    run_llm.main()


def test_resume_only_plays_seeds_missing_from_the_journal(tmp_path, monkeypatch):
    journal, out = tmp_path / "a1.jsonl", tmp_path / "a1.md"
    calls = []
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode(calls))

    _run(monkeypatch, "--episodes", "2", "--concurrency", "2", "--journal", str(journal))
    assert sorted(seed for seed, _ in calls) == [0, 1]

    calls.clear()
    _run(monkeypatch, "--episodes", "3", "--journal", str(journal), "--out", str(out))
    assert [seed for seed, _ in calls] == [2]

    table = out.read_text()
    for seed in (0, 1, 2):
        assert f"| {seed} | non | 7 | 0.500 |" in table


def test_table_stays_in_seed_order_when_resuming_a_middle_seed(tmp_path, monkeypatch):
    journal, out = tmp_path / "a1.jsonl", tmp_path / "a1.md"
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([]))
    _run(monkeypatch, "--first-seed", "1", "--episodes", "1", "--journal", str(journal))

    _run(monkeypatch, "--episodes", "3", "--journal", str(journal), "--out", str(out))

    seeds = [line.split("|")[1].strip() for line in out.read_text().splitlines()
             if line.startswith("| ") and line.split("|")[1].strip().isdigit()]
    assert seeds == ["0", "1", "2"]


def test_base_url_option_overrides_the_default_server(monkeypatch):
    calls = []
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode(calls))

    _run(monkeypatch, "--episodes", "1", "--pilot", "--base-url", "http://127.0.0.1:1234/v1")

    assert calls == [(0, "http://127.0.0.1:1234/v1")]
