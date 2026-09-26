"""Tests de scripts/run_llm.py avec un épisode factice (pas de réseau, pas de
LLM) : journal écrit au fil de l'eau, reprise d'un run interrompu, adresse du
serveur surchargeable."""

from __future__ import annotations

import json
import sys

import scripts.run_llm as run_llm
from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.llm.client import LLMCallResult
from chase.llm.config import LLMConfig
from chase.llm.logging import EpisodeLLMStats
from chase.moves import Move


def _fake_episode(calls: list, traces: list | None = None):
    def fake(cfg, llm_cfg, seed, on_step=None, trace_path=None):
        calls.append((seed, llm_cfg.base_url))
        if traces is not None:
            traces.append(trace_path)
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


class _StayClient:
    def decide(self, system_prompt, user_prompt):
        return LLMCallResult(move=Move.STAY, reasoning="j'attends", prompt_tokens=1,
                             completion_tokens=2, latency_ms=1.0, retries=0, fallback=False,
                             thinking="rien ne bouge")


def test_trace_records_every_decision_with_the_true_target_position(tmp_path):
    cfg = ChaseConfig(max_steps=3)
    trace = tmp_path / "seed_3.jsonl"
    run_llm.run_llm_episode(cfg, LLMConfig(), 3, trace_path=str(trace), client=_StayClient())

    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [(r["step"], r["pursuer"]) for r in records] == [
        (step, p) for step in range(3) for p in range(cfg.n_pursuers)]
    env = ChaseEnv(cfg)
    env.reset(seed=3)
    first = records[0]
    assert first["seed"] == 3
    assert first["target"] == list(env.target_pos)
    assert first["pos"] == list(env.pursuer_pos(0))
    assert first["move"] == "STAY"
    assert first["thinking"] == "rien ne bouge"
    assert first["perception"].startswith("Cible")


def test_trace_option_writes_one_file_per_seed(tmp_path, monkeypatch):
    traces = []
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([], traces))

    _run(monkeypatch, "--first-seed", "8", "--episodes", "2", "--pilot",
         "--trace", str(tmp_path / "trace"))

    assert traces == [str(tmp_path / "trace" / "seed_8.jsonl"),
                      str(tmp_path / "trace" / "seed_9.jsonl")]
    assert (tmp_path / "trace").is_dir()
