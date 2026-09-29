"""Tests de scripts/run_llm.py avec un épisode factice (pas de réseau, pas de
LLM) : journal écrit au fil de l'eau, reprise d'un run interrompu, adresse du
serveur surchargeable."""

from __future__ import annotations

import json
import sys

import pytest

import scripts.run_llm as run_llm
from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.llm.client import LLMCallResult
from chase.llm.config import LLMConfig
from chase.llm.logging import EpisodeLLMStats
from chase.llm.message import render
from chase.llm.places import Places
from chase.llm.prompts import system_prompt
from chase.moves import Move


def _fake_episode(calls: list, traces: list | None = None):
    def fake(cfg, llm_cfg, seed, on_step=None, trace_path=None, arm="A1"):
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


def test_timeout_option_overrides_the_client_timeout(tmp_path, monkeypatch):
    """En local, un GPU lent génère ~7 tokens/s par partie à 4 en parallèle :
    une réponse de 1 600 tokens dépasse les 180 s par défaut."""
    timeouts = []

    def fake(cfg, llm_cfg, seed, on_step=None, trace_path=None, arm="A1"):
        timeouts.append(llm_cfg.timeout_s)
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0)

    monkeypatch.setattr(run_llm, "run_llm_episode", fake)
    _run(monkeypatch, "--episodes", "1", "--timeout", "400", "--journal", str(tmp_path / "j.jsonl"))

    assert timeouts == [400.0]
    first = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["params"]["llm"]["timeout_s"] == 400.0


def test_arm_option_reaches_the_episode_and_the_journal(tmp_path, monkeypatch):
    arms = []

    def fake(cfg, llm_cfg, seed, on_step=None, trace_path=None, arm="A1"):
        arms.append(arm)
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0)

    monkeypatch.setattr(run_llm, "run_llm_episode", fake)
    _run(monkeypatch, "--episodes", "1", "--arm", "A2", "--journal", str(tmp_path / "j.jsonl"))
    assert arms == ["A2"]
    first = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["params"]["arm"] == "A2"


def test_a1_journal_keeps_the_a1v3_params(tmp_path, monkeypatch):
    """Un journal A1 garde les paramètres d'A1v3 : il reste reprenable."""
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([]))
    _run(monkeypatch, "--episodes", "1", "--journal", str(tmp_path / "j.jsonl"))
    first = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(first["params"]) == {"game", "llm"}


def test_resuming_the_journal_of_another_arm_is_refused(tmp_path, monkeypatch):
    journal = str(tmp_path / "j.jsonl")
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([]))
    _run(monkeypatch, "--episodes", "1", "--arm", "A2", "--journal", journal)
    with pytest.raises(ValueError, match="paramètres"):
        _run(monkeypatch, "--episodes", "2", "--arm", "A1bis", "--journal", journal)


def test_trace_dir_keeps_the_system_prompt_of_the_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([]))
    _run(monkeypatch, "--episodes", "1", "--pilot", "--arm", "A2", "--trace", str(tmp_path / "trace"))
    saved = (tmp_path / "trace" / "system_prompt.txt").read_text(encoding="utf-8")
    assert saved == system_prompt("A2")


_SAID = {"moi": "K1", "cible": None, "candidates": {"C1": 100}, "intention": [], "je_couvre": None}


class _TalkingClient:
    def decide(self, system_prompt, user_prompt, with_message=False):
        return LLMCallResult(move=Move.STAY, reasoning="r", prompt_tokens=1, completion_tokens=2,
                             latency_ms=1.0, retries=0, fallback=False, thinking="t",
                             raw_arguments="{}", message=dict(_SAID))


def test_a2_trace_keeps_prompts_and_messages_for_a4(tmp_path):
    cfg = ChaseConfig(max_steps=2)
    trace = tmp_path / "seed_3.jsonl"
    run_llm.run_llm_episode(cfg, LLMConfig(), 3, trace_path=str(trace), client=_TalkingClient(),
                            arm="A2")
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    sent = render(_SAID)
    assert [r["message_out"] for r in records] == [sent] * 4
    assert [r["message_in"] for r in records] == [None, None, sent, sent]
    assert records[2]["user_prompt"].startswith(records[2]["perception"])
    assert records[2]["user_prompt"].endswith(sent)
    assert records[0]["raw_arguments"] == "{}"
    assert records[0]["message_tokens"] == len(sent)  # tokenizer de test : 1 token par caractère
    env = ChaseEnv(cfg)
    env.reset(seed=3)
    names = Places.from_graph(env.graph).names
    assert records[0]["unknown_names"] == [n for n in ("K1", "C1") if n not in names]
