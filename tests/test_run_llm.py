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
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0), 0.25
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
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0), 0.25

    monkeypatch.setattr(run_llm, "run_llm_episode", fake)
    _run(monkeypatch, "--episodes", "1", "--timeout", "400", "--journal", str(tmp_path / "j.jsonl"))

    assert timeouts == [400.0]
    first = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["params"]["llm"]["timeout_s"] == 400.0


def test_max_tokens_option_reaches_the_episode_and_the_journal(tmp_path, monkeypatch):
    """A2 pense ~2 900 tokens par appel : son budget (4 000) dépasse celui d'A1."""
    budgets = []

    def fake(cfg, llm_cfg, seed, on_step=None, trace_path=None, arm="A1"):
        budgets.append(llm_cfg.max_tokens)
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0), 0.25

    monkeypatch.setattr(run_llm, "run_llm_episode", fake)
    _run(monkeypatch, "--episodes", "1", "--max-tokens", "4000", "--journal", str(tmp_path / "j.jsonl"))

    assert budgets == [4000]
    first = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["params"]["llm"]["max_tokens"] == 4000


def test_journal_keeps_the_default_max_tokens_without_the_option(tmp_path, monkeypatch):
    def fake(cfg, llm_cfg, seed, on_step=None, trace_path=None, arm="A1"):
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0), 0.25

    monkeypatch.setattr(run_llm, "run_llm_episode", fake)
    _run(monkeypatch, "--episodes", "1", "--journal", str(tmp_path / "j.jsonl"))

    first = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["params"]["llm"]["max_tokens"] == 1600


def test_arm_option_reaches_the_episode_and_the_journal(tmp_path, monkeypatch):
    arms = []

    def fake(cfg, llm_cfg, seed, on_step=None, trace_path=None, arm="A1"):
        arms.append(arm)
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0), 0.25

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
    def decide(self, system_prompt, user_prompt, message_spec=None):
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


# --- durcissement avant les campagnes --------------------------------------------------

def test_trace_records_carry_latency_prompt_tokens_finish_reason_and_attempt_errors(tmp_path):
    class _Cut:
        def decide(self, system_prompt, user_prompt):
            return LLMCallResult(move=Move.STAY, reasoning="r", prompt_tokens=77,
                                 completion_tokens=2, latency_ms=12.5, retries=1, fallback=False,
                                 finish_reason="stop", attempt_errors=["coupé"])

    trace = tmp_path / "seed_3.jsonl"
    run_llm.run_llm_episode(ChaseConfig(max_steps=1), LLMConfig(), 3, trace_path=str(trace),
                            client=_Cut())
    first = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert (first["prompt_tokens"], first["latency_ms"]) == (77, 12.5)
    assert (first["finish_reason"], first["attempt_errors"]) == ("stop", ["coupé"])


def test_episode_returns_the_mean_confinement_like_run_episode():
    from chase.policies import GreedyPursuers
    from chase.runner import run_episode

    class Stay(GreedyPursuers):
        def act(self, percepts):
            return [Move.STAY] * len(percepts)

    cfg = ChaseConfig(max_steps=6)
    *_, mean_conf = run_llm.run_llm_episode(cfg, LLMConfig(), 3, client=_StayClient())
    ref = run_episode(cfg, Stay(cfg, fused=False), 3)
    assert mean_conf == pytest.approx(ref.mean_confinement)


def test_trace_dir_keeps_tools_json_for_a4(tmp_path, monkeypatch):
    from chase.llm.client import move_tool
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([]))
    _run(monkeypatch, "--episodes", "1", "--pilot", "--arm", "A2", "--trace", str(tmp_path / "t"))
    tools = json.loads((tmp_path / "t" / "tools.json").read_text(encoding="utf-8"))
    from chase.llm.message import A2_SPEC
    assert tools == [move_tool(A2_SPEC)]


def test_journal_params_fingerprint_the_prompts_outside_a1(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([]))
    _run(monkeypatch, "--episodes", "1", "--arm", "A2", "--journal", str(tmp_path / "j.jsonl"))
    first = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(first["params"]["prompts"]) == 16
    assert first["mean_confinement"] == 0.25
    monkeypatch.setattr(run_llm, "system_prompt", lambda arm: "autre prompt")
    with pytest.raises(ValueError, match="paramètres"):
        _run(monkeypatch, "--episodes", "2", "--arm", "A2", "--journal", str(tmp_path / "j.jsonl"))


def test_a_failing_episode_does_not_stop_the_campaign(tmp_path, monkeypatch, capsys):
    def fake(cfg, llm_cfg, seed, on_step=None, trace_path=None, arm="A1"):
        if seed == 1:
            raise RuntimeError("episode casse")
        return False, 7, 0.5, EpisodeLLMStats(10, 5, 0, 100.0, 1, 1.0), 0.25

    monkeypatch.setattr(run_llm, "run_llm_episode", fake)
    journal, out = tmp_path / "j.jsonl", tmp_path / "t.md"
    _run(monkeypatch, "--episodes", "3", "--journal", str(journal), "--out", str(out))
    seen = [json.loads(l)["seed"] for l in journal.read_text(encoding="utf-8").splitlines()]
    assert sorted(seen) == [0, 2]
    text = out.read_text(encoding="utf-8")
    assert "Confinement moyen" in text and "seeds en échec : 1" in text
    assert "| 1 | non" not in text
    assert "episode casse" in capsys.readouterr().err


def test_resuming_an_old_journal_without_mean_confinement_gives_nan(tmp_path, monkeypatch):
    journal = tmp_path / "j.jsonl"
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([]))
    _run(monkeypatch, "--episodes", "1", "--journal", str(journal))
    rec = json.loads(journal.read_text(encoding="utf-8"))
    del rec["mean_confinement"]
    journal.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = tmp_path / "t.md"
    _run(monkeypatch, "--episodes", "1", "--journal", str(journal), "--out", str(out))
    assert "nan" in out.read_text(encoding="utf-8")


# --- bras A3 ------------------------------------------------------------------------------

from pathlib import Path

from chase.llm.client import move_tool
from chase.llm.message import A3_SPEC

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm", ["A1bis", "A2", "A2p"])
def test_played_arms_keep_the_fingerprint_of_their_campaign_journal(arm):
    """Les journaux des campagnes restent reprenables et comparables : l'empreinte
    calculée aujourd'hui est celle qu'ils portent."""
    with open(ROOT / "results" / f"jalon2_{arm.lower()}" / "journal.jsonl", encoding="utf-8") as f:
        params = json.loads(f.readline())["params"]
    assert params["arm"] == arm
    assert run_llm.prompts_fingerprint(arm) == params["prompts"]


def test_a3_journal_and_trace_dir_carry_the_a3_prompt_and_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "run_llm_episode", _fake_episode([]))
    _run(monkeypatch, "--episodes", "1", "--arm", "A3", "--journal", str(tmp_path / "j.jsonl"),
         "--trace", str(tmp_path / "t"))
    first = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["params"]["arm"] == "A3"
    assert first["params"]["prompts"] == run_llm.prompts_fingerprint("A3")
    assert first["params"]["prompts"] != run_llm.prompts_fingerprint("A2")
    tools = json.loads((tmp_path / "t" / "tools.json").read_text(encoding="utf-8"))
    assert tools == [move_tool(A3_SPEC)]
    assert (tmp_path / "t" / "system_prompt.txt").read_text(encoding="utf-8") == system_prompt("A3")


_SAID3 = {"moi": "K1", "cible": None, "intention": ["C1"]}


class _MinimalClient:
    def __init__(self):
        self.specs = []

    def decide(self, system_prompt, user_prompt, message_spec=None):
        self.specs.append(message_spec)
        return LLMCallResult(move=Move.STAY, reasoning="r", prompt_tokens=1, completion_tokens=2,
                             latency_ms=1.0, retries=0, fallback=False, thinking="t",
                             raw_arguments="{}", message=dict(_SAID3))


def test_a3_trace_keeps_the_minimal_messages(tmp_path):
    cfg = ChaseConfig(max_steps=2)
    trace = tmp_path / "seed_3.jsonl"
    client = _MinimalClient()
    run_llm.run_llm_episode(cfg, LLMConfig(), 3, trace_path=str(trace), client=client, arm="A3")
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    sent = render(_SAID3)
    assert sent == '{"moi":"K1","cible":null,"intention":["C1"]}'
    assert [r["message_out"] for r in records] == [sent] * 4
    assert [r["message_in"] for r in records] == [None, None, sent, sent]
    assert records[0]["message_tokens"] == len(sent)  # tokenizer de test : 1 token par caractère
    assert client.specs == [A3_SPEC] * 4
