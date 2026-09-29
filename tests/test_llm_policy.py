"""Tests unitaires de LLMPursuers avec un faux client LLM (pas de réseau,
pas d'inférence réelle)."""

from __future__ import annotations

import numpy as np
import pytest

from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.llm.client import LLMCallResult
from chase.llm.config import LLMConfig
from chase.llm.logging import REASONING_LOG_CHARS
from chase.llm.message import render
from chase.llm.policy import LLMPursuers
from chase.llm.prompts import MESSAGE_HEADER, NO_MESSAGE, SYSTEM_PROMPT, build_perception, system_prompt
from chase.moves import Move
from chase.policies import GreedyPursuers, Percept

CFG = ChaseConfig(max_steps=20)
LLM_CFG = LLMConfig()


class FakeLLMClient:
    """Renvoie une séquence fixée de résultats, un par appel à `decide`."""

    def __init__(self, results: list[LLMCallResult]):
        self._results = list(results)
        self.calls: list[tuple[str, str]] = []
        self.with_message: list[bool] = []

    def decide(self, system_prompt: str, user_prompt: str,
               with_message: bool = False) -> LLMCallResult:
        self.calls.append((system_prompt, user_prompt))
        self.with_message.append(with_message)
        return self._results.pop(0)


def _ok(move: Move) -> LLMCallResult:
    return LLMCallResult(move=move, reasoning="parce que", prompt_tokens=42,
                         completion_tokens=7, latency_ms=123.0, retries=0, fallback=False)


def _fallback() -> LLMCallResult:
    return LLMCallResult(move=Move.STAY, reasoning="repli après échec : boom",
                         prompt_tokens=0, completion_tokens=0, latency_ms=0.0,
                         retries=2, fallback=True)


def _percepts(env, vis):
    return [Percept(env.pursuer_pos(p), vis[p], env.target_seen_by(vis[p]))
            for p in range(env.cfg.n_pursuers)]


@pytest.mark.parametrize("arm", ["A1", "A1bis", "A2"])
def test_individual_belief_matches_r1(arm):
    """La croyance individuelle de LLMPursuers doit être identique à celle de
    GreedyPursuers(fused=False) sur la même séquence de percepts : même calcul,
    aucune fusion, aucune position de coéquipier transmise (bras sans
    communication)."""
    env = ChaseEnv(CFG)
    env.reset(seed=2)
    rng = np.random.default_rng(0)

    r1 = GreedyPursuers(CFG, fused=False)
    r1.reset(env.graph, rng)
    llm = LLMPursuers(CFG, LLM_CFG, client=FakeLLMClient([]), arm=arm)
    llm.reset(env.graph, rng)

    vis = env.visibility()
    percepts = _percepts(env, vis)
    r1.update(percepts)
    llm.update(percepts)
    for b1, b2 in zip(r1.beliefs(), llm.beliefs()):
        assert (b1 == b2).all()

    for _ in range(5):
        moves = [Move.STAY] * (CFG.n_pursuers + 1)  # tout le monde immobile
        vis = env.step_moves(moves)
        percepts = _percepts(env, vis)
        r1.update(percepts)
        llm.update(percepts)
        for b1, b2 in zip(r1.beliefs(), llm.beliefs()):
            assert (b1 == b2).all()


def test_act_returns_move_from_client():
    env = ChaseEnv(CFG)
    env.reset(seed=3)
    client = FakeLLMClient([_ok(Move.NORTH), _ok(Move.EAST)])
    policy = LLMPursuers(CFG, LLM_CFG, client=client)
    policy.reset(env.graph, np.random.default_rng(0))
    vis = env.visibility()
    percepts = _percepts(env, vis)
    policy.update(percepts)
    moves = policy.act(percepts)
    assert moves == [Move.NORTH, Move.EAST]


def test_fallback_is_logged():
    env = ChaseEnv(CFG)
    env.reset(seed=3)
    client = FakeLLMClient([_fallback(), _ok(Move.WEST)])
    policy = LLMPursuers(CFG, LLM_CFG, client=client)
    policy.reset(env.graph, np.random.default_rng(0))
    vis = env.visibility()
    percepts = _percepts(env, vis)
    policy.update(percepts)
    moves = policy.act(percepts)
    assert moves[0] == Move.STAY
    assert policy.step_logs[0].fallback is True
    assert policy.step_logs[1].fallback is False


def test_step_log_fields_present_and_reasoning_truncated():
    env = ChaseEnv(CFG)
    env.reset(seed=3)
    long_reasoning = "x" * 500
    result = LLMCallResult(move=Move.SOUTH, reasoning=long_reasoning, prompt_tokens=10,
                            completion_tokens=20, latency_ms=99.0, retries=1, fallback=False)
    client = FakeLLMClient([result, result])
    policy = LLMPursuers(CFG, LLM_CFG, client=client)
    policy.reset(env.graph, np.random.default_rng(0))
    vis = env.visibility()
    percepts = _percepts(env, vis)
    policy.update(percepts)
    policy.act(percepts)

    log = policy.step_logs[0]
    assert log.pursuer == 0
    assert log.move == Move.SOUTH
    assert len(log.reasoning) == REASONING_LOG_CHARS
    assert log.prompt_tokens == 10
    assert log.completion_tokens == 20
    assert log.latency_ms == 99.0
    assert log.retries == 1
    assert log.fallback is False
    assert log.message_tokens == 0


def test_step_log_keeps_what_the_model_saw_and_thought_for_diagnosis():
    env = ChaseEnv(CFG)
    env.reset(seed=3)
    thinking = "y" * 3000  # pensée longue : jamais tronquée, c'est l'objet du diagnostic
    result = LLMCallResult(move=Move.EAST, reasoning="r", prompt_tokens=1, completion_tokens=1,
                           latency_ms=1.0, retries=0, fallback=False, thinking=thinking)
    client = FakeLLMClient([result, result])
    policy = LLMPursuers(CFG, LLM_CFG, client=client)
    policy.reset(env.graph, np.random.default_rng(0))
    percepts = _percepts(env, env.visibility())
    policy.update(percepts)
    policy.act(percepts)

    for i, log in enumerate(policy.step_logs):
        assert log.pos == percepts[i].pos
        assert log.perception == client.calls[i][1]
        assert log.thinking == thinking


def test_probability_map_matches_r1_and_reaches_the_perception():
    """Même carte de probabilité que GreedyPursuers(fused=False), qui s'en sert
    pour ne pas revenir sur une case qu'il vient de vider ; le LLM la reçoit
    résumée par issue."""
    env = ChaseEnv(CFG)
    env.reset(seed=2)
    rng = np.random.default_rng(0)
    r1 = GreedyPursuers(CFG, fused=False)
    r1.reset(env.graph, rng)
    client = FakeLLMClient([_ok(Move.STAY)] * (2 * CFG.n_pursuers))
    llm = LLMPursuers(CFG, LLM_CFG, client=client)
    llm.reset(env.graph, rng)

    percepts = _percepts(env, env.visibility())
    for _ in range(2):
        r1.update(percepts)
        llm.update(percepts)
        for p1, p2 in zip(r1._probs, llm.probs()):
            assert np.allclose(p1, p2)
        llm.act(percepts)
        percepts = _percepts(env, env.step_moves([Move.STAY] * (CFG.n_pursuers + 1)))
    assert all("% de la probabilité" in user for _, user in client.calls)


def _say(moi: str, **changes) -> LLMCallResult:
    """Coup valide avec un message A2 minimal, dont `moi` identifie l'auteur."""
    fields = dict(move=Move.STAY, reasoning="r", prompt_tokens=1, completion_tokens=1,
                  latency_ms=1.0, retries=0, fallback=False,
                  message={"moi": moi, "cible": None, "candidates": {}, "intention": [],
                           "je_couvre": None})
    fields.update(changes)
    return LLMCallResult(**fields)


def _two_steps(arm: str, results: list[LLMCallResult]):
    env = ChaseEnv(CFG)
    env.reset(seed=3)
    client = FakeLLMClient(results)
    policy = LLMPursuers(CFG, LLM_CFG, client=client, arm=arm)
    policy.reset(env.graph, np.random.default_rng(0))
    percepts = _percepts(env, env.visibility())
    for _ in range(2):
        policy.update(percepts)
        policy.act(percepts)
        percepts = _percepts(env, env.step_moves([Move.STAY] * (CFG.n_pursuers + 1)))
    return policy, client


def test_a2_message_is_read_by_the_other_pursuer_at_the_next_step_only():
    policy, client = _two_steps("A2", [_say("K1"), _say("K2"), _say("K3"), _say("K4")])
    (_, p0_t0), (_, p1_t0), (_, p0_t1), (_, p1_t1) = client.calls
    assert p0_t0.endswith(f"{MESSAGE_HEADER}\n{NO_MESSAGE}")
    assert p1_t0.endswith(f"{MESSAGE_HEADER}\n{NO_MESSAGE}")  # pas le message de P0 du même pas
    assert p0_t1.endswith(render(_say("K2").message))
    assert p1_t1.endswith(render(_say("K1").message))
    assert client.with_message == [True] * 4
    assert all(system == system_prompt("A2") for system, _ in client.calls)


def test_a2_after_a_fallback_the_teammate_reads_no_message():
    policy, client = _two_steps("A2", [_fallback(), _say("K2"), _say("K3"), _say("K4")])
    _, _, (_, p0_t1), (_, p1_t1) = client.calls
    assert p1_t1.endswith(f"{MESSAGE_HEADER}\n{NO_MESSAGE}")
    assert p0_t1.endswith(render(_say("K2").message))
    assert policy.step_logs[0].message_out is None
    assert policy.step_logs[0].message_tokens == 0


def test_a2_logs_message_size_thinking_raw_arguments_and_unknown_names():
    said = _say("Z9", thinking="abcd", raw_arguments='{"direction":"STAY"}')
    policy, _ = _two_steps("A2", [said] * 4)
    log = policy.step_logs[0]
    assert log.message_out == render(said.message)
    assert log.message_tokens == len(log.message_out)  # tokenizer de test : 1 token par caractère
    assert log.thinking_tokens == 4
    assert log.unknown_names == ["Z9"]
    assert log.raw_arguments == '{"direction":"STAY"}'
    assert log.message_in is None
    assert policy.step_logs[2].message_in == render(said.message)
    assert log.user_prompt.startswith(log.perception)


def test_a1bis_names_places_but_has_no_message():
    policy, client = _two_steps("A1bis", [_ok(Move.STAY)] * 4)
    assert all(user.startswith("Tu es en ") for _, user in client.calls)
    assert all(MESSAGE_HEADER not in user for _, user in client.calls)
    assert client.with_message == [False] * 4
    assert all(system == system_prompt("A1bis") for system, _ in client.calls)
    assert all(log.message_out is None and log.message_tokens == 0 for log in policy.step_logs)


def test_a1_keeps_the_a1v3_prompts():
    env = ChaseEnv(CFG)
    env.reset(seed=3)
    client = FakeLLMClient([_ok(Move.STAY)] * 2)
    policy = LLMPursuers(CFG, LLM_CFG, client=client)
    policy.reset(env.graph, np.random.default_rng(0))
    percepts = _percepts(env, env.visibility())
    policy.update(percepts)
    policy.act(percepts)
    for i, (system, user) in enumerate(client.calls):
        assert system == SYSTEM_PROMPT
        assert user == build_perception(percepts[i].pos, env.graph, policy.beliefs()[i],
                                        policy.probs()[i], percepts[i].target_seen,
                                        CFG.track_threshold)


def test_a2_requires_two_pursuers():
    with pytest.raises(ValueError, match="2 poursuivants"):
        LLMPursuers(CFG.replace(n_pursuers=3), LLM_CFG, client=FakeLLMClient([]), arm="A2")


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="bras inconnu"):
        LLMPursuers(CFG, LLM_CFG, client=FakeLLMClient([]), arm="A3")
