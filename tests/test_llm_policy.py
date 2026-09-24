"""Tests unitaires de LLMPursuers avec un faux client LLM (pas de réseau,
pas d'inférence réelle)."""

from __future__ import annotations

import numpy as np

from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.llm.client import LLMCallResult
from chase.llm.config import LLMConfig
from chase.llm.logging import REASONING_LOG_CHARS
from chase.llm.policy import LLMPursuers
from chase.moves import Move
from chase.policies import GreedyPursuers, Percept

CFG = ChaseConfig(max_steps=20)
LLM_CFG = LLMConfig()


class FakeLLMClient:
    """Renvoie une séquence fixée de résultats, un par appel à `decide`."""

    def __init__(self, results: list[LLMCallResult]):
        self._results = list(results)
        self.calls: list[tuple[str, str]] = []

    def decide(self, system_prompt: str, user_prompt: str) -> LLMCallResult:
        self.calls.append((system_prompt, user_prompt))
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


def test_individual_belief_matches_r1():
    """La croyance individuelle de LLMPursuers doit être identique à celle de
    GreedyPursuers(fused=False) sur la même séquence de percepts : même calcul,
    aucune fusion, aucune position de coéquipier transmise (bras sans
    communication)."""
    env = ChaseEnv(CFG)
    env.reset(seed=2)
    rng = np.random.default_rng(0)

    r1 = GreedyPursuers(CFG, fused=False)
    r1.reset(env.graph, rng)
    llm = LLMPursuers(CFG, LLM_CFG, client=FakeLLMClient([]))
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
