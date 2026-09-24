"""Tests unitaires de LMStudioClient (chase/llm/client.py), sans appel réseau :
le SDK OpenAI est remplacé par un faux client injecté au constructeur."""

from __future__ import annotations

import json
from types import SimpleNamespace

from chase.llm.client import LMStudioClient, _MOVE_TOOL
from chase.llm.config import LLMConfig
from chase.moves import Move

CFG = LLMConfig(max_retries=2)


def _tool_call_response(direction: str, reasoning: str = "parce que",
                        prompt_tokens: int = 10, completion_tokens: int = 5):
    call = SimpleNamespace(function=SimpleNamespace(
        arguments=json.dumps({"direction": direction, "reasoning": reasoning})))
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                                                  completion_tokens=completion_tokens))


def _no_tool_call_response():
    message = SimpleNamespace(tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeOpenAI:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def test_decide_parses_valid_tool_call():
    fake = _FakeOpenAI([_tool_call_response("NORTH", "je vois la cible au nord")])
    client = LMStudioClient(CFG, client=fake)
    result = client.decide("système", "perception")
    assert result.move == Move.NORTH
    assert result.reasoning == "je vois la cible au nord"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.retries == 0
    assert result.fallback is False


def test_decide_forces_the_move_tool():
    fake = _FakeOpenAI([_tool_call_response("STAY")])
    client = LMStudioClient(CFG, client=fake)
    client.decide("système", "perception")
    kwargs = fake.chat.completions.calls[0]
    assert kwargs["tools"] == [_MOVE_TOOL]
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "move"}}
    assert kwargs["temperature"] == CFG.temperature
    assert kwargs["max_tokens"] == CFG.max_tokens


def test_decide_retries_then_succeeds():
    fake = _FakeOpenAI([_no_tool_call_response(), _tool_call_response("EAST")])
    client = LMStudioClient(CFG, client=fake)
    result = client.decide("système", "perception")
    assert result.move == Move.EAST
    assert result.retries == 1
    assert result.fallback is False


def test_decide_falls_back_after_exhausting_retries():
    responses = [_no_tool_call_response(), RuntimeError("timeout"), _no_tool_call_response()]
    fake = _FakeOpenAI(responses)
    client = LMStudioClient(CFG, client=fake)
    result = client.decide("système", "perception")
    assert result.move == Move.STAY
    assert result.fallback is True
    assert result.retries == CFG.max_retries
