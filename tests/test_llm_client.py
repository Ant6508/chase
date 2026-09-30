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
                        prompt_tokens: int = 10, completion_tokens: int = 5, **message_fields):
    call = SimpleNamespace(function=SimpleNamespace(
        arguments=json.dumps({"direction": direction, "reasoning": reasoning})))
    message = SimpleNamespace(tool_calls=[call], **message_fields)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                                                  completion_tokens=completion_tokens))


def _no_tool_call_response():
    message = SimpleNamespace(tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _tool_call_response_no_usage(direction: str, reasoning: str = "parce que"):
    """Appel d'outil valide sans statistiques d'usage (certains serveurs
    compatibles OpenAI, dont potentiellement LM Studio, omettent `usage`
    même sur une réponse bien formée)."""
    call = SimpleNamespace(function=SimpleNamespace(
        arguments=json.dumps({"direction": direction, "reasoning": reasoning})))
    message = SimpleNamespace(tool_calls=[call])
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
    assert kwargs["tool_choice"] == "required"
    assert kwargs["temperature"] == CFG.temperature
    assert kwargs["max_tokens"] == CFG.max_tokens


def test_decide_retries_then_succeeds():
    fake = _FakeOpenAI([_no_tool_call_response(), _tool_call_response("EAST")])
    client = LMStudioClient(CFG, client=fake)
    result = client.decide("système", "perception")
    assert result.move == Move.EAST
    assert result.retries == 1
    assert result.fallback is False


def test_decide_handles_missing_usage_on_success():
    fake = _FakeOpenAI([_tool_call_response_no_usage("SOUTH", "cible au sud")])
    client = LMStudioClient(CFG, client=fake)
    result = client.decide("système", "perception")
    assert result.move == Move.SOUTH
    assert result.fallback is False
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_decide_falls_back_after_exhausting_retries():
    responses = [_no_tool_call_response(), RuntimeError("timeout"), _no_tool_call_response()]
    fake = _FakeOpenAI(responses)
    client = LMStudioClient(CFG, client=fake)
    result = client.decide("système", "perception")
    assert result.move == Move.STAY
    assert result.fallback is True
    assert result.retries == CFG.max_retries


def test_decide_keeps_the_chain_of_thought_emitted_before_the_tool_call():
    thinking = "La cible est à l'ouest, le couloir ouest est praticable."
    fake = _FakeOpenAI([_tool_call_response("WEST", reasoning_content=thinking)])
    client = LMStudioClient(CFG, client=fake)
    result = client.decide("système", "perception")
    assert result.thinking == thinking


def test_decide_without_chain_of_thought_leaves_thinking_empty():
    fake = _FakeOpenAI([_tool_call_response("WEST")])
    client = LMStudioClient(CFG, client=fake)
    result = client.decide("système", "perception")
    assert result.thinking == ""


from chase.llm.message import MESSAGE_SCHEMA

_MSG = {"moi": "C2a.3", "cible": None, "candidates": {"C7a": 7}, "intention": ["K1"],
        "je_couvre": None}


def _args_response(args: dict):
    call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(args)))
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_message_parameter_is_declared_only_for_the_channel():
    with_msg = {"direction": "STAY", "reasoning": "r", "message": _MSG}
    fake = _FakeOpenAI([_tool_call_response("STAY"), _args_response(with_msg)])
    client = LMStudioClient(CFG, client=fake)
    client.decide("système", "perception")
    client.decide("système", "perception", with_message=True)
    without, with_ = (c["tools"][0]["function"]["parameters"] for c in fake.chat.completions.calls)
    assert "message" not in without["properties"] and "message" not in without["required"]
    assert with_["properties"]["message"] == MESSAGE_SCHEMA
    assert with_["required"] == ["direction", "reasoning", "message"]


def test_valid_message_and_raw_arguments_are_returned():
    args = {"direction": "EAST", "reasoning": "r", "message": _MSG}
    fake = _FakeOpenAI([_args_response(args)])
    result = LMStudioClient(CFG, client=fake).decide("s", "p", with_message=True)
    assert result.move == Move.EAST
    assert result.message == _MSG
    assert result.raw_arguments == json.dumps(args)


def test_invalid_message_is_retried_then_accepted():
    bad = {"direction": "EAST", "reasoning": "r", "message": {**_MSG, "intention": "K1"}}
    good = {"direction": "WEST", "reasoning": "r", "message": _MSG}
    fake = _FakeOpenAI([_args_response(bad), _args_response(good)])
    result = LMStudioClient(CFG, client=fake).decide("s", "p", with_message=True)
    assert (result.move, result.retries, result.message) == (Move.WEST, 1, _MSG)


def test_missing_message_exhausts_retries_and_falls_back_without_message():
    no_msg = {"direction": "EAST", "reasoning": "r"}
    fake = _FakeOpenAI([_args_response(no_msg)] * 3)
    result = LMStudioClient(CFG, client=fake).decide("s", "p", with_message=True)
    assert result.fallback is True and result.move == Move.STAY and result.message is None
    assert "message invalide" in result.reasoning


def test_message_is_ignored_outside_the_channel():
    args = {"direction": "EAST", "reasoning": "r", "message": "n'importe quoi"}
    fake = _FakeOpenAI([_args_response(args)])
    result = LMStudioClient(CFG, client=fake).decide("s", "p")
    assert result.move == Move.EAST and result.message is None and result.fallback is False


# --- durcissement avant les campagnes A2 / A1bis -------------------------------------

from chase.llm.client import move_tool


def _finish(response, reason):
    response.choices[0].finish_reason = reason
    return response


def test_move_tool_is_public_and_keeps_the_plain_tool_outside_the_channel():
    assert move_tool(False) is _MOVE_TOOL
    assert move_tool(True)["function"]["parameters"]["properties"]["message"] == MESSAGE_SCHEMA


def test_finish_reason_is_kept_on_success():
    fake = _FakeOpenAI([_finish(_tool_call_response("EAST"), "tool_calls")])
    result = LMStudioClient(CFG, client=fake).decide("s", "p")
    assert result.finish_reason == "tool_calls" and result.attempt_errors == []


def test_attempt_errors_are_kept_after_a_retry_and_name_the_finish_reason():
    cut = _finish(_no_tool_call_response(), "length")
    fake = _FakeOpenAI([cut, RuntimeError("timeout"), _tool_call_response("EAST")])
    result = LMStudioClient(CFG, client=fake).decide("s", "p")
    assert result.retries == 2
    assert result.attempt_errors == [
        "réponse sans appel d'outil valide (finish_reason=length)", "timeout"]


def test_fallback_keeps_all_errors_and_the_last_answer_received():
    bad = {"direction": "EAST", "reasoning": "r", "message": "non"}
    resp = _finish(_args_response(bad), "stop")
    resp.choices[0].message.reasoning_content = "je réfléchis"
    fake = _FakeOpenAI([_no_tool_call_response(), resp, RuntimeError("boum")])
    result = LMStudioClient(CFG, client=fake).decide("s", "p", with_message=True)
    assert result.fallback and result.move == Move.STAY and result.message is None
    assert len(result.attempt_errors) == 3
    assert result.thinking == "je réfléchis"
    assert result.raw_arguments == json.dumps(bad)
    assert result.finish_reason == "stop"


def test_exception_in_validation_becomes_a_failed_attempt(monkeypatch):
    import chase.llm.client as client_mod

    calls = []

    def boom(obj):
        calls.append(obj)
        if len(calls) == 1:
            raise OverflowError("int too large")
        return None

    monkeypatch.setattr(client_mod, "validate", boom)
    args = {"direction": "EAST", "reasoning": "r", "message": _MSG}
    fake = _FakeOpenAI([_args_response(args), _args_response(args)])
    result = LMStudioClient(CFG, client=fake).decide("s", "p", with_message=True)
    assert result.retries == 1 and not result.fallback
    assert result.attempt_errors[0].startswith("réponse illisible : OverflowError(")


def test_thinking_and_reasoning_are_made_encodable_but_the_message_is_not_touched():
    args = {"direction": "EAST", "reasoning": "abc\ud83d", "message": _MSG}
    resp = _args_response(args)
    resp.choices[0].message.reasoning_content = "pens\ud83d\u00e9e"
    fake = _FakeOpenAI([resp])
    result = LMStudioClient(CFG, client=fake).decide("s", "p", with_message=True)
    result.reasoning.encode("utf-8")
    result.thinking.encode("utf-8")
    assert result.thinking.endswith("\u00e9e")
    assert result.message == _MSG
