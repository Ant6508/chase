"""Tests du message A2 (chase/llm/message.py). Le tokenizer est remplacé par un
faux, un token par caractère (tests/conftest.py) : aucun téléchargement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chase.llm.message import FIELDS, MESSAGE_SCHEMA, count_tokens, render, unknown_names, validate


def _msg(**changes) -> dict:
    msg = {"moi": "C2a.3", "cible": None, "candidates": {"C7a": 7, "C9": 6},
           "intention": ["K1", "C4a"], "je_couvre": None}
    msg.update(changes)
    return msg


def test_schema_requires_the_five_fields():
    assert MESSAGE_SCHEMA["required"] == list(FIELDS)
    assert list(MESSAGE_SCHEMA["properties"]) == list(FIELDS)


def test_valid_messages_pass():
    assert validate(_msg()) is None
    assert validate(_msg(cible="C6b.3", je_couvre="K5", candidates={"C6b": 100.0})) is None
    assert validate(_msg(note="champ en plus : transmis tel quel")) is None


@pytest.mark.parametrize("bad, field", [
    ("pas un objet", "objet"),
    ({"moi": "K1"}, "manquant"),
    (_msg(moi=3), "moi"),
    (_msg(cible=3), "cible"),
    (_msg(je_couvre=["K1"]), "je_couvre"),
    (_msg(candidates=["C1"]), "candidates"),
    (_msg(candidates={"C1": "7"}), "candidates"),
    (_msg(candidates={"C1": True}), "candidates"),
    (_msg(intention="K1"), "intention"),
    (_msg(intention=[1]), "intention"),
])
def test_invalid_messages_say_why(bad, field):
    assert field in validate(bad)


def test_render_is_compact_and_keeps_the_writer_key_order():
    assert render(_msg()) == ('{"moi":"C2a.3","cible":null,"candidates":{"C7a":7,"C9":6},'
                              '"intention":["K1","C4a"],"je_couvre":null}')
    assert render({"cible": None, "moi": "é"}) == '{"cible":null,"moi":"é"}'


def test_count_tokens_uses_the_tokenizer_without_special_tokens():
    assert count_tokens("abc") == 3   # faux tokenizer : un token par caractère
    assert count_tokens("") == 0


def test_unknown_names_are_listed_in_message_order():
    places = SimpleNamespace(names=frozenset({"C2a.3", "K1", "C4a", "C7a"}))
    assert unknown_names(_msg(), places) == ["C9"]
    assert unknown_names(_msg(cible="Z1", je_couvre="Z2"), places) == ["Z1", "C9", "Z2"]
