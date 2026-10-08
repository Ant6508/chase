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
    (_msg(candidates={"C1": float("nan")}), "candidates"),
    (_msg(candidates={"C1": float("inf")}), "candidates"),
    (_msg(intention="K1"), "intention"),
    (_msg(intention=[1]), "intention"),
    (_msg(moi="C1\ud83d"), "moi"),
    (_msg(candidates={"C1\ud83d": 7}), "candidates"),
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


def test_a_huge_integer_is_finite_and_accepted():
    assert validate(_msg(candidates={"C1": 10 ** 400})) is None


def test_a_surrogate_in_an_extra_key_is_rejected():
    problem = validate(_msg(**{"not\ud83d": 1}))
    assert problem == "le message contient une chaîne non encodable en UTF-8"


def test_schema_properties_carry_no_description():
    assert all("description" not in p for p in MESSAGE_SCHEMA["properties"].values())
    assert MESSAGE_SCHEMA["description"]


# --- bras A3 : spec de message par bras ------------------------------------------------

from chase.llm.message import A2_SPEC, A3_SPEC


def _a3(**changes) -> dict:
    msg = {"moi": "C2a.3", "cible": None, "intention": ["K1", "C4a"]}
    msg.update(changes)
    return msg


def test_a2_spec_is_the_frozen_a2_schema():
    assert A2_SPEC.fields == FIELDS and A2_SPEC.schema is MESSAGE_SCHEMA
    assert A2_SPEC.max_intention is None and not A2_SPEC.closed


def test_a3_schema_declares_three_fields_a_bound_and_no_extra_field():
    assert A3_SPEC.fields == ("moi", "cible", "intention")
    assert A3_SPEC.schema["required"] == list(A3_SPEC.fields)
    assert list(A3_SPEC.schema["properties"]) == list(A3_SPEC.fields)
    assert A3_SPEC.schema["properties"]["intention"]["maxItems"] == 3 == A3_SPEC.max_intention
    assert A3_SPEC.schema["additionalProperties"] is False and A3_SPEC.closed
    assert all("description" not in p for p in A3_SPEC.schema["properties"].values())


def test_valid_a3_messages_pass():
    assert validate(_a3(), A3_SPEC) is None
    assert validate(_a3(cible="C6b.3", intention=[]), A3_SPEC) is None
    assert validate(_a3(intention=["K1", "C4a", "C4b"]), A3_SPEC) is None


@pytest.mark.parametrize("bad, why", [
    (_a3(intention=["K1", "C4a", "C4b", "K3"]), "intention dépasse 3 lieux"),
    (_a3(candidates={"C7a": 7}), "en trop"),
    (_a3(je_couvre=None), "en trop"),
    ({"moi": "K1", "cible": None}, "manquant"),
    (_a3(moi=3), "moi"),
    (_a3(cible=3), "cible"),
    (_a3(intention="K1"), "intention"),
    (_a3(intention=[1]), "intention"),
    ("pas un objet", "objet"),
])
def test_invalid_a3_messages_say_why(bad, why):
    assert why in validate(bad, A3_SPEC)


def test_an_a2_message_is_not_a_valid_a3_message():
    assert validate(_msg(), A3_SPEC) == "champ(s) en trop : 'candidates', 'je_couvre'"


def test_an_extra_key_with_a_surrogate_gives_an_encodable_reason():
    problem = validate(_a3(**{"not\ud83d": 1}), A3_SPEC)
    problem.encode("utf-8")
    assert "en trop" in problem


def test_unknown_names_skip_the_fields_an_a3_message_lacks():
    places = SimpleNamespace(names=frozenset({"C2a.3", "K1"}))
    assert unknown_names(_a3(cible="Z1"), places) == ["Z1", "C4a"]
