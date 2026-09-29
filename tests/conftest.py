"""Configuration commune des tests : aucun test ne télécharge le tokenizer de gemma-4."""

from __future__ import annotations

import pytest

from chase.llm import message


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class CharTokenizer:
    """Un token par caractère : déterministe et hors ligne."""

    def encode(self, text, add_special_tokens=True):
        return _Encoding(list(text))


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    monkeypatch.setattr(message, "_tokenizer", CharTokenizer())
