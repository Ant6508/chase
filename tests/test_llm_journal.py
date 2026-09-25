"""Tests du journal JSONL des épisodes (chase/llm/journal.py) : écriture au fil
de l'eau et reprise d'un run interrompu, sans réseau ni LLM."""

from __future__ import annotations

import threading

import pytest

from chase.llm.journal import EpisodeJournal

PARAMS = {"game": {"size": 15, "max_steps": 60, "spawn": (1, 2)}, "llm": {"max_tokens": 1600}}


def _record(seed: int) -> dict:
    return {"seed": seed, "captured": False, "steps": 60, "confinement": 0.25,
            "wall_time_s": 12.5, "stats": {"fallback_count": 0}}


def test_completed_is_empty_without_journal_file(tmp_path):
    journal = EpisodeJournal(tmp_path / "a1.jsonl", PARAMS)
    assert journal.completed() == {}


def test_appended_episode_is_read_back_by_seed(tmp_path):
    path = tmp_path / "a1.jsonl"
    EpisodeJournal(path, PARAMS).append(_record(3))

    done = EpisodeJournal(path, PARAMS).completed()

    assert list(done) == [3]
    assert done[3]["captured"] is False
    assert done[3]["confinement"] == 0.25


def test_params_with_tuples_match_after_json_round_trip(tmp_path):
    # un tuple de ChaseConfig redevient une liste une fois relu depuis le JSON :
    # la reprise ne doit pas le prendre pour un changement de paramètres
    path = tmp_path / "a1.jsonl"
    EpisodeJournal(path, PARAMS).append(_record(0))
    assert list(EpisodeJournal(path, PARAMS).completed()) == [0]


def test_resuming_with_other_params_is_refused(tmp_path):
    path = tmp_path / "a1.jsonl"
    EpisodeJournal(path, PARAMS).append(_record(0))
    other = {**PARAMS, "llm": {"max_tokens": 1300}}

    with pytest.raises(ValueError, match="paramètres"):
        EpisodeJournal(path, other).completed()


def test_line_truncated_by_a_kill_is_ignored_and_next_append_stays_readable(tmp_path):
    path = tmp_path / "a1.jsonl"
    journal = EpisodeJournal(path, PARAMS)
    journal.append(_record(0))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"seed": 1, "capt')  # écriture coupée net par un kill

    journal.append(_record(2))

    assert sorted(EpisodeJournal(path, PARAMS).completed()) == [0, 2]


def test_concurrent_appends_each_produce_one_valid_line(tmp_path):
    path = tmp_path / "a1.jsonl"
    journal = EpisodeJournal(path, PARAMS)
    threads = [threading.Thread(target=journal.append, args=(_record(s),)) for s in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(EpisodeJournal(path, PARAMS).completed()) == list(range(16))
