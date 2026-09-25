"""Journal JSONL des épisodes d'un run LLM : une ligne par épisode, écrite (et
forcée sur disque) dès que l'épisode se termine. Un run interrompu (kill,
coupure réseau, pod arrêté) ne perd ainsi que les épisodes en cours, et une
relance avec le même journal reprend aux seeds manquantes.

Chaque ligne porte les paramètres du run (jeu + LLM) : reprendre un journal
écrit avec d'autres paramètres est refusé, pour ne jamais mélanger dans un
même tableau des épisodes qui ne sont pas comparables.
"""

from __future__ import annotations

import json
import os
import threading


class EpisodeJournal:
    def __init__(self, path, params: dict):
        self.path = path
        # aller-retour JSON : les tuples deviennent des listes, comme à la relecture
        self.params = json.loads(json.dumps(params))
        self._lock = threading.Lock()  # les épisodes parallèles écrivent depuis leurs threads

    def completed(self) -> dict[int, dict]:
        """Épisodes déjà journalisés, indexés par seed."""
        if not os.path.exists(self.path):
            return {}
        done = {}
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # ligne coupée par un kill en pleine écriture : épisode à refaire
                if record["params"] != self.params:
                    raise ValueError(
                        f"{self.path} a été écrit avec d'autres paramètres que ce run "
                        f"(seed {record['seed']}) : changer de journal plutôt que mélanger")
                done[record["seed"]] = record
        return done

    def append(self, record: dict) -> None:
        line = (json.dumps({**record, "params": self.params}, ensure_ascii=False) + "\n").encode()
        # binaire : pas de traduction \n -> \r\n sous Windows, et seek fiable en fin de fichier
        with self._lock, open(self.path, "ab+") as f:
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    line = b"\n" + line  # isole une ligne coupée laissée par un kill
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
