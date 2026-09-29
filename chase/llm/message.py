"""Message du canal A2 : schéma figé, validation, réécriture et comptage.

Le harnais ne lit jamais le contenu d'un message pour décider ou calculer quoi que
ce soit (croyance, perception). Il vérifie sa forme, le réécrit en JSON compact
pour le prompt du récepteur, compte ses positions et relève les noms inconnus.
Voir docs/superpowers/specs/2026-09-29-jalon2-a2-design.md, § Le message.
"""

from __future__ import annotations

import json
import math
import threading

TOKENIZER_REPO = "google/gemma-4-12B-it"  # tokenizer.json seul, sans les poids

FIELDS = ("moi", "cible", "candidates", "intention", "je_couvre")

MESSAGE_SCHEMA = {
    "type": "object",
    "description": "Message pour ton coéquipier, qu'il lira au pas suivant.",
    "properties": {
        "moi": {"type": "string", "description": "Ta case, par exemple \"C2a.3\"."},
        "cible": {"type": ["string", "null"],
                  "description": "La case de la cible si tu la vois, sinon null."},
        "candidates": {"type": "object", "additionalProperties": {"type": "number"},
                       "description": "Les lieux où la cible peut être d'après ta perception, "
                                      "avec leur probabilité en pourcentage entier."},
        "intention": {"type": "array", "items": {"type": "string"},
                      "description": "Les prochains lieux que tu comptes traverser, dans l'ordre."},
        "je_couvre": {"type": ["string", "null"],
                      "description": "Le lieu que tu bloques ou gardes, sinon null."},
    },
    "required": list(FIELDS),
}

_tokenizer = None
_lock = threading.Lock()


def _utf8(s: str) -> bool:
    """False si `s` contient un demi-caractère de substitution isolé (par ex. un
    emoji tronqué) : un texte pareil ne s'encode pas en UTF-8, et `count_tokens`
    lèverait une `TypeError` plus loin (de même que l'écriture de la trace)."""
    try:
        s.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def validate(obj) -> str | None:
    """None si le message a la bonne forme, sinon la raison. Les noms de lieux ne
    sont pas vérifiés ici : un nom inconnu est transmis tel quel (voir unknown_names)."""
    if not isinstance(obj, dict):
        return "le message n'est pas un objet"
    missing = [f for f in FIELDS if f not in obj]
    if missing:
        return f"champ(s) manquant(s) : {', '.join(missing)}"
    if not isinstance(obj["moi"], str):
        return "moi n'est pas une chaîne"
    if not _utf8(obj["moi"]):
        return "moi n'est pas un texte UTF-8 valide"
    for f in ("cible", "je_couvre"):
        if obj[f] is not None:
            if not isinstance(obj[f], str):
                return f"{f} n'est ni une chaîne ni null"
            if not _utf8(obj[f]):
                return f"{f} n'est pas un texte UTF-8 valide"
    cand = obj["candidates"]
    if not isinstance(cand, dict) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in cand.values()):
        return "candidates n'est pas un objet lieu -> nombre"
    if not all(_utf8(k) for k in cand):
        return "candidates contient une clé qui n'est pas un texte UTF-8 valide"
    if not isinstance(obj["intention"], list) or not all(isinstance(s, str) for s in obj["intention"]):
        return "intention n'est pas une liste de chaînes"
    if not all(_utf8(s) for s in obj["intention"]):
        return "intention contient une chaîne qui n'est pas un texte UTF-8 valide"
    return None


def render(msg: dict) -> str:
    """JSON compact, dans l'ordre des clés de l'émetteur : seuls les espaces changent."""
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":"))


def load_tokenizer():
    """Charge (une fois) et renvoie le tokenizer de gemma-4-12B-it. Public pour que
    l'appelant (scripts/run_llm.py) le charge une seule fois, avant les threads."""
    global _tokenizer
    with _lock:
        if _tokenizer is None:
            # Un antivirus ou un proxy peut intercepter TLS : le système reconnaît son
            # certificat, mais pas le paquet certifi utilisé par huggingface_hub. On
            # bascule donc httpx sur les certificats du système avant le téléchargement.
            import truststore
            truststore.inject_into_ssl()
            from tokenizers import Tokenizer
            _tokenizer = Tokenizer.from_pretrained(TOKENIZER_REPO)
    return _tokenizer


def count_tokens(text: str) -> int:
    """Positions occupées par `text` dans le contexte du modèle, sans token spécial."""
    if not text:
        return 0
    return len(load_tokenizer().encode(text, add_special_tokens=False).ids)


def unknown_names(msg: dict, places) -> list[str]:
    """Noms du message absents de la carte (`places.names`), dans l'ordre des champs."""
    names = [msg["moi"]]
    if msg["cible"] is not None:
        names.append(msg["cible"])
    names += list(msg["candidates"])
    names += msg["intention"]
    if msg["je_couvre"] is not None:
        names.append(msg["je_couvre"])
    return [n for n in names if n not in places.names]
