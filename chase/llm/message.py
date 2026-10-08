"""Message des bras à canal (A2, A2', A3) : schémas figés, validation, réécriture, comptage.

Le harnais ne lit jamais le contenu d'un message pour décider ou calculer quoi que
ce soit (croyance, perception). Il vérifie sa forme, le réécrit en JSON compact
pour le prompt du récepteur, compte ses positions et relève les noms inconnus.
Voir docs/superpowers/specs/2026-09-29-jalon2-a2-design.md, § Le message.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass

TOKENIZER_REPO = "google/gemma-4-12B-it"  # tokenizer.json seul, sans les poids

FIELDS = ("moi", "cible", "candidates", "intention", "je_couvre")

# Sans description par champ : le paragraphe du canal du prompt système les dit déjà,
# et le contexte par créneau est serré (~110 tokens par appel A2 économisés).
MESSAGE_SCHEMA = {
    "type": "object",
    "description": "Message pour ton coéquipier, qu'il lira au pas suivant.",
    "properties": {
        "moi": {"type": "string"},
        "cible": {"type": ["string", "null"]},
        "candidates": {"type": "object", "additionalProperties": {"type": "number"}},
        "intention": {"type": "array", "items": {"type": "string"}},
        "je_couvre": {"type": ["string", "null"]},
    },
    "required": list(FIELDS),
}


@dataclass(frozen=True, eq=False)
class MessageSpec:
    """Ce qu'un bras à canal attend du message : ses champs, dans l'ordre, le schéma
    déclaré dans l'outil `move`, la borne d'`intention` et l'interdiction des champs en
    trop. `validate` fait foi : le schéma n'est qu'annoncé au modèle, LM Studio ne
    l'impose pas forcément. Les specs sont des singletons (comparaison par identité)."""
    name: str
    fields: tuple[str, ...]
    schema: dict
    max_intention: int | None = None
    closed: bool = False


A2_SPEC = MessageSpec("A2", FIELDS, MESSAGE_SCHEMA)

# A3 : le budget est imposé par le schéma borné
# (docs/superpowers/specs/2026-10-08-jalon2-a3-design.md, § Le budget).
A3_FIELDS = ("moi", "cible", "intention")
A3_MAX_INTENTION = 3
A3_SCHEMA = {
    "type": "object",
    "description": "Message pour ton coéquipier, qu'il lira au pas suivant.",
    "properties": {
        "moi": {"type": "string"},
        "cible": {"type": ["string", "null"]},
        "intention": {"type": "array", "items": {"type": "string"},
                      "maxItems": A3_MAX_INTENTION},
    },
    "required": list(A3_FIELDS),
    "additionalProperties": False,
}
A3_SPEC = MessageSpec("A3", A3_FIELDS, A3_SCHEMA, max_intention=A3_MAX_INTENTION, closed=True)

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


def validate(obj, spec: MessageSpec = A2_SPEC) -> str | None:
    """None si le message a la bonne forme pour la spec du bras (A2 par défaut), sinon
    la raison. Les noms de lieux ne sont pas vérifiés ici : un nom inconnu est transmis
    tel quel (voir unknown_names)."""
    if not isinstance(obj, dict):
        return "le message n'est pas un objet"
    missing = [f for f in spec.fields if f not in obj]
    if missing:
        return f"champ(s) manquant(s) : {', '.join(missing)}"
    if spec.closed:
        extra = [k for k in obj if k not in spec.fields]
        if extra:
            # ascii() : une clé peut porter un demi-caractère de substitution, et la
            # raison finit dans la trace, écrite en UTF-8
            return f"champ(s) en trop : {', '.join(ascii(k) for k in extra)}"
    if not isinstance(obj["moi"], str):
        return "moi n'est pas une chaîne"
    if not _utf8(obj["moi"]):
        return "moi n'est pas un texte UTF-8 valide"
    for f in ("cible", "je_couvre"):
        if f in spec.fields and obj[f] is not None:
            if not isinstance(obj[f], str):
                return f"{f} n'est ni une chaîne ni null"
            if not _utf8(obj[f]):
                return f"{f} n'est pas un texte UTF-8 valide"
    if "candidates" in spec.fields:
        cand = obj["candidates"]
        if not isinstance(cand, dict) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                # isfinite sur les seuls float : un int géant le fait lever OverflowError
                and (not isinstance(v, float) or math.isfinite(v))
                for v in cand.values()):
            return "candidates n'est pas un objet lieu -> nombre"
        if not all(_utf8(k) for k in cand):
            return "candidates contient une clé qui n'est pas un texte UTF-8 valide"
    if not isinstance(obj["intention"], list) or not all(isinstance(s, str) for s in obj["intention"]):
        return "intention n'est pas une liste de chaînes"
    if not all(_utf8(s) for s in obj["intention"]):
        return "intention contient une chaîne qui n'est pas un texte UTF-8 valide"
    if spec.max_intention is not None and len(obj["intention"]) > spec.max_intention:
        return f"intention dépasse {spec.max_intention} lieux"
    # filet : champs en trop (A2), clés imbriquées, etc.
    try:
        render(obj).encode("utf-8")
    except (UnicodeEncodeError, ValueError, TypeError):
        return "le message contient une chaîne non encodable en UTF-8"
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
    """Noms du message absents de la carte (`places.names`), dans l'ordre des champs
    d'A2 ; les champs qu'un message n'a pas (A3) sont sautés."""
    names = [msg["moi"]]
    if msg.get("cible") is not None:
        names.append(msg["cible"])
    names += list(msg.get("candidates", {}))
    names += msg["intention"]
    if msg.get("je_couvre") is not None:
        names.append(msg["je_couvre"])
    return [n for n in names if n not in places.names]
