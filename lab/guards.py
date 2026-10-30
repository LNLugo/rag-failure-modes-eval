"""Query rewriting, metadata filters and ingestion-time injection screening."""
from __future__ import annotations

import json
import re

from .ingest import CORPUS, Chunk

# ------------------------------------------------------------ query rewriting

_GLOSSARY = {k: v for k, v in json.loads((CORPUS / "glossary.json").read_text())["terms"].items()}
_FOLLOWUP = re.compile(r"^(and|what about|how about|also|does it|is it|what if)\b", re.I)


def expand_with_glossary(query: str) -> str:
    """Append the corpus's own term for any user term we know. Appending (not
    replacing) keeps the user's words available to keyword search."""
    q = query.lower()
    extra = [canon for term, canon in _GLOSSARY.items() if re.search(rf"\b{re.escape(term)}\b", q)]
    return query if not extra else f"{query} ({'; '.join(dict.fromkeys(extra))})"


def condense_followup(query: str, history: list[str]) -> str:
    """Turn a follow-up into a standalone question.

    Offline stand-in: if the question looks like a follow-up, prepend the previous
    question. With an LLM available, lab.generation.llm_condense does this properly
    (rewrite 'And for Nonprofit Checking?' into a full question)."""
    if history and (_FOLLOWUP.search(query.strip()) or len(query.split()) <= 4):
        return f"{history[-1]} {query}"
    return query


# ------------------------------------------------------------ metadata filters

def allowed(chunk: Chunk, user_roles: list[str], current_only: bool = True) -> bool:
    if current_only and chunk.meta.get("status") != "current":
        return False
    return bool(set(chunk.meta.get("acl", [])) & set(user_roles))


# ------------------------------------------------- ingestion-time injection screen

_INJECTION = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"note to (ai|llm|assistant)s?",
    r"you are now",
    r"disregard (the|your) (system|previous)",
    r"system prompt",
]
_INJ = re.compile("|".join(_INJECTION), re.I)


def screen_for_injection(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    """Rules in code, not in the prompt. Quarantine chunks that carry instructions
    aimed at the model. Production systems add a classifier and stricter handling
    for documents marked trust=external."""
    keep, quarantined = [], []
    for c in chunks:
        (quarantined if _INJ.search(c.text) else keep).append(c)
    return keep, quarantined
