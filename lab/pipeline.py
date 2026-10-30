"""The RAG pipeline, assembled from a stage configuration.

Each stage turns on one more fix, so the eval scorecard shows exactly which fix
moved which question type.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import guards
from .ingest import Chunk, naive_chunks, structured_chunks
from .retrievers import DenseRetriever, HybridRetriever


@dataclass(frozen=True)
class Stage:
    key: str
    name: str
    fixes: str
    structured: bool = False
    hybrid: bool = False
    rewrite: bool = False
    filters: bool = False
    screen: bool = False
    top_k: int = 4


STAGES = [
    Stage("S0", "Naive baseline", "Raw text, 500 character chunks, dense search only, nothing else."),
    Stage("S1", "Structure-aware parsing and chunking",
          "Table rows keep their column headers; chunks follow headings; each chunk says which document and section it came from.",
          structured=True),
    Stage("S2", "Hybrid search", "BM25 keyword search fused with dense search (reciprocal rank fusion).",
          structured=True, hybrid=True),
    Stage("S3", "Query rewriting", "Glossary expansion (PTO to annual leave) and follow-up questions made standalone.",
          structured=True, hybrid=True, rewrite=True),
    Stage("S4", "Metadata filters", "Current versions only, and access control applied inside retrieval.",
          structured=True, hybrid=True, rewrite=True, filters=True),
    Stage("S5", "Ingestion screening", "Chunks carrying instructions aimed at the model are quarantined at ingestion.",
          structured=True, hybrid=True, rewrite=True, filters=True, screen=True),
]
STAGE_BY_KEY = {s.key: s for s in STAGES}


class Pipeline:
    def __init__(self, stage: Stage):
        self.stage = stage
        chunks = structured_chunks() if stage.structured else naive_chunks()
        self.quarantined: list[Chunk] = []
        if stage.screen:
            chunks, self.quarantined = guards.screen_for_injection(chunks)
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.retriever = HybridRetriever(texts) if stage.hybrid else DenseRetriever(texts)

    def rewrite(self, question: str, history: list[str]) -> str:
        if not self.stage.rewrite:
            # naive systems embed the follow-up as is, with no history
            return question
        return guards.expand_with_glossary(guards.condense_followup(question, history))

    def retrieve(self, question: str, history: list[str] | None = None,
                 user_roles: list[str] | None = None) -> tuple[str, list[Chunk]]:
        history, user_roles = history or [], user_roles or ["all_employees"]
        query = self.rewrite(question, history)
        scores = self.retriever.scores(query).astype(float)
        if self.stage.filters:
            # pre-filter: disallowed chunks can never rank, so top-k is always full of allowed chunks
            mask = np.array([guards.allowed(c, user_roles) for c in self.chunks])
            scores[~mask] = -np.inf
        order = [i for i in np.argsort(-scores) if np.isfinite(scores[i])][: self.stage.top_k]
        return query, [self.chunks[i] for i in order]
