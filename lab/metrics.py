"""Retrieval metrics as DeepEval custom metrics.

Why evidence spans instead of chunk IDs: chunk IDs change every time you change the
chunking strategy, so a golden set labeled with chunk IDs is useless for comparing
chunking strategies. Labeling the evidence text itself makes the golden set
independent of chunking, which is the point of this lab.

All three metrics are deterministic (no LLM judge), so they are cheap enough to run
on every pull request.
"""
from __future__ import annotations

import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _contains(chunk: str, span: str) -> bool:
    return _norm(span) in _norm(chunk)


class _SpanMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.evaluation_model = None

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    def _finish(self, score: float, reason: str) -> float:
        self.score, self.reason = score, reason
        self.success = score >= self.threshold
        return score


class EvidenceRecall(_SpanMetric):
    """Share of required evidence spans found anywhere in the retrieved chunks."""

    @property
    def __name__(self):
        return "Evidence recall@k"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ev = test_case.metadata.get("evidence", [])
        if not ev:
            return self._finish(1.0, "No evidence required (unanswerable or access-denied case).")
        ctx = test_case.retrieval_context or []
        found = [e for e in ev if any(_contains(c, e) for c in ctx)]
        missing = [e for e in ev if e not in found]
        return self._finish(len(found) / len(ev), "All evidence retrieved." if not missing else f"Missing: {missing}")


class EvidenceInOneChunk(_SpanMetric):
    """1 if a single retrieved chunk holds all the evidence. Catches answers split
    across a chunk boundary and table values separated from their column header."""

    @property
    def __name__(self):
        return "Evidence in one chunk"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ev = test_case.metadata.get("evidence", [])
        if len(ev) < 2:
            return self._finish(1.0, "Fewer than two evidence spans; not applicable.")
        ok = any(all(_contains(c, e) for e in ev) for c in test_case.retrieval_context or [])
        return self._finish(1.0 if ok else 0.0, "Evidence together in one chunk." if ok else "Evidence split across chunks or missing.")


class ContextSafety(_SpanMetric):
    """0 if any forbidden span reaches the model: a superseded value, restricted
    data the user may not see, or an injected instruction."""

    @property
    def __name__(self):
        return "Context safety"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        bad = [f for f in test_case.metadata.get("forbidden", [])
               if any(_contains(c, f) for c in test_case.retrieval_context or [])]
        return self._finish(0.0 if bad else 1.0, "Clean context." if not bad else f"Forbidden content retrieved: {bad}")


def retrieval_metrics() -> list[BaseMetric]:
    return [EvidenceRecall(), EvidenceInOneChunk(), ContextSafety()]
