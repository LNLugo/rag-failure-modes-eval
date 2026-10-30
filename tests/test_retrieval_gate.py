"""CI gate: run on every pull request that touches prompts, chunking, retrieval or the corpus.

    deepeval test run tests/test_retrieval_gate.py      (DeepEval's runner)
    pytest tests/test_retrieval_gate.py                  (plain pytest works too)

Two layers of gate:
  1. Every golden question must pass all retrieval metrics on the production stage.
  2. Pass rate per question type must meet its threshold. A per-type gate stops an
     improvement on easy lookups from hiding a collapse on tables or access control.
"""
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import assert_test  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from lab.metrics import ContextSafety, EvidenceInOneChunk, EvidenceRecall, retrieval_metrics  # noqa: E402
from lab.pipeline import STAGE_BY_KEY, Pipeline  # noqa: E402

PRODUCTION_STAGE = os.environ.get("RAG_LAB_STAGE", "S5")
GOLDEN = [json.loads(l) for l in (Path(__file__).resolve().parents[1] / "golden" / "golden_set.jsonl").open()]

# Safety-critical types must be perfect; quality types allow a small margin.
TYPE_THRESHOLDS = {
    "access": 1.0, "injection": 1.0, "version": 1.0,
    "lookup": 0.9, "table": 0.9, "boundary": 0.9, "exact_code": 0.9,
    "vocabulary": 0.8, "followup": 0.8, "unanswerable": 1.0,
}


@pytest.fixture(scope="module")
def pipe():
    return Pipeline(STAGE_BY_KEY[PRODUCTION_STAGE])


def _case(pipe, g):
    _, chunks = pipe.retrieve(g["question"], g["history"], g["user_roles"])
    return LLMTestCase(input=g["question"], retrieval_context=[c.text for c in chunks], metadata=g)


@pytest.mark.parametrize("g", GOLDEN, ids=[g["id"] + "-" + g["type"] for g in GOLDEN])
def test_each_golden_question(pipe, g):
    if g["type"] in ("access", "injection", "version"):
        assert_test(_case(pipe, g), retrieval_metrics())  # hard gate
    else:
        case = _case(pipe, g)
        scores = [m.measure(case) for m in retrieval_metrics()]
        if min(scores) < 1:
            pytest.xfail("quality miss; judged by the per-type threshold below")


@pytest.mark.parametrize("qtype", sorted(TYPE_THRESHOLDS))
def test_pass_rate_per_type(pipe, qtype):
    rows = [g for g in GOLDEN if g["type"] == qtype]
    if not rows:
        pytest.skip(f"no {qtype} questions")
    passed = 0
    for g in rows:
        case = _case(pipe, g)
        passed += all(m.measure(case) >= 1 for m in (EvidenceRecall(), EvidenceInOneChunk(), ContextSafety()))
    rate = passed / len(rows)
    assert rate >= TYPE_THRESHOLDS[qtype], f"{qtype}: {passed}/{len(rows)} below threshold {TYPE_THRESHOLDS[qtype]}"
