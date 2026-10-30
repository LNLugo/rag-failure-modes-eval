"""RAG break-and-fix lab runner.

  python run_lab.py                     score every retrieval stage S0..S5 with DeepEval
  python run_lab.py --show Q11          see what each stage retrieved for one question
  python run_lab.py --generation        also run the LLM half (needs ANTHROPIC_API_KEY)

Outputs go to results/: scorecard.md (read this), scorecard.json, per_question.csv.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
warnings.filterwarnings("ignore")

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from lab.metrics import retrieval_metrics  # noqa: E402
from lab.pipeline import STAGE_BY_KEY, STAGES, Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent
TYPE_ORDER = ["lookup", "table", "boundary", "exact_code", "vocabulary", "followup",
              "version", "access", "injection", "unanswerable"]


def load_golden() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "golden" / "golden_set.jsonl").open()]


def build_cases(pipe: Pipeline, golden: list[dict]) -> list[LLMTestCase]:
    cases = []
    for g in golden:
        query, chunks = pipe.retrieve(g["question"], g["history"], g["user_roles"])
        cases.append(LLMTestCase(
            name=g["id"],
            input=g["question"],
            expected_output=g["expected_answer"],
            retrieval_context=[c.text for c in chunks],
            metadata={**g, "rewritten_query": query, "chunk_ids": [c.chunk_id for c in chunks]},
            tags=[g["type"], pipe.stage.key],
        ))
    return cases


def run_deepeval(cases, metrics):
    """DeepEval runs the metrics; we keep its per-metric scores and reasons."""
    with contextlib.redirect_stdout(io.StringIO()):  # DeepEval's banner is noisy across 6 stages
        res = evaluate(
            test_cases=cases,
            metrics=metrics,
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False),
        )
    out = {}
    for tr in res.test_results:
        out[tr.name] = {
            "pass": bool(tr.success),
            "metrics": {m.name: {"score": m.score, "pass": m.success, "reason": m.reason} for m in tr.metrics_data or []},
        }
    return out


def show(qid: str, golden: list[dict]) -> None:
    g = next(x for x in golden if x["id"] == qid)
    print(f"\n{qid} [{g['type']}] {g['question']}")
    if g["history"]:
        print(f"  history: {g['history']}")
    print(f"  evidence needed: {g['evidence'] or '(none)'}   forbidden: {g['forbidden'] or '(none)'}\n")
    for st in STAGES:
        pipe = Pipeline(st)
        query, chunks = pipe.retrieve(g["question"], g["history"], g["user_roles"])
        case = LLMTestCase(input=g["question"], retrieval_context=[c.text for c in chunks], metadata=g)
        results = [(m.__name__, m.measure(case), m.reason) for m in retrieval_metrics()]
        verdict = "PASS" if all(s >= 1 for _, s, _ in results) else "FAIL"
        print(f"=== {st.key} {st.name}: {verdict}")
        if query != g["question"]:
            print(f"  rewritten query: {query}")
        for c in chunks:
            print(f"  - {c.chunk_id}: {' '.join(c.text.split())[:150]}")
        for name, s, reason in results:
            if s < 1:
                print(f"  ! {name}: {reason}")
        print()


def write_reports(all_results: dict, golden: list[dict], gen_results: dict | None) -> None:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    by_id = {g["id"]: g for g in golden}
    types = [t for t in TYPE_ORDER if any(g["type"] == t for g in golden)]
    lines = ["# RAG break-and-fix scorecard", "",
             "Retrieval evals run by DeepEval with three deterministic metrics: evidence recall@k, "
             "evidence in one chunk, and context safety. A question passes only if all three pass.", "",
             "| Stage | What it fixes | Pass | " + " | ".join(types) + " |",
             "|---|---|---|" + "---|" * len(types)]
    prev = None
    changes = []
    for st in STAGES:
        if st.key not in all_results:
            continue
        r = all_results[st.key]
        cells = []
        for t in types:
            ids = [g["id"] for g in golden if g["type"] == t]
            cells.append(f"{sum(r[i]['pass'] for i in ids)}/{len(ids)}")
        total = sum(v["pass"] for v in r.values())
        lines.append(f"| {st.key} {st.name} | {st.fixes} | **{total}/{len(r)}** | " + " | ".join(cells) + " |")
        if prev:
            fixed = [i for i in r if r[i]["pass"] and not prev[i]["pass"]]
            broke = [i for i in r if not r[i]["pass"] and prev[i]["pass"]]
            changes.append((st, fixed, broke))
        prev = r
    lines += ["", "Unanswerable questions have no evidence to retrieve, so retrieval metrics cannot fail them. "
              "They are tested in the generation half (`--generation`), where the model must say it cannot find the answer.",
              "", "## What each fix changed", ""]
    for st, fixed, broke in changes:
        lines.append(f"**{st.key} {st.name}**")
        lines.append(f"- Fixed: {', '.join(f'{i} ({by_id[i]['type']})' for i in fixed) or 'none'}")
        if broke:
            lines.append(f"- Regressed: {', '.join(f'{i} ({by_id[i]['type']})' for i in broke)}  <- a fix that breaks something else; check it")
        lines.append("")
    last = all_results[max(all_results)]
    failing = [i for i, v in last.items() if not v["pass"]]
    lines += ["## Still failing at the last stage run", ""]
    for i in failing:
        reasons = "; ".join(f"{m}: {d['reason']}" for m, d in last[i]["metrics"].items() if not d["pass"])
        lines.append(f"- {i} ({by_id[i]['type']}): {reasons}")
    if not failing:
        lines.append("- none")
    if gen_results:
        lines += ["", "## Generation evals (LLM judge)", "",
                  "| Setup | Pass | Correctness | Faithfulness | Answer relevancy | Answer safety |", "|---|---|---|---|---|---|"]
        for label, r in gen_results.items():
            def rate(m):
                vals = [v["metrics"][m]["pass"] for v in r.values() if m in v["metrics"]]
                return f"{sum(vals)}/{len(vals)}" if vals else "n/a"
            names = list(next(iter(r.values()))["metrics"].keys())
            corr = next((n for n in names if n.startswith("Correctness")), "Correctness")
            lines.append(f"| {label} | **{sum(v['pass'] for v in r.values())}/{len(r)}** | {rate(corr)} | {rate('Faithfulness')} | {rate('Answer Relevancy')} | {rate('Answer safety')} |")
    (out / "scorecard.md").write_text("\n".join(lines) + "\n")
    (out / "scorecard.json").write_text(json.dumps({"retrieval": all_results, "generation": gen_results}, indent=2, default=str))
    with (out / "per_question.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question"] + list(all_results))
        for g in golden:
            w.writerow([g["id"], g["type"], g["question"]] + ["PASS" if all_results[k][g["id"]]["pass"] else "FAIL" for k in all_results])
    print(f"\nWrote {out / 'scorecard.md'}")


def run_generation(golden: list[dict]) -> dict:
    from lab import generation as gen

    results = {}
    for label, stage_key, grounded in [("S0 retrieval + naive prompt", "S0", False),
                                       ("S5 retrieval + grounded prompt", "S5", True)]:
        pipe = Pipeline(STAGE_BY_KEY[stage_key])
        cases = []
        for g in golden:
            _, chunks = pipe.retrieve(g["question"], g["history"], g["user_roles"])
            q = g["question"] if not g["history"] else (gen.llm_condense(g["question"], g["history"]) if grounded else g["question"])
            cases.append(LLMTestCase(name=g["id"], input=q, actual_output=gen.generate(q, chunks, grounded),
                                     expected_output=g["expected_answer"], retrieval_context=[c.text for c in chunks],
                                     metadata=g, tags=[g["type"]]))
        print(f"Judging: {label}")
        results[label] = run_deepeval(cases, gen.generation_metrics())
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(s.key for s in STAGES))
    ap.add_argument("--show", help="question id to trace through every stage, e.g. Q11")
    ap.add_argument("--generation", action="store_true", help="also run generation evals (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()
    golden = load_golden()
    if args.show:
        show(args.show, golden)
        return
    all_results = {}
    for key in args.stages.split(","):
        st = STAGE_BY_KEY[key]
        r = run_deepeval(build_cases(Pipeline(st), golden), retrieval_metrics())
        all_results[key] = r
        print(f"{st.key} {st.name:<38} {sum(v['pass'] for v in r.values()):>2}/{len(r)} questions pass")
    gen_results = None
    if args.generation:
        from lab import generation as gen

        if not gen.available():
            print("\n--generation skipped: set ANTHROPIC_API_KEY to run the LLM half.")
        else:
            gen_results = run_generation(golden)
    write_reports(all_results, golden, gen_results)


if __name__ == "__main__":
    main()
