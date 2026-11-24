# RAG Break-and-Fix Lab

A small, runnable RAG system built to fail in the ways production RAG fails, then fixed one layer at a time, with DeepEval scoring every stage so each fix shows up as a number.

Companion repo: [rag-azure-enterprise-lab](https://github.com/LNLugo/rag-azure-enterprise-lab) builds each RAG technique on Azure; this lab shows how to diagnose RAG failures and gate them with evals.

The corpus is a fictional bank, Harbor Point Bank: ten policy documents and one PDF fee schedule. Each one has a failure planted in it.

## Quick start

```bash
pip install -r requirements.txt
python run_lab.py                 # scores stages S0 to S5, writes results/scorecard.md
python run_lab.py --show Q06      # trace one question through every stage
pytest tests -q                   # the CI gate (also: deepeval test run tests/)
RAG_LAB_STAGE=S0 pytest tests -q  # watch the gate block the naive pipeline
```

The retrieval half runs fully offline in about 10 seconds. For the generation half:

```bash
export ANTHROPIC_API_KEY=...
python run_lab.py --generation    # naive prompt on S0 vs grounded prompt on S5, judged by an LLM
```

## The six stages

The lab builds the same RAG pipeline six times. Each stage keeps every fix from the stage before it and turns on exactly one more, so when a question starts passing, the scorecard can name the fix that made the difference.

| Stage | Fix turned on | Layer | Code |
|---|---|---|---|
| **S0** Naive baseline | None: raw text, 500 character chunks, dense search only | (baseline) | `ingest.naive_chunks` |
| **S1** Structure-aware parsing and chunking | Table rows keep their column headers; chunks follow headings; each chunk is labeled with its document and section | Ingestion | `ingest.structured_chunks` |
| **S2** Hybrid search | BM25 keyword search fused with dense search (reciprocal rank fusion) | Retrieval | `retrievers.HybridRetriever` |
| **S3** Query rewriting | Glossary expansion (PTO to annual leave); follow-up questions made standalone | Query | `guards.expand_with_glossary`, `guards.condense_followup` |
| **S4** Metadata filters | Current versions only; access control applied inside retrieval | Governance | `guards.allowed` |
| **S5** Ingestion screening | Chunks carrying instructions aimed at the model are quarantined | Ingestion security | `guards.screen_for_injection` |

Each stage is a set of switches in `lab/pipeline.py` (S1 is `structured=True`, S2 adds `hybrid=True`, and so on).

**What "fixed at" means.** A break is fixed at the first stage where its questions pass for the right reason. The scorecard can only see when a question first passes, not why, so check the reason with `python run_lab.py --show <id>`. Two questions pass early by ranking luck: Q24 at S2 (its real fix is the S4 access filter) and Q28 at S1 (its real fix is the S3 rewrite). The table below lists the stage that actually fixes each break.

## The planted breaks and the stage that fixes each one

| Break | Where it starts | Questions | Fixed at | How |
|---|---|---|---|---|
| PDF table flattened, values cut off from their column headers | Parsing | Q06 to Q10 | S1 Structure-aware chunking | pdfplumber table extraction, one chunk per row with headers repeated |
| Rule and its exception split by a chunk boundary | Chunking | Q11, Q12 | S1 Structure-aware chunking | Chunk along headings, never mid-section |
| Ten near-identical table rows; dense search picks the wrong one | Retrieval | Q06, Q09 | S2 Hybrid search | Hybrid BM25 + dense with reciprocal rank fusion |
| Form codes (KYC-7B vs KYC-7C) blurred by embeddings | Retrieval | Q13 to Q16 | S1 then S2 | S1 gives each form its own chunk; S2's BM25 matches the exact code |
| User says PTO and WFH, policy says annual leave and remote work | Query | Q17 to Q19 | S3 Query rewriting | Glossary expansion from a business-owned glossary |
| Follow-up question has no subject ("What about the domestic one?") | Query | Q28, Q29 | S3 Query rewriting | Make the follow-up standalone using the conversation history |
| 2024 and 2025 versions of the travel policy both indexed | Corpus | Q20 to Q22 | S4 Metadata filters | Metadata filter: current versions only |
| Restricted executive compensation doc reachable by any employee | Access control | Q23 to Q25 | S4 Metadata filters | Access filter inside retrieval (pre-filter, not post-filter) |
| Vendor FAQ contains "ignore all previous instructions..." | Ingestion | Q26, Q27 | S5 Ingestion screening | Screen chunks at ingestion; quarantine instructions aimed at the model |
| Question the corpus cannot answer | Generation | Q30, Q31 | Generation half (`--generation`) | Grounded prompt with an explicit "I can't find this" path |

Q06 to Q10 need S1 first: without header-labeled rows, the right answer isn't in any chunk. Q06 and Q09 then also need S2, because ten nearly identical rows are hard for dense search to tell apart.

Result on the reference run: **S0 14/31, S1 20/31, S2 23/31, S3 25/31, S4 29/31, S5 31/31.**

## How the evals work

- **Golden set** (`golden/golden_set.jsonl`): 31 questions across 10 question types. Each has evidence spans (text that must reach the model), forbidden spans (text that must never reach it) and an expected answer.
- **Evidence spans, not chunk IDs.** Chunk IDs change whenever you change chunking, so a golden set labeled by chunk ID can't compare chunking strategies. Labeling the evidence text makes the golden set independent of how the corpus is chunked.
- **Retrieval metrics** (`lab/metrics.py`), written as DeepEval custom metrics. They are deterministic and cheap enough for every pull request:
  - *Evidence recall@k:* is every evidence span somewhere in the top k?
  - *Evidence in one chunk:* is all the evidence in a single chunk? This catches split clauses and table headers separated from their values.
  - *Context safety:* did any stale value, restricted data or injected instruction reach the model?
- **Generation metrics** (`lab/generation.py`), judged by an LLM through DeepEval: Correctness (G-Eval against the expected answer), Faithfulness, Answer Relevancy. There is also a deterministic Answer safety check that doesn't depend on the judge.
- **CI gate** (`tests/test_retrieval_gate.py`):
  - Access, injection and version questions are hard gates: every single one must pass.
  - Other question types are gated by pass rate per type, so gains on easy lookups can't hide a collapse on tables.

## Lessons the scorecard shows

1. **Most breaks surface at retrieval and start upstream.** S1 changes only parsing and chunking, not search, and still fixes 6 questions (5 for the right reason; see lesson 3).
2. **Better search can't fix a governance problem.** With every retrieval fix in place (S3) the system tops out at 25/31: stale versions, restricted data and injected text are still retrieved, because they really are the most relevant chunks. Only filters and screening (S4, S5) close those.
3. **A pass can be luck.** Q24 (access) and Q28 (follow-up) pass before their real fix lands, because of ranking luck. Only the S4 filter and S3 rewrite make them pass for the right reason. Use `--show` to check *why* a question passes, not just that it does.
4. **Unanswerable questions can't fail a retrieval eval.** There's nothing to retrieve. Only the generation half tests whether the system admits it doesn't know.
5. **Rules in code beat rules in prompts.** The access filter and injection screen run before the model sees anything.

## Exercises (break it yourself)

- **Overlap:** set `overlap=100` in `naive_chunks`. Does anything improve? (2026 benchmarks say usually not.)
- **Chunk size:** change the 500-character naive chunk size to 300 and 1,000 and watch which question types move.
- **Remove a fix:** set `hybrid=False` on S5 and see which gate fails first.
- **Add a reranker:** add a cross-encoder stage (sentence-transformers `CrossEncoder`, for example bge-reranker) between retrieval and top-k. Score it as S6.
- **Use real embeddings:** `pip install sentence-transformers`, then `RAG_LAB_EMBEDDER=st:BAAI/bge-small-en-v1.5 python run_lab.py`. Compare against the offline numbers.
- **Plant your own break:** add a third version of the travel policy with no `status` field. What does the filter do with missing metadata?
- **Add tracing:** wrap `Pipeline.retrieve` with Langfuse or MLflow tracing and inspect one failing trace.

## Honest notes on the offline stand-ins

- **Dense search** defaults to LSA (TF-IDF + SVD, 24 dimensions) so the lab runs without downloading a model. Its tokenizer deliberately ignores tokens containing digits, to imitate how embedding models blur identifiers like KYC-7B. Run with a real embedding model to see real-world magnitudes; the pattern holds, but the exact numbers will differ.
- **Follow-up rewriting** offline is a heuristic: it prepends the previous question. `lab/generation.llm_condense` is the production version, and is used in `--generation`.
- **The injection screen** is a pattern list. Production systems add a classifier and stricter handling for documents marked `trust: external`.
- **The judge** defaults to a larger Claude model than the generator. Best practice is a judge from a different model family; swap it via `RAG_LAB_JUDGE` or a DeepEval model class.

## Layout

```
corpus/      documents, manifest.json (status, acl, trust), glossary.json
golden/      golden_set.jsonl
lab/         ingest.py, retrievers.py, guards.py, pipeline.py, metrics.py, generation.py
tests/       test_retrieval_gate.py (CI gate)
scripts/     make_fee_schedule_pdf.py
results/     scorecard.md, scorecard.json, per_question.csv
run_lab.py   runner
```
