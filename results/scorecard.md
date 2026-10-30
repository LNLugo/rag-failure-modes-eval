# RAG break-and-fix scorecard

Retrieval evals run by DeepEval with three deterministic metrics: evidence recall@k, evidence in one chunk, and context safety. A question passes only if all three pass.

| Stage | What it fixes | Pass | lookup | table | boundary | exact_code | vocabulary | followup | version | access | injection | unanswerable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 Naive baseline | Raw text, 500 character chunks, dense search only, nothing else. | **14/31** | 5/5 | 1/5 | 0/2 | 3/4 | 2/3 | 0/2 | 0/3 | 1/3 | 0/2 | 2/2 |
| S1 Structure-aware parsing and chunking | Table rows keep their column headers; chunks follow headings; each chunk says which document and section it came from. | **20/31** | 5/5 | 3/5 | 2/2 | 4/4 | 2/3 | 1/2 | 0/3 | 1/3 | 0/2 | 2/2 |
| S2 Hybrid search | BM25 keyword search fused with dense search (reciprocal rank fusion). | **23/31** | 5/5 | 5/5 | 2/2 | 4/4 | 2/3 | 1/2 | 0/3 | 2/3 | 0/2 | 2/2 |
| S3 Query rewriting | Glossary expansion (PTO to annual leave) and follow-up questions made standalone. | **25/31** | 5/5 | 5/5 | 2/2 | 4/4 | 3/3 | 2/2 | 0/3 | 2/3 | 0/2 | 2/2 |
| S4 Metadata filters | Current versions only, and access control applied inside retrieval. | **29/31** | 5/5 | 5/5 | 2/2 | 4/4 | 3/3 | 2/2 | 3/3 | 3/3 | 0/2 | 2/2 |
| S5 Ingestion screening | Chunks carrying instructions aimed at the model are quarantined at ingestion. | **31/31** | 5/5 | 5/5 | 2/2 | 4/4 | 3/3 | 2/2 | 3/3 | 3/3 | 2/2 | 2/2 |

Unanswerable questions have no evidence to retrieve, so retrieval metrics cannot fail them. They are tested in the generation half (`--generation`), where the model must say it cannot find the answer.

## What each fix changed

**S1 Structure-aware parsing and chunking**
- Fixed: Q07 (table), Q10 (table), Q11 (boundary), Q12 (boundary), Q15 (exact_code), Q28 (followup)

**S2 Hybrid search**
- Fixed: Q06 (table), Q09 (table), Q24 (access)

**S3 Query rewriting**
- Fixed: Q18 (vocabulary), Q29 (followup)

**S4 Metadata filters**
- Fixed: Q20 (version), Q21 (version), Q22 (version), Q23 (access)

**S5 Ingestion screening**
- Fixed: Q26 (injection), Q27 (injection)

## Still failing at the last stage run

- none
