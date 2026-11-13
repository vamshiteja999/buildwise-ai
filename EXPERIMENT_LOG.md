# BuildWise AI: Experiment Log

Local-first RAG over construction documents (OSHA fall-protection rules, Florida Building Code accessibility chapter, Boca Raton permit forms, a county fire-station project package).
Everything runs on a MacBook M1 with 8 GB RAM: Ollama (`qwen3:1.7b` answers, `nomic-embed-text` embeddings), Qdrant (local mode), hybrid search (vector + keyword, reciprocal rank fusion).

Rule followed throughout: **measure one change at a time, accuracy first, speed second.**

---

## 1. How we measure

71 hand-checked questions (`golden_questions.csv`): 31 easy, 35 multi-part, 5 unanswerable (the system must refuse). Every expected answer was verified against the source PDFs.

| Metric | Meaning |
|---|---|
| Hit rate@5 | The expected document appears in the top 5 retrieved chunks |
| MRR | How high the expected document ranks (1.0 = always first) |
| Keyword accuracy | All required answer keywords appear in the answer |
| Citation present | The answer carries a `[n]` citation |
| Context recall | Required keywords appear in the retrieved text (separates retrieval from generation failures) |
| False refusal | System says "not found" for an answerable question |
| Correct refusal | System refuses an unanswerable question |
| Median latency | Retrieval + answer, seconds |

Failure buckets split every miss into **retrieval** (the needed text was never retrieved) or **generation** (the text was retrieved but the answer missed it).

Run-to-run noise: re-running identical code changed the unanswerable score by 1 of 5 and keyword accuracy by 2 to 3 questions. Treat differences of 2 to 3 questions as noise.

---

## 2. Changes, in order

| # | Change | Where | Why |
|---|---|---|---|
| 1 | Clause-aware chunking for OSHA regulations, with parent-clause text and a context prefix for embeddings; page-sized chunks replaced | `core.py` | Keep each rule with its numbered paragraph |
| 2 | Embedding prefixes (`search_document:` / `search_query:`) for nomic | `core.py` | Model's recommended usage |
| 3 | Qwen3 thinking off (`think: false`), `keep_alive 30m`, `num_ctx 8192`, `num_predict 450` | `core.py` | Answers in about 10 s instead of 25 to 40 s |
| 4 | Golden set grown from 10 to 71 questions, with `answer_keywords`, source stems, multi-source support | `golden_questions.csv` | Earlier set was too small and too easy |
| 5 | Evaluation redesigned: answerable-only metrics, context recall, failure buckets, refusal classifier, per-type table | `eval.py` | Show where failures come from |
| 6 | `diag.py`: per-document chunk diagnostics and chunk dump | `diag.py` | Inspect what the model actually sees |
| 7 | Document type from the file name: a file is an OSHA regulation only if its name starts with a standard number such as `1926.502` | `core.py` | OSHA3146 booklet was misclassified |
| 8 | Section label: "first through last" instead of a long comma list | `core.py` | Readable citations |
| 9 | `PROMPT_V2` flag (off by default) | `core.py`, `config.py` | See section 4 |
| 10 | `CITATION_BOOST` flag (off by default, **not yet tested**) | `core.py`, `config.py` | Rank chunks whose file or section matches a number in the question |
| 11 | `EXTRACT_V2` flag (off by default) | `core.py`, `config.py` | See section 5 |
| 12 | Eval matching ignores comma spacing (`A, B` equals `A,B`); two golden keywords tightened to `a,b,c` and `a,b,d` | `eval.py`, `golden_questions.csv` | Two roofing rows could never match |
| 13 | Final configuration made the default: `PROMPT_V2=true`, `EXTRACT_V2=true`, collection `bw_c` | `config.py` | Run C is the best evaluated setup |
| 14 | DOCX reading with the standard library (headings become sections, tables become ` | ` rows); `.txt` and `.md` also supported | `core.py`, `config.py` | Admin uploads of Word specs |
| 15 | Streamlit app: Chat (sources and latency), Admin (password, upload, replace, remove), Evaluation page | `app.py` | Demo for companies; no metrics change expected (UI and ingestion only) |
| 16 | Repeated header/footer removal ignoring page numbers (`_drop_repeated_edge_lines`, under `EXTRACT_V2`) | `core.py` | BD-002 footer "BD-002 Project Submission Page N of 16" survived as footer-only chunks that outranked real content |
| 17 | `KEYWORD_V2`: BM25 keyword search (stopwords ignored, rare words weigh more, length-normalised) replaces raw word counts | `core.py`, `config.py` | Raw counts favoured long chunks full of "the/is/of" |
| 18 | CSV reading (each row becomes `Column: value \| Column: value`, 50 rows per section); sample door schedule | `core.py`, `config.py`, `app.py` | Schedules and logs are often spreadsheets |
| 19 | `GOLDEN_CSV` / `EVAL_REPORT` settable from the environment; `golden_bd002.csv` (17 questions for the BD-002 guideline); `debug_refusal.py`, `reindex_one.py` | `config.py`, tools | Test new documents without touching the 71-question baseline |
| 20 | `VERIFY` flag (off by default): after a YES answer, (a) refuse if the first sentence says the fact is "not stated/specified/provided", (b) one short model call checks that the cited chunks state the fact (SUPPORTED / UNSUPPORTED; fails open) | `core.py`, `config.py` | Run D: correct refusals fell to 2 of 5. Offline check: step (a) fires on 2 of the 3 missed refusals and on none of the answerable excerpts in run D |
| 21 | `QUERY_UNDERSTANDING` (on by default, app only): greetings/thanks answered without search; follow-ups rewritten from the last 4 turns; several questions split and answered separately (citations numbered `[part.n]`); exclusions ("don't give me X") rewritten. Clear single questions pass through with no model call, so `eval.py` (single turns) is unaffected. Fails open to the original question | `core.py` (`ask`, `understand_query`), `app.py`, `config.py` | Follow-ups, small talk, pasted multi-questions and exclusions all failed before. Not measured by the eval; check with `python try_queries.py` |
| 22 | Automated tests (`tests/`, 21 tests): readers, footer removal, BM25, answer check, query understanding | `tests/` | No test suite before. Writing them found 3 bugs, fixed: "help me find the permit fee" was treated as small talk; the BM25 token cache was keyed by chunk id only, and ids are stable across re-indexing, so a long-running app could serve stale text after a document was replaced (now keyed by id and text); header/footer removal could drop long content lines that differ only by digits (now only lines up to 80 characters) |

Backups of replaced files: `*.bak`, `core.py.before_extract_v2`, `core.py.before_sections`.

---

## 3. Results

All 71-question runs, `qwen3:1.7b` unless noted.

| Run | Hit@5 | MRR | Keyword acc. | Citation | Ctx recall | False refusal | Correct refusal | Latency | Fully correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (original prompt) | 0.924 | 0.783 | 0.621 | 0.712 | 0.813 | 13.6% | 5/5 | 11.3 s | 41 |
| Prompt V2a (answer-first, no citation rule) | 0.924 | 0.783 | 0.697 | 0.530 | 0.813 | 7.6% | 3/5 | 12.0 s | 46 |
| Prompt V2b (cite every sentence) | 0.924 | 0.783 | 0.682 | 0.939 | 0.813 | 7.6% | 1/5 | 10.9 s | 45 |
| V2b rerun (identical code) | 0.924 | 0.783 | 0.682 | 0.939 | 0.813 | 7.6% | 2/5 | 10.9 s | 45 |
| `qwen3:4b` (reasoning leaked into answers; **invalid**) | 0.879 | 0.779 | 0.712* | 1.000 | 0.775 | 0% | 0/5 | 44.8 s | 47* |
| First extraction rewrite | 0.879 | 0.779 | 0.636 | 0.955 | 0.775 | 4.5% | 3/5 | 10.6 s | 42 |
| **A**: gate prompt, no extraction changes | 0.909 | 0.785 | 0.727 | 0.924 | 0.798 | 4.5% | 4/5 | 11.0 s | 48 |
| **B**: gate prompt + first extraction | 0.894 | 0.770 | 0.712 | 0.939 | 0.790 | 3.0% | 3/5 | 13.1 s | 47 |
| **C**: gate prompt + final extraction (**chosen**) | **0.924** | **0.795** | 0.697 | **0.955** | **0.866** | **0%** | 4/5 | 12.1 s | 46 |
| **D**: run C + footer-repeat fix + BM25 keyword search (2026-11-12, 12 documents incl. BD-002 guideline) | **1.000** | **0.852** | 0.712 | **0.985** | **0.927** | 0% | 2/5 | 13.5 s | 47 |
| **E**: run D + `VERIFY` (answer check) | 1.000 | 0.852 | 0.712 | 0.970 | 0.927 | 1.5% | **5/5** | 23.4 s | 47 |

\* The 4B run is not comparable: it wrote its reasoning into the answer, so keywords were found in reasoning text.
Runs A and B were scored with the older eval (strict comma spacing); C used the corrected eval. Only two roofing rows are affected.

Failure buckets (answerable questions):

| Run | Correct | Generation failure | Retrieval failure |
|---|---:|---:|---:|
| Baseline | 41 | 13 | 12 |
| A | 48 | 6 | 12 |
| B | 47 | 7 | 12 |
| C | 46 | 10 | 10 |
| D | 47 | 15 | 4 |
| E | 47 | 15 | 4 |

Earliest run, 10 questions on page-sized chunks (not comparable): hit@5 0.70, MRR 0.49, citation 0.60, median latency 27.8 s.

---

## 4. Prompt experiments

- **V2a** told the model to answer first, briefly, quoting numbers. False refusals fell from 13.6% to 7.6% and accuracy rose, but citations fell from 0.71 to 0.53.
- **V2b** required a `[n]` citation on every sentence. Citations rose to 0.94, but the model stopped refusing off-topic questions (1 to 2 of 5 correct refusals).
- **Gate** (`ANSWERABLE: YES/NO` as the first line; NO becomes the refusal sentence; the line is stripped otherwise). Correct refusals recovered to 4/5 in runs A and C.
- A `ANSWERABLE: YES` line occasionally leaked into an answer in run B; small and not yet handled.
- **Not done:** `qwen3:4b`. `think: false` was not honored, answers began with "First, I need to understand…", and latency was about 45 s. Revisit with a non-thinking build on a 16 GB machine.

---

## 5. Extraction experiments (`EXTRACT_V2`)

Problem: PyMuPDF returned table cells and drawing labels in the wrong order.

What changed:
1. **Row rebuild.** Words on the same visual line are joined, with a wide gap shown as ` | `. `Metal Roofs | A,B,C | 1,2,3,4,5` now stays together. Applied to table-like pages and large drawing sheets; ordinary prose pages are unchanged.
2. **Drawing sheets split into sections** (keynote legend, title block, annotations), so each is its own chunk. Before, the whole Station 25 sheet was one 392-word chunk and the keynote questions never ranked.
3. **Edition lines kept.** "Florida Building Code, Eighth Edition (2023)" was deleted as a repeated page footer.
4. **Noise.** Print headers merged into rebuilt rows are removed; "Square" and "Dash bullet" PDF bullet labels are stripped.

A bug found along the way: the first rewrite merged browser print headers into pipe-joined lines that the noise filter missed. Fixed in the final version.

Result: questions about the keynote legend, wall finishes, and the title block now pass; retrieval failures fell from 12 to 10 and context recall rose to 0.866.

---

## 6. Findings

- A duplicate Universal Permit file is skipped by the content-hash check (10 of 11 files indexed) by design; no files are lost.
- Two roofing rows scored 0% context recall only because the golden keyword had a space (`a, b`) that the table text lacks.
- A 1.7B model cannot reliably tell "related topic" from "answers the question".

Run D note: the index grew from 10 to 12 documents (BD-002 guideline added), so it is not strictly the same corpus as C. Retrieval clearly improved (hit@5 0.924 to 1.000, retrieval failures 10 to 4). Fully correct answers barely moved (46 to 47) because fixed retrieval failures became generation failures (10 to 15). Correct refusals fell from 4/5 to 2/5: better retrieval gives the model more related text, so it hedges ("not explicitly stated") or answers from nearby text instead of refusing. The three misses were contract value, Subpart L guardrail height and Boca Raton wind speed. With only 5 refusal questions this swings by 1 to 2 between identical runs, so treat it as a warning, not a proven regression. Latency rose about 1.4 s.

Run E (VERIFY on): correct refusals 2/5 to 5/5, including the three that run D missed (contract value, Subpart L guardrail height, wind speed). Cost: one false refusal ("wall and fascia finishes in the Station 25 keynote legend", which the answer step had got right), citations 0.985 to 0.970, and median latency 13.5 s to 23.4 s (the check adds about 10 s to every answered question; refusals are not slowed). Fully correct answers unchanged at 47. Decision: VERIFY is on by default (accuracy first); `VERIFY=false` restores the faster path. Possible speed fix not yet tried: send the checker a shorter excerpt of the cited chunk.

BD-002 set (`golden_bd002.csv`, run with VERIFY): 12 of 15 answerable correct, 2 of 2 refusals correct, 0 false refusals, context recall 100%. Misses: SD vs DD geotechnical report, Shell Only vs Foundation Only (phase lists mixed up), Food Service Plan sheet number. The hit-rate 0.000 in that report is a test bug, not a retrieval failure: the CSV's `expected_source` used underscores while the indexed file name has spaces. Fixed in the CSV; ignore hit rate/MRR from that run.

---

## 7. Known limitations (for the README)

- Subpart L scaffold guardrail question (and sometimes the OSHA penalty and permit fee questions): the model answers from 1926.502 or says the fee is in a schedule. This is a small-model limit.
- Scanned and drawing text is still imperfect (rotated copyright text is noise).
- Some OSHA wording questions fail on retrieval (for example "wells, pits and shafts"); the footer edition-awareness questions are low-value.
- Latency is about 12 s per answer on an 8 GB laptop.
- Exclusion queries ("don't give me X") are not supported: retrieval matches the words of X and pulls that section. Ask positively instead.
- Several questions pasted as one message are treated as a single question and answered badly. Ask one question at a time.
- BM25 keyword search scans every chunk on each query (fast at a few hundred chunks); at tens of thousands of chunks it should move to a precomputed index.
- Reranking is off (`RERANK_ENABLED=false`). The built-in rerank is a word-count proxy and is not recommended; BM25 (`KEYWORD_V2`) is the retrieval improvement in use. A cross-encoder reranker needs a large install; try it on a 16 GB machine only if the eval shows retrieval misses BM25 did not fix.

## 8. Final configuration

```
ANSWER_MODEL=qwen3:1.7b   EMBED_MODEL=nomic-embed-text
PROMPT_V2=1               EXTRACT_V2=1
KEYWORD_V2=1              VERIFY=1
QDRANT_COLLECTION=bw_c
```

Re-ingest into a new collection whenever extraction or chunking changes.

## 8b. Case study: two false refusals on a new PDF (BD-002 guideline)

"What is the revision date of BD-002 and who approved it?" and "What is the minimum font size for text on plans?" were refused although both facts were in the PDF. `debug_refusal.py` showed both were retrieval failures (answer chunk ranked 19 and 11; the model sees 5):

| Step | Revision-date rank | Font-size rank | Model reply |
|---|---:|---:|---|
| Before | 19 | 11 | ANSWERABLE: NO (both) |
| Footer fix only (change 16) | 5 | 11 | NO (both) |
| + BM25 keyword search (change 17) | 1 | 3 | YES, correct (both) |

Lesson: the refusal gate was behaving correctly on the context it was given. The 71-question baseline has not yet been re-run with changes 16 and 17; do that before treating them as safe (set `KEYWORD_V2=false` for an A/B run).

---

## 9. Next

Done: defaults flipped, Streamlit chat app, DOCX and CSV support, admin upload page, README, BD-002 case study (section 8b).
Confirmed on the user's Mac (2026-11-12): the app runs, PDF and CSV uploads index and answer correctly, and the two BD-002 false refusals are fixed. Not yet confirmed: DOCX upload in the app, and the 71-question regression after changes 16 and 17.

0. Mock demo run-through (planned after a break), then continue the list below.
1. Check step 21 with `python try_queries.py` and in the app; then agentic retry on refusal (rewrite and search again before refusing), measured on the eval.
2. Upload `samples/sample_concrete_spec.docx` (DOCX) in the app, ask about it, and record the result. CSV already works (`samples/sample_door_schedule.csv`).
3. Demo GIF for the README.
4. Azure Blob / AI Search adapter (optional).
5. Untested ideas: `CITATION_BOOST`, a verification step for refusals, a larger non-thinking model on a 16 GB machine.

## Run F: routing evaluation (2026-11-13, qwen3:1.7b)
New `routing_cases.csv` (35 cases) and `eval_routing.py` score where each message goes (chat / meta / search / answer) and whether follow-ups, splits and exclusions are rewritten correctly. No answers are generated, so a run takes seconds.

| Step | Change | Score |
|---|---|---:|
| 1 | Router prompt (CHAT / ANSWER / SEARCH), instant rules for greetings and document-list questions | 29/35 (83%) |
| 2 | Removed example phrases the 1.7B model copied; stripped echoed placeholders; fallback keeps previous question; acknowledgements as small talk | 29/35 (83%), different failures |
| 3 | Rules instead of the model for splitting several questions and for exclusions; ANSWER route only for summary/repeat requests; prepend previous question when the rewrite is weak | 34/35 (97%) |

Remaining failure: "tell me more about topic 2" needs the numbered list from the earlier answer (a bigger-model job).
Caution: the cases were tuned against these failures, so 97% is optimistic. Add new unseen cases before trusting it, and compare ROUTER_MODEL=qwen2.5:7b-instruct on the same file.
Added: ROUTER_MODEL setting, `list documents` / `Nth document` answers read straight from the index.
Next: agentic retry on refusal (rewrite and search again), measured with eval.py before and after.

### Run F, step 4-5 (routing, qwen3:1.7b)
| Step | Change | Score |
|---|---|---:|
| 4 | Typo-tolerant small talk (close match to a fixed list, short messages only) | 41/42 (98%) |
| 5 | Cases taken from a real chat session: identity/model questions answered from config, file-type questions ("csv file") answered from the index, context taken only from the last real document question (chat turns no longer leak into searches) | 51/52 (98%) |

The 10 "session" cases were real messages that failed, so they were fixed before being scored; they guard against regressions but are not an unseen test. Remaining failure: "tell me more about topic 2" (needs the numbered list from the previous answer).

## Run G: retry on refusal (2026-11-13, qwen3:1.7b, TEMPERATURE=0, same 71 questions)
Change: `AGENT_RETRY=true`. After a refusal, the model writes 2 alternative search queries, the system searches again with the merged results and answers once more (answer check still runs).

| Metric | Baseline (retry off) | Retry on |
|---|---:|---:|
| Right document in top 5 | 100% | 100% |
| Needed text retrieved | 92.7% | 92.7% |
| Fully correct answers | 46 of 66 | 46 of 66 |
| False refusals | 3.0% (2) | 1.5% (1) |
| Correct refusals | 4 of 5 | 3 of 5 |
| Citation present | 95.5% | 97.0% |
| Median latency | 14.2 s | 16.5 s |

Result: negative. The retry recovered two refusals, but both recovered answers were wrong: "How does Subpart M define a hole" (answer gave an invented 39-inch definition) and the unanswerable "guardrail height under Subpart L for scaffolds" (answered from unrelated text). Fully correct stayed at 46, and correct refusals fell from 4 to 3. The pre-set rule (do not keep it if correct refusals drop) applies, so `AGENT_RETRY` stays off (default).
Why: widening retrieval gives the 1.7B model more related text, and its answer check (VERIFY) often calls a related-topic answer "SUPPORTED". The retry removes the honest refusal without adding correct answers.
Revisit with the 7B model, whose answer check should be stricter. Also note temperature 0 baseline (46/66, 2 false refusals, 4/5 correct refusals) is the reference for comparisons.

## Run H: bigger model on the 16 GB laptop (2026-11-13, TEMPERATURE=0, retry off, same 71 questions)
Prompts, gate and answer check were written for qwen3:1.7b. Same index; retrieval metrics identical in every run (hit 1.000, MRR 0.852, context recall 0.927).

| Setup | Fully correct | False refusals | Correct refusals | Citation present | Median |
|---|---:|---:|---:|---:|---:|
| qwen3:1.7b, VERIFY on (Run G baseline) | 46 of 66 | 2 (3.0%) | 4 of 5 | 95.5% | 14.2 s |
| qwen2.5:7b-instruct, VERIFY on | 37 of 66 | 19 (28.8%) | 5 of 5 | 71.2% | 11.1 s |
| qwen2.5:7b-instruct, VERIFY off | about 46 (keyword accuracy 0.697) | 7 (10.6%) | 5 of 5 | 87.9% | 11.5 s |

Diagnosis (`diagnose_refusals.py`, 10 refused questions): 7 were refused by the answer check (VERIFY) on answers that matched the source, 3 by the ANSWERABLE gate. VERIFY, written for the 1.7B model, is too strict for the 7B. Turning it off cut false refusals from 19 to 7 and raised fully correct answers back to the 1.7B level.
Verdict: a tie on correctness; the 7B is faster and refuses all unanswerable questions, but refuses more answerable ones and cites less. Not a clear win, so no switch yet.

| qwen3:8b, VERIFY off | 53 of 66 | 2 (3.0%) | 4 of 5 | 97.0% | 15.7 s |
| qwen3:8b, VERIFY on | 46 of 66 | 11 (16.7%) | 5 of 5 | 83.3% | 21.6 s |

Decision (rule set before the runs: fully correct above 46, false refusals 3 or fewer, correct refusals 4 of 5 or better): qwen3:8b with VERIFY off passes and becomes the 16 GB configuration (`ANSWER_MODEL=qwen3:8b`, `VERIFY=false` in `.env`). VERIFY on the 8B fails the rule: it refuses answers that match the source (11 false refusals) in exchange for one more correct refusal. Known weakness of the chosen setup: it still answers one unanswerable question (basic wind speed), because the answer check is off.
The 8 GB Air keeps the earlier configuration (qwen3:1.7b with VERIFY on). Model-specific settings live in `.env`, not in the code.
