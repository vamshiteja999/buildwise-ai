# BuildWise AI: Local Construction-Document Assistant

Ask plain-English questions about construction documents and get answers with citations. Everything runs on your own machine: no cloud API, and no document leaves the laptop.

**Built for:** OSHA fall-protection rules, the Florida Building Code accessibility chapter, Boca Raton permit forms, drawing sheets, and project packages.

## What it does
- **Chat with citations.** Every answer lists the source file, page and section it came from.
- **Refuses when it should.** Questions the documents cannot answer get "I could not find this in the documents".
- **Reads messy documents.** Text PDFs, scanned PDFs (OCR), tables, drawing sheets, Word files (`.docx`), spreadsheets saved as `.csv`, text and Markdown.
- **Understands the conversation.** Greetings and small talk get an instant reply; follow-ups such as "what about midrails?" or "does 16.1 inch work?" are resolved from earlier turns; several questions in one message are answered one by one; "don't give me X, give me Y" searches for Y.
- **Knows its own library.** "What documents do you have?", "do you have a CSV?", "how many pages is the 3rd document?" and "summarize the Chapter 10 file" are answered straight from the index.
- **Admin upload page.** Password-protected: upload, index, replace or remove documents without touching code.
- **Measured, not guessed.** Evaluation sets for answers (71 questions), routing (60 messages) and follow-ups; every change and every run is logged in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), including the experiments that did not work.

## Results (71 hand-checked questions, temperature 0)

| Metric | 16 GB laptop: `qwen3:8b` | 8 GB laptop: `qwen3:1.7b` |
|---|---:|---:|
| Right document in top 5 | 100% | 100% |
| Needed text retrieved | 92.7% | 92.7% |
| Fully correct answers | **53 of 66** | 46 of 66 |
| Answers carrying a citation | 97.0% | 95.5% |
| False refusals (answerable questions refused) | 3.0% (2 of 66) | 3.0% (2 of 66) |
| Correct refusals (unanswerable questions) | 4 of 5 | 4 of 5 |
| Median time per answer | 15.7 s | 14.2 s |

Retrieval is identical in both columns, so the difference comes from the language model. The 16 GB configuration runs with the answer check off (`VERIFY=false`); the 8 GB one keeps it on. Details and the runs that led here are in the experiment log.

**What did not work, and why it is documented:**
- *Retry on refusal* (search again with alternative wording): recovered two refusals, but both answers were wrong, and one unanswerable question got an invented answer. Left off (`AGENT_RETRY=false`).
- *Answer check on bigger models:* it rejected correct answers (11 false refusals on `qwen3:8b`, 19 on `qwen2.5:7b`), so it is off for them.

**Known limits:** the 1.7B and 8B models can still answer one unanswerable question (the Boca Raton basic wind speed) with related text; "tell me more about topic 2" cannot resolve a numbered list from an earlier answer; scanned pages and drawing annotations are read less reliably than text; the app cannot display page images (it cites the file and page instead).

## How it works
1. **Extract.** PyMuPDF reads PDFs (Tesseract OCR for scanned pages). Table rows are rebuilt so `Metal Roofs | A,B,C | 1,2,3,4,5` stays together; drawing sheets are split into keynote legend, title block and annotations; repeated headers and footers are removed. `.docx` and `.csv` files are read with the standard library.
2. **Chunk.** Regulations are split by numbered clause with parent-clause context; other documents by paragraph (about 350 words).
3. **Embed and store.** `nomic-embed-text` through Ollama, stored in a local Qdrant database.
4. **Route.** Fixed rules handle greetings, document-list questions, summaries, splitting and exclusions instantly. A model call resolves follow-ups; its output is checked by code (a chat reply is overruled for numbers, units or construction words).
5. **Retrieve.** Vector search (meaning) plus BM25 keyword search (exact terms), merged with reciprocal rank fusion.
6. **Answer.** The model answers only from the retrieved text, cites `[n]` sources, and first decides whether the question is answerable at all.
7. **Verify (optional).** A second check refuses an answer the cited text does not support (`VERIFY=true`; best with the 1.7B model).

## Setup
Requires macOS (Apple Silicon), Python 3.9 or newer, and [Ollama](https://ollama.com).

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:8b          # 16 GB machine; use qwen3:1.7b on an 8 GB machine
```

Put documents in `data/raw/` (or upload them in the app), then:

```bash
python ingest.py              # index the documents
streamlit run app.py          # open the chat and admin app
```

To use the Admin page, set a password in `.env`: `ADMIN_PASSWORD=your-password`.

**Choose the model for your machine in `.env`:**

| Machine | `.env` |
|---|---|
| 16 GB | `ANSWER_MODEL=qwen3:8b` and `VERIFY=false` |
| 8 GB | `ANSWER_MODEL=qwen3:1.7b` and `VERIFY=true` (the defaults) |

Only one program can use the local database at a time: close the app before running `ingest.py` or any evaluation, and the reverse.

## Evaluation commands
| Command | What it checks |
|---|---|
| `python eval.py` | 71 golden questions: retrieval, answers, citations, refusals. Use `TEMPERATURE=0 EVAL_REPORT=name.md` for comparable runs |
| `python eval_routing.py` | Where 60 messages are routed (chat, document list, summary, search) and whether follow-ups, splits and exclusions are rewritten correctly |
| `python eval_followups.py` | Whether the conversation-aware rewrite helps or hurts retrieval |
| `python diagnose_refusals.py <report>` | Which stage (answer gate or answer check) refused each question |
| `bash run_model_compare.sh <model>` | Routing plus the full answer eval for one model |
| `python -m pytest -q` | Automated tests (install `requirements-dev.txt` first) |

The routing cases were written against observed failures, so the routing score is optimistic; the answer evaluation is the number to trust.

## Configuration
All settings live in `config.py` and can be overridden in `.env`: `ANSWER_MODEL`, `ROUTER_MODEL`, `VERIFY`, `AGENT_RETRY`, `TEMPERATURE`, `QDRANT_COLLECTION`, `QUERY_UNDERSTANDING`, and the earlier flags (`PROMPT_V2`, `EXTRACT_V2`, `KEYWORD_V2`). After changing extraction or chunking, re-ingest into a new collection by setting `QDRANT_COLLECTION`.

## Project files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app: Chat, Admin, Evaluation |
| `buildwise/core.py` | Extraction, chunking, retrieval, routing, answering |
| `config.py` | Settings and flags |
| `eval.py`, `golden_questions.csv`, `golden_bd002.csv` | Answer evaluation and test questions |
| `routing_cases.csv`, `eval_routing.py`, `followup_retrieval_cases.csv`, `eval_followups.py` | Routing and follow-up evaluation |
| `tests/` | Automated tests |
| `EXPERIMENT_LOG.md` | Every change and every run's metrics |
| `PROJECT_GOAL.md` | Goal, quality gate and what is deliberately out of scope |
| `samples/` | A synthetic `.docx` and a door schedule `.csv` for trying the upload page |

## Storage
Qdrant data is local under `data/qdrant`. The storage interface is isolated in `buildwise/core.py` (`get_client`, `hybrid_search`), so a cloud backend (for example Azure AI Search with Blob storage) can replace those functions without changing the app.

## License
MIT. See [`LICENSE`](LICENSE). The sample company in `samples/demo_company/` is fictional.
