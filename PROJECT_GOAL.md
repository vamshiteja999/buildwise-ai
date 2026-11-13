# North star and guardrails

**Goal:** a local-first RAG assistant for construction documents that I can demo to construction and medical companies in Boca Raton and explain in an AI/ML interview (about four years of experience). Accuracy first, then speed, then polish.

**Done means:**
- Retrieval and answer quality are measured and not worse than Run E (see below).
- Follow-ups, small talk and document-list questions behave sensibly (routing eval).
- A short demo (GIF or video) and a README an interviewer can read in two minutes.
- One agentic feature (retry on refusal) with a before/after number.

**Quality gate: run before keeping any change** (app closed):
1. `python eval.py` must stay at or above Run E: right document in top 5 100%, citations 97%, needed text retrieved 92.7%, false refusals 1.5% or lower, correct refusals 5 of 5, fully correct 47 of 66.
2. `python eval_followups.py`: with rewrite must be at least as good as raw search.
3. `python eval_routing.py`: stay at 98% or better.

**Off route (park these, do not start them):** more phrase rules for small talk, UI restyling, new document types, cloud deployment, fine-tuning, anything not tied to a number above.
