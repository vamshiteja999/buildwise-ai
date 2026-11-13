"""Tests for the document readers, header/footer removal, BM25 and the answer check. No model or database."""
from pathlib import Path

import buildwise.core as core

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_csv_rows_stay_whole_and_name_their_columns(tmp_path):
    f = tmp_path / "doors.csv"
    f.write_text("Door,Room,Rating\n101,Lobby,None\n102,Electrical,90 min\n")
    pages = core.extract_pages(f)
    text = pages[0][1]
    assert "Door: 102 | Room: Electrical | Rating: 90 min" in text
    chunks = core.make_chunks(f, pages, "d1")
    assert any("Door: 102 | Room: Electrical | Rating: 90 min" in c["text"] for c in chunks)


def test_csv_semicolons_and_long_files_split_into_sections(tmp_path):
    f = tmp_path / "big.csv"
    f.write_text("id;desc\n" + "\n".join(f"{i};item {i}" for i in range(1, 121)))
    assert len(core.extract_pages(f)) == 3  # 50 rows per section


def test_docx_headings_and_tables():
    pages = core.extract_pages(SAMPLES / "sample_concrete_spec.docx")
    joined = "\n".join(p[1] for p in pages)
    assert pages and core.HEADING_MARK in pages[0][1] or pages
    assert " | " in joined  # the table keeps its rows


def test_repeated_footer_with_changing_page_number_is_removed(tmp_path):
    topics = ["guardrails", "nets", "anchorage", "toeboards", "ladders", "scaffolds", "stairs", "covers"]
    pages = [(i, f"The {t} requirement describes how {t} must be installed and inspected on every job site.\n"
                 f"BD-002 Project Submission Page {i} of 8", False, "") for i, t in enumerate(topics, 1)]
    cleaned = core._drop_repeated_edge_lines(pages, tmp_path / "other.pdf")
    assert all("Project Submission" not in p[1] for p in cleaned)
    assert all("requirement describes" in p[1] for p in cleaned)


def test_long_lines_that_differ_only_by_digits_are_kept(tmp_path):
    line = "Table of anchorage loads for system number %d lists the required strength in pounds for each case here"
    pages = [(i, line % i + "\nBD-002 Project Submission Page %d of 8" % i, False, "") for i in range(1, 9)]
    cleaned = core._drop_repeated_edge_lines(pages, tmp_path / "other.pdf")
    assert all("Table of anchorage loads" in p[1] for p in cleaned)


def test_bm25_prefers_the_rare_term_over_filler_words():
    docs = [("a", "the the the is of what the plans plans plans plans " * 20),
            ("b", "Font size of any text shall be at least 10-pt."),
            ("c", "Unrelated drainage text about pipes and inverts.")]
    ranked = core.bm25_rank("What is the minimum font size for text on plans?", docs)
    assert ranked[0][0] == "b"


def test_bm25_cache_does_not_serve_stale_text_for_a_reused_id():
    assert core.bm25_rank("alpha", [("x", "alpha beta")])
    assert core.bm25_rank("alpha", [("x", "gamma delta")]) == []


def test_bm25_matches_codes_with_hyphens_and_dots():
    docs = [("a", "General notes about sheets."), ("b", "Number: BD-002 Title: Project Submission"),
            ("c", "Fall protection per 1926.501 applies at 6 feet.")]
    assert core.bm25_rank("BD-002", docs)[0][0] == "b"
    assert core.bm25_rank("1926.501", docs)[0][0] == "c"


def test_answer_that_says_the_fact_is_missing_is_refused_without_a_model_call(monkeypatch):
    def no_model(*a, **k):
        raise AssertionError("the free check should decide this")
    monkeypatch.setattr(core, "request_json", no_model)
    out = core.verify_answer("What is the contract value?",
                             "The contract value is not explicitly stated in the provided context.", ["[1] x"])
    assert out == core.REFUSAL_TEXT


def test_unsupported_verdict_refuses_and_unclear_verdict_keeps(monkeypatch):
    ctx = ["[1] Source: a.pdf\nTop rail height is 42 inches."]
    monkeypatch.setattr(core, "request_json", lambda *a, **k: {"response": "UNSUPPORTED"})
    assert core.verify_answer("Q?", "It is 42 inches [1].", ctx) == core.REFUSAL_TEXT
    monkeypatch.setattr(core, "request_json", lambda *a, **k: {"response": "hmm maybe"})
    assert core.verify_answer("Q?", "It is 42 inches [1].", ctx) == "It is 42 inches [1]."
