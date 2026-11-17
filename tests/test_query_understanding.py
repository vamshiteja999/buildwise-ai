"""Tests that need no model and no database. Run:  python -m pytest -q"""
import buildwise.core as core


def _fake_model(monkeypatch, reply):
    monkeypatch.setattr(core, "request_json", lambda *a, **k: {"response": reply})


HISTORY = [("user", "What is the minimum top rail height for a guardrail system?"), ("assistant", "42 inches [1].")]


# ---- instant replies (no model, no search) -------------------------------------------------
def test_greetings_get_a_reply_without_search():
    for msg in ["good morning", "Hello!", "hi there", "thanks", "Thank you so much", "bye", "what can you do?",
                "wassup", "hey, hows it going", "wow great", "ok cool"]:
        assert core.fast_chat_reply(msg), msg


def test_typos_in_small_talk_are_tolerated():
    for msg in ["helo", "hellooo", "thnaks", "gud morning", "hwo are you", "hii", "heyy"]:
        assert core.fast_chat_reply(msg), msg


def test_real_questions_are_not_small_talk():
    for msg in ["What top rail height is required for a guardrail system?",
                "hello what is the revision date of BD-002?", "help me find the permit fee",
                "pool lift requirements", "hollow core slab requirements", "yellow warning line",
                "what does figure 1003.3.1 show", "show me the guardrail rules"]:
        assert core.fast_chat_reply(msg) is None, msg


def test_identity_and_capability_questions_have_direct_answers():
    assert "BuildWise" in core.fast_chat_reply("who are you?")
    assert core.ANSWER_MODEL in core.fast_chat_reply("what model are you?")
    assert "can't show images" in core.fast_chat_reply("can you show me images?")
    assert "can't show images" in core.fast_chat_reply("an you show me images of pool planing")
    assert "Admin" in core.fast_chat_reply("how do I upload a document?")
    for msg in ["I want to add a document", "let me test some more want to add a docx", "i wanna upload a file"]:
        assert "Admin" in core.fast_chat_reply(msg), msg
    assert core.fast_chat_reply("what is the fee to add a document to the permit") is None


# ---- routing helpers -----------------------------------------------------------------------
def test_clear_single_question_passes_through_untouched():
    plan = core.understand_query("What top rail height is required for a guardrail system?", [])
    assert plan["how"] == "passthrough" and plan["queries"] == ["What top rail height is required for a guardrail system?"]


def test_rewrite_triggers():
    assert core.needs_rewrite("Don't give me the foundation permit, what does DD need?", []) == "exclusion"
    assert core.needs_rewrite("What is the fee? What is the deadline?", []) == "several questions"
    assert core.needs_rewrite("What is the top rail height and what is the midrail height for guardrails?", []) == "several questions"
    assert core.needs_rewrite("what about midrails?", HISTORY) == "follow-up"
    assert core.needs_rewrite("how strong must it be?", HISTORY) == "follow-up"


def test_a_new_topic_is_not_treated_as_a_follow_up():
    assert not core.looks_like_followup("what is the fire rating of door 101?")
    assert core.needs_rewrite("what is the fire rating of door 101?", HISTORY) == ""
    assert core.looks_like_followup("can you elaborate section B")
    assert core.looks_like_followup("does 16.1 inch works?")
    assert not core.looks_like_followup("is BlueWave HVAC prequalified?")
    assert not core.looks_like_followup("Who is the Structural Steel sub?")
    assert core.looks_like_followup("what about midrails?")


def test_split_questions():
    assert core.split_questions("1) What is the permit fee? 2) How long is a roof permit valid?") == [
        "What is the permit fee?", "How long is a roof permit valid?"]
    parts = core.split_questions("What is the top rail height and what is the midrail height for guardrails?")
    assert parts and len(parts) == 2
    assert core.split_questions("What is the top rail height for guardrails?") is None


def test_strip_exclusion():
    kept = core.strip_exclusion("dont give me foundation only permit requirements, give me the roofing ones")
    assert "roofing" in kept and "foundation" not in kept.split("permit")[0]
    assert core.strip_exclusion("everything about guardrails except scaffolds") == "everything about guardrails"
    assert core.strip_exclusion("what is the top rail height") is None


def test_parse_route():
    assert core.parse_route("SEARCH: What are guardrail midrail requirements?") == (
        "question", ["What are guardrail midrail requirements?"])
    assert core.parse_route("SEARCH: <question> - What does topic 2 cover?") == ("question", ["What does topic 2 cover?"])
    assert core.parse_route("SEARCH: /no_think") is None
    assert core.parse_route("CHAT: Hi there! Ask me about permits.") == ("chat", "Hi there! Ask me about permits.")
    assert core.parse_route("ANSWER: The third one.\nSEARCH: x y") == ("answer", "The third one.")
    assert core.parse_route("I think you should ask about guardrails") is None


# ---- understand_query with a fake model ----------------------------------------------------
def test_follow_up_is_rewritten(monkeypatch):
    _fake_model(monkeypatch, "SEARCH: What are the guardrail midrail requirements?")
    plan = core.understand_query("what about midrails?", HISTORY)
    assert plan["how"] == "model" and "midrail" in plan["queries"][0].lower()


def test_unreadable_rewrite_keeps_the_topic_of_the_previous_question(monkeypatch):
    _fake_model(monkeypatch, "I am not sure what you mean")
    plan = core.understand_query("what about midrails?", HISTORY)
    assert plan["how"] == "fallback"
    assert "top rail height" in plan["queries"][0] and "midrails" in plan["queries"][0]


def test_model_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(core, "request_json", boom)
    plan = core.understand_query("what about midrails?", [("user", "a"), ("assistant", "b")])
    assert plan["how"] == "fallback" and plan["queries"] == ["what about midrails?"]


def test_chat_route_is_overruled_for_technical_follow_ups(monkeypatch):
    _fake_model(monkeypatch, "CHAT: You're welcome! Let me know if you need further assistance.")
    history = [("user", "What are the seat height requirements for pool lifts?"), ("assistant", "16 to 19 inches [1].")]
    plan = core.understand_query("does 15.9 inches works?", history)
    assert plan["kind"] == "question" and "pool lifts" in plan["queries"][0]


def test_answer_route_only_for_summary_requests(monkeypatch):
    _fake_model(monkeypatch, "ANSWER: Scaffolds need guardrails too.")
    plan = core.understand_query("and for scaffolds?", HISTORY)
    assert plan["kind"] == "question"


def test_new_topic_after_a_chat_turn_does_not_get_old_context(monkeypatch):
    _fake_model(monkeypatch, "SEARCH: what is the fire rating of door 101?")
    history = [("user", "who are you?"), ("assistant", "I'm BuildWise AI"), ("user", "hi"), ("assistant", "Hello!")]
    plan = core.understand_query("what is the fire rating of door 101?", history)
    assert plan["queries"] == ["what is the fire rating of door 101?"]


# ---- ask() ---------------------------------------------------------------------------------
def test_ask_chat_does_not_search(monkeypatch):
    def no_search(*a, **k):
        raise AssertionError("must not search for small talk")
    monkeypatch.setattr(core, "answer_question", no_search)
    answer, citations = core.ask("good morning")
    assert "documents" in answer and citations == []


def test_ask_combines_several_questions_with_distinct_citations(monkeypatch):
    monkeypatch.setattr(core, "answer_question",
                        lambda q, extra_queries=None: (f"Answer to {q} [1]", ["[1] Source: a.pdf, page 1, section x"]))
    answer, citations = core.ask("1. Revision date of BD-002? 2. Minimum font size?")
    assert "[1.1]" in answer and "[2.1]" in answer
    assert citations[0].startswith("[1.1]") and citations[1].startswith("[2.1]")


def test_switch_off_goes_straight_to_the_pipeline(monkeypatch):
    monkeypatch.setattr(core, "QUERY_UNDERSTANDING", False)
    monkeypatch.setattr(core, "answer_question", lambda q, extra_queries=None: ("direct", []))
    assert core.ask("good morning") == ("direct", [])


# ---- agentic retry (off by default) ---------------------------------------------------------
def test_retry_is_off_by_default_and_leaves_refusals_alone(monkeypatch):
    monkeypatch.setattr(core, "AGENT_RETRY", False)
    monkeypatch.setattr(core, "answer_question", lambda q, extra_queries=None: (core.REFUSAL_TEXT, []))
    assert core.answer_with_retry("anything")[0] == core.REFUSAL_TEXT


def test_retry_recovers_a_refusal_only_when_the_second_attempt_answers(monkeypatch):
    monkeypatch.setattr(core, "AGENT_RETRY", True)
    monkeypatch.setattr(core, "alternative_queries", lambda q, n=2: ["other words here"])

    def fake(q, extra_queries=None):
        return ("Found it [1]", ["[1] Source: a.pdf"]) if extra_queries else (core.REFUSAL_TEXT, [])
    monkeypatch.setattr(core, "answer_question", fake)
    assert core.answer_with_retry("question")[0] == "Found it [1]"
    monkeypatch.setattr(core, "answer_question", lambda q, extra_queries=None: (core.REFUSAL_TEXT, []))
    assert core.answer_with_retry("question")[0] == core.REFUSAL_TEXT


def test_document_position_is_understood_with_typos_and_spacing():
    assert core.ordinal_number("what is 2 nd document about?") == 2
    assert core.ordinal_number("can you summarize 3rd documnent?") == 3
    assert core.ordinal_number("summarize the third file") == 3
    assert core.ordinal_number("summarize document 5") == 5
    for q in ["What is the fire rating of door 101?", "summarize 1926.502", "summarize the chapter 10 file",
              "what does keynote 07 61 03 specify", "what is the first one about"]:
        assert core.ordinal_number(q) is None, q
    docs = [("a.pdf", 1), ("b.pdf", 1), ("c.csv", 1)]
    assert core.resolve_document("what is 2nd document about?", [], docs) == ("b.pdf", None)
    name, ask_back = core.resolve_document("summarize the 9th document", [], docs)
    assert name is None and "only have 3" in ask_back


def test_recent_document_questions_are_recognised():
    for q in ["what is the recent document indexed?", "which file was added last?", "most recent file uploaded"]:
        assert core.RECENT_DOC_RE.search(q), q
    for q in ["what is the latest revision of sheet A-002", "what is the last step in the permit process",
              "what documents do you have"]:
        assert not core.RECENT_DOC_RE.search(q), q
