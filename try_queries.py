"""Try the query-understanding step on typical cases (stop the Streamlit app first).
  python try_queries.py            show how each message would be handled (fast)
  python try_queries.py --answer   also run the full answer for each (slow)
"""
import sys
import time

import buildwise.core as core

H_GUARD = [("user", "What top rail height is required for a guardrail system?"),
           ("assistant", "The top rail must be 42 inches plus or minus 3 inches above the walking surface [1].")]
H_ROOF = [("user", "Which sections and attachments does a Boca Raton permit for a metal roof require?"),
          ("assistant", "Section A General Information, Section B, and Section C for low and steep sloped roofs [1].")]
CASES = [
    ("small talk", "good morning", []),
    ("thanks", "Thanks!", []),
    ("ordinary question (should pass through)", "What top rail height is required for a guardrail system?", []),
    ("follow-up 1", "what about midrails?", H_GUARD),
    ("follow-up 2", "and for built-up roofs?", H_ROOF),
    ("follow-up that needs the earlier answer", "how high is that again?", H_GUARD),
    ("exclusion", "Don't give me the foundation only permit requirements, what does design development require?", []),
    ("two questions, numbered", "1. What is the revision date of BD-002? 2. What is the minimum font size for text on plans?", []),
    ("two questions, ?", "What is the OSHA penalty for violating 1926.501? What is the permit fee for a Boca Raton roofing permit?", []),
    ("follow-up with no history (nothing to resolve)", "what about midrails?", []),
]
core.health_check()
for label, message, history in CASES:
    started = time.perf_counter()
    plan = core.understand_query(message, history)
    took = time.perf_counter() - started
    print("=" * 88)
    print(f"{label}: {message!r}  [{'with history' if history else 'no history'}]")
    print(f"  handled: {plan['how']} ({plan['why'] or 'no rewrite needed'}) in {took:.1f}s -> {plan['kind']}")
    for q in plan["queries"]:
        print("  search:", q)
    if plan["reply"]:
        print("  reply :", plan["reply"][:100])
    if "--answer" in sys.argv:
        answer, _ = core.ask(message, history)
        print("  ANSWER:", answer.replace("\n", " ")[:300])
