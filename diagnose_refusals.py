"""Why does a model refuse questions it should answer?  Which stage says no: the ANSWERABLE gate or the VERIFY check?

Usage (app closed):
    ANSWER_MODEL=qwen2.5:7b-instruct ROUTER_MODEL=qwen2.5:7b-instruct TEMPERATURE=0 \
        python diagnose_refusals.py reports/run_qwen2_5_7b-instruct.md

Takes the answerable questions that were refused in an eval report and re-runs 10 of them,
printing for each: the model's first reply, the verify decision, and where the refusal came from.
"""
import sys
from collections import Counter

import buildwise.core as core
from config import ANSWER_MODEL, VERIFY, TEMPERATURE


def refused_questions(report, limit=10):
    out = []
    for line in open(report, encoding="utf-8"):
        if not line.startswith("|"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 8 and c[1] in ("easy", "multi") and c[5] == "yes":
            out.append(c[0])
    return out[:limit]


def main():
    report = sys.argv[1] if len(sys.argv) > 1 else "reports/run_qwen2_5_7b-instruct.md"
    questions = refused_questions(report)
    print(f"model={ANSWER_MODEL} VERIFY={VERIFY} TEMPERATURE={TEMPERATURE}; re-running {len(questions)} refused questions\n")

    log = []
    original = core.request_json

    def spy(url, payload, timeout):
        reply = original(url, payload, timeout)
        log.append((payload.get("prompt", ""), reply.get("response") or ""))
        return reply

    core.request_json = spy
    stages = Counter()
    for q in questions:
        log.clear()
        core.LAST_VERIFY.clear()
        answer, _ = core.answer_question(q)
        gen = next((r for p, r in log if p.startswith("You are BuildWise")), "")
        refused = answer.strip() == core.REFUSAL_TEXT
        if not refused:
            stage = "ANSWERED this time"
        elif "ANSWERABLE: NO" in gen.upper():
            stage = "GATE said ANSWERABLE: NO"
        elif core.LAST_VERIFY.get("decision") == "refused":
            stage = "VERIFY refused (" + core.LAST_VERIFY.get("reason", "") + ")"
        else:
            stage = "other"
        stages[stage.split(" (")[0]] += 1
        print(f"Q: {q[:100]}")
        print(f"   stage : {stage}")
        print(f"   reply : {gen.strip()[:220]!r}\n")
    print("Summary:", dict(stages))


if __name__ == "__main__":
    main()
