import csv, os, re, statistics, time
from collections import defaultdict
from config import GOLDEN_CSV, EVAL_REPORT, REPORT_DIR
from buildwise.core import health_check, hybrid_search, answer_with_retry as answer_question

REFUSAL = "i could not find this in the documents"
# Other wordings a small model uses to decline. A refusal is a SHORT answer containing one of them.
REFUSAL_PATTERNS = [REFUSAL, "does not contain", "do not contain", "not specified in",
                    "no information", "not provided in", "documents do not provide"]


def classify_refusal(low):
    """Return (refused, hedged). refused = short answer that declines.
    hedged = a long answer that also tacks on the exact refusal sentence."""
    has_pattern = any(p in low for p in REFUSAL_PATTERNS)
    refused = has_pattern and len(low) < 350
    hedged = (REFUSAL in low) and not refused
    return refused, hedged


def squash(text):
    """Make comma spacing irrelevant: 'A, B, C' and 'A,B,C' compare equal."""
    return re.sub(r",\s+", ",", text)


def split_list(value, sep):
    return [squash(part.strip().lower()) for part in (value or "").split(sep) if part.strip()]


def main():
    health_check()
    if not GOLDEN_CSV.exists():
        raise SystemExit(f"ERROR: Missing {GOLDEN_CSV}. Restore golden_questions.csv.")
    rows = list(csv.DictReader(GOLDEN_CSV.open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise SystemExit("ERROR: golden_questions.csv has no test rows.")
    # Quick runs: EVAL_EVERY=3 python eval.py  (every 3rd question). Default 1 = all.
    rows = rows[::max(1, int(os.getenv("EVAL_EVERY", "1")))]

    results = []
    for row in rows:
        q = row["question"].strip()
        kind = row["type"].strip().lower()
        sources_expected = split_list(row.get("expected_source"), ";")
        keywords = split_list(row.get("answer_keywords"), "|")
        answerable = kind != "unanswerable"

        # One retrieval only: answer_question returns the citations of the chunks the
        # model actually saw ("[n] Source: <file>, page <p>, section <s>").
        # Context check (not timed): which expected keywords appear in the top-5 chunk text?
        ctx = squash(" ".join(((p.payload or {}).get("text", "") or "").lower() for p in hybrid_search(q, 5)))
        context_recall = (sum(1 for k in keywords if k in ctx) / len(keywords)) if keywords else None
        start = time.perf_counter()
        answer, citations = answer_question(q)
        latency = time.perf_counter() - start
        low = answer.lower()
        low_kw = squash(low)

        found = []
        for c in citations:
            m = re.search(r"Source:\s*(.+?),\s*page\s", c)
            found.append((m.group(1) if m else c).lower())
        rank = 0
        for i, s in enumerate(found, 1):
            if any(e in s for e in sources_expected):
                rank = i
                break
        refused, hedged = classify_refusal(low)
        results.append({
            "hedged": hedged, "context_recall": context_recall,
            "q": q, "kind": kind, "answerable": answerable, "rank": rank, "latency": latency,
            "refused": refused,
            "keywords_ok": bool(keywords) and all(k in low_kw for k in keywords),
            "has_keywords": bool(keywords),
            "cited": bool(re.search(r"\[\d+\]|page", low)) or any(e in low for e in sources_expected),
            "excerpt": answer.replace("\n", " ")[:200],
        })

    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    graded = [r for r in ans if r["has_keywords"]]

    def rate(items, key):
        return sum(1 for r in items if r[key]) / len(items) if items else float("nan")

    metrics = {
        "Hit rate@5 (answerable)": rate([dict(r, k=bool(r["rank"])) for r in ans], "k"),
        "MRR (answerable)": sum(1 / r["rank"] for r in ans if r["rank"]) / len(ans) if ans else float("nan"),
        "Answer keyword accuracy (answerable)": rate(graded, "keywords_ok"),
        "Citation present (answerable)": rate(ans, "cited"),
        "Context recall (answerable, keywords in top-5 text)": (
            sum(r["context_recall"] for r in graded) / len(graded) if graded else float("nan")),
        "False refusal rate (answerable)": rate(ans, "refused"),
        "Hedged answers (answerable)": rate(ans, "hedged"),
        "Correct refusal rate (unanswerable)": rate(unans, "refused"),
        "Median latency, retrieval + answer (s)": statistics.median(r["latency"] for r in results),
    }

    ok = [r for r in graded if r["keywords_ok"]]
    gen_fail = [r for r in graded if not r["keywords_ok"] and r["context_recall"] == 1]
    ret_fail = [r for r in graded if not r["keywords_ok"] and r["context_recall"] < 1]
    by_type = defaultdict(list)
    for r in ans:
        by_type[r["kind"]].append(r)

    out = ["# BuildWise Evaluation Results", "",
           f"Questions: {len(results)} ({len(ans)} answerable, {len(unans)} unanswerable)", "",
           "| Metric | Result |", "|---|---:|"]
    out += [f"| {k} | {v:.3f} |" for k, v in metrics.items()]
    out += ["", "## Where answerable questions fail", "", "| Outcome | Count |", "|---|---:|",
            f"| Correct (all keywords in answer) | {len(ok)} |",
            f"| Generation failure (keywords WERE in retrieved text, answer missed them) | {len(gen_fail)} |",
            f"| Retrieval failure (keywords NOT in retrieved text) | {len(ret_fail)} |"]
    out += ["", "## By question type (answerable)", "", "| Type | N | Hit rate@5 | Keyword accuracy |", "|---|---:|---:|---:|"]
    for kind, items in sorted(by_type.items()):
        g = [r for r in items if r["has_keywords"]]
        hit = sum(1 for r in items if r["rank"]) / len(items)
        kacc = rate(g, "keywords_ok")
        out.append(f"| {kind} | {len(items)} | {hit:.3f} | {kacc:.3f} |")
    out += ["", "## Per-question results", "",
            "| Question | Type | Source rank | Ctx recall | Keywords OK | Refused | Latency (s) | Answer excerpt |",
            "|---|---|---:|---:|:-:|:-:|---:|---|"]
    for r in results:
        kw = "-" if not r["has_keywords"] else ("yes" if r["keywords_ok"] else "NO")
        out.append(
            f"| {r['q'].replace('|', '/')} | {r['kind']} | {r['rank'] or 'Miss'} | "
            f"{'-' if r['context_recall'] is None else format(r['context_recall'], '.0%')} | {kw} | "
            f"{'yes' if r['refused'] else ('hedged' if r['hedged'] else 'no')} | "
            f"{r['latency']:.2f} | {r['excerpt'].replace('|', '/')} |"
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()