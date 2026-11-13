"""Does the conversation-aware rewrite help or hurt retrieval?
For each case: search with the raw message (no rewriting) vs with BuildWise's rewritten queries,
and check whether the expected document appears in the top 5. No answers are generated.
Run with the Streamlit app closed:  python eval_followups.py
"""
import csv, json, time
from config import ROOT
from buildwise.core import hybrid_search, understand_query


def hit(queries, wanted):
    seen = set()
    for q in queries:
        for p in hybrid_search(q, 5):
            seen.add(str((p.payload or {}).get("source", "")).lower())
    return any(w.lower() in s for w in wanted for s in seen)


def main():
    rows = list(csv.DictReader(open(ROOT / "followup_retrieval_cases.csv", encoding="utf-8")))
    raw_hits = new_hits = 0
    lines = ["# Follow-up retrieval check (%s)" % time.strftime("%Y-%m-%d %H:%M"), "",
             "| # | Message | Raw search | With rewrite | Searched for |", "|---|---|:-:|:-:|---|"]
    for r in rows:
        history = [tuple(x) for x in json.loads(r["history"] or "[]")]
        wanted = r["expect_source_any"].split("|")
        plan = understand_query(r["message"], history)
        queries = plan["queries"] or [r["message"]]
        a, b = hit([r["message"]], wanted), hit(queries, wanted)
        raw_hits += a
        new_hits += b
        lines.append(f"| {r['id']} | {r['message']} | {'yes' if a else 'no'} | {'yes' if b else 'no'} | {' / '.join(queries)[:110]} |")
        print(f"{r['id']:>2} raw={'yes' if a else 'no ':3} rewrite={'yes' if b else 'no ':3} {r['message'][:45]!r} -> {queries}")
    n = len(rows)
    lines += ["", f"**Raw search: {raw_hits}/{n}. With rewrite: {new_hits}/{n}.**"]
    out = ROOT / "reports" / f"followups_{time.strftime('%Y%m%d_%H%M')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRaw {raw_hits}/{n}  |  With rewrite {new_hits}/{n}   Report: {out}")


if __name__ == "__main__":
    main()
