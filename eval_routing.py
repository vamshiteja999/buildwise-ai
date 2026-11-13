"""Routing evaluation: does each message go to the right place (chat / meta / search / answer)?

Fast (no document answers are generated): one router model call per message that needs it.
Run with the Streamlit app closed (the document list uses the local database):
    python eval_routing.py            # all cases
    python eval_routing.py --only followup
Writes reports/routing_<time>.md and prints failures.
"""
import argparse, csv, json, time
from pathlib import Path

from config import ROOT, ROUTER_MODEL, QUERY_UNDERSTANDING
from buildwise.core import meta_reply, ordinal_reply, summary_reply, understand_query


def route(message, history):
    """Return (route, queries, how). Mirrors the first steps of core.ask() without answering."""
    if summary_reply(message, history):
        return "summary", [], "rule"
    if meta_reply(message) or ordinal_reply(message, history):
        return "meta", [], "rule"
    plan = understand_query(message, history)
    if plan["kind"] == "chat":
        return ("answer" if str(plan.get("why", "")).endswith("answer") else "chat"), [], plan["how"]
    return "search", plan["queries"], plan["how"]


def check(case, got, queries):
    expected = case["expect_route"].split("|")
    problems = []
    if got not in expected:
        problems.append(f"route {got!r}, wanted {case['expect_route']!r}")
    if got == "search":
        text = " ".join(queries).lower()
        for term in filter(None, case["must_have"].split("|")):
            # several terms separated by | mean "at least one of these"
            pass
        wanted = [t for t in case["must_have"].split("|") if t]
        if wanted and not any(t.lower() in text for t in wanted):
            problems.append(f"queries lack any of {wanted}")
        for t in filter(None, case["must_not"].split("|")):
            if t.lower() in text:
                problems.append(f"queries still contain {t!r}")
        if case["category"] == "multi" and len(queries) < 2:
            problems.append("expected the message to be split into 2+ queries")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "routing_cases.csv"))
    ap.add_argument("--only", default="", help="run one category")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.cases, encoding="utf-8")))
    if args.only:
        rows = [r for r in rows if r["category"] == args.only]
    results, by_cat = [], {}
    for case in rows:
        history = [tuple(x) for x in json.loads(case["history"] or "[]")]
        t0 = time.time()
        try:
            got, queries, how = route(case["message"], history)
            problems = check(case, got, queries)
        except Exception as exc:
            got, queries, how, problems = "error", [], "error", [f"exception: {exc}"]
        secs = time.time() - t0
        ok = not problems
        by_cat.setdefault(case["category"], []).append(ok)
        results.append((case, got, queries, how, problems, secs))
        print(("PASS " if ok else "FAIL ") + f"{case['id']:>3} {case['category']:<13} {case['message'][:60]!r} -> {got} ({how}, {secs:.1f}s)")
        for p in problems:
            print("        ", p)
        if not ok and queries:
            print("         searched for:", queries)
    total = sum(sum(v) for v in by_cat.values())
    n = sum(len(v) for v in by_cat.values())
    lines = [f"# Routing evaluation ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Router model: `{ROUTER_MODEL}`, query understanding: {QUERY_UNDERSTANDING}", "",
             f"**Overall: {total}/{n} ({100*total/max(n,1):.0f}%)**", "", "| Category | Passed |", "|---|---:|"]
    lines += [f"| {c} | {sum(v)}/{len(v)} |" for c, v in by_cat.items()]
    lines += ["", "## Failures", ""]
    for case, got, queries, how, problems, secs in results:
        if problems:
            lines.append(f"- #{case['id']} `{case['message']}` -> {got}; {'; '.join(problems)}; searched: {queries}")
    avg = sum(r[5] for r in results) / max(len(results), 1)
    lines += ["", f"Average routing time: {avg:.1f}s"]
    out = ROOT / "reports" / f"routing_{time.strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nOverall {total}/{n}.  Report: {out}")


if __name__ == "__main__":
    main()
