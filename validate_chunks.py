import re
from collections import Counter
from buildwise.core import get_client
from config import COLLECTION, REPORT_DIR

# Deliberately line-anchored. A URL or date embedded in a regulation paragraph is
# content, not footer noise. Page-margin stripping itself is done during PDF ingest.
FOOTER_LINE_PATTERNS = [
    re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.I),
    re.compile(r"^\s*(?:page\s*)?\d+\s*(?:/|of)\s*\d+\s*$", re.I),
    re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$"),
    re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\s*$", re.I),
]
WORD_EDGE = re.compile(r"^[a-zA-Z]{1,2}\b|\b[a-zA-Z]{1,2}$")


def is_footer_line(line):
    return any(pattern.fullmatch(line or "") for pattern in FOOTER_LINE_PATTERNS)


def main():
    client = get_client()
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            COLLECTION, limit=256, offset=offset,
            with_payload=True, with_vectors=False
        )
        points.extend(batch)
        if offset is None:
            break

    total = len(points)
    if not total:
        print("No chunks found. Run python ingest.py first.")
        return

    checks = Counter()
    normalized = []
    footer_examples = []
    for point in points:
        payload = point.payload or {}
        text = payload.get("text", "") or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        flagged = [line for line in lines if is_footer_line(line)]

        checks["empty"] += not bool(text.strip())
        checks["footer_noise"] += bool(flagged)
        checks["literal_backslash_n"] += "\\n" in text
        checks["empty_section"] += not bool((payload.get("section") or "").strip())
        checks["mid_word_edge"] += bool(WORD_EDGE.search(text.strip()))
        if flagged and len(footer_examples) < 12:
            footer_examples.append({
                "source": payload.get("source", "unknown"),
                "page": payload.get("page_number", "?"),
                "lines": flagged[:3],
            })
        compact = re.sub(r"\s+", " ", text.strip()).lower()
        normalized.append(compact)

    counts = Counter(normalized)
    checks["duplicated_text"] = sum(1 for value in normalized if value and counts[value] > 1)
    issue_order = (
        "empty", "footer_noise", "duplicated_text", "literal_backslash_n",
        "mid_word_edge", "empty_section"
    )
    lines = [
        "# Chunk Quality Report", "", f"Total chunks: {total}", "",
        "Footer detection: standalone page-artifact lines only; embedded URLs/dates are not flagged.", "",
        "| Issue | Chunks | Percentage |", "|---|---:|---:|"
    ]
    for key in issue_order:
        value = int(checks[key])
        lines.append(f"| {key} | {value} | {value / total * 100:.1f}% |")

    if footer_examples:
        lines.extend(["", "## Footer examples requiring inspection", ""])
        for item in footer_examples:
            lines.append(
                f"- **{item['source']}**, page {item['page']}: "
                + " | ".join(item["lines"])
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "chunk_quality.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
