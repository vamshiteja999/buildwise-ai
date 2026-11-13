"""Diagnostics for the BuildWise index. Run: python diag.py
Writes reports/diagnostics.md and reports/chunks_dump.txt (every chunk, readable)."""
import re
from collections import defaultdict
from buildwise.core import get_client
from config import COLLECTION, REPORT_DIR

NOISE = {
    "print timestamp": re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)", re.I),
    "osha page title": re.compile(r"\|\s*Occupational Safety and Health Administration", re.I),
    "agency line": re.compile(r"^\s*Occupational Safety and Health Administration\s*$", re.I | re.M),
    "page counter": re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.M),
    "osha url": re.compile(r"osha\.gov/laws-regs", re.I),
    "website chrome": re.compile(r"Subscribe|Sign In|Try Free|Building Safety Journal|ICC Family of Solutions|Privacy Policy|Terms of Use", re.I),
    "copyright banner": re.compile(r"COPYRIGHT ©", re.I),
}


def load_points():
    client, points, offset = get_client(), [], None
    while True:
        batch, offset = client.scroll(COLLECTION, limit=256, offset=offset,
                                      with_payload=True, with_vectors=False)
        points.extend(batch)
        if offset is None:
            return points


def main():
    points = load_points()
    if not points:
        raise SystemExit("No chunks found. Run python ingest.py first.")
    docs = defaultdict(lambda: {"type": set(), "n": 0, "words": 0, "no_section": 0,
                                "noise": defaultdict(int), "pages": set(), "ocr": 0})
    examples = defaultdict(list)
    dump = []
    for i, p in enumerate(sorted(points, key=lambda x: ((x.payload or {}).get("source", ""),
                                                        (x.payload or {}).get("page_number", 0))), 1):
        pl = p.payload or {}
        text = pl.get("text", "") or ""
        d = docs[pl.get("source", "unknown")]
        d["type"].add(pl.get("document_type", "?"))
        d["n"] += 1
        d["words"] += len(text.split())
        d["no_section"] += not (pl.get("section") or "").strip()
        d["pages"].add(pl.get("page_number"))
        d["ocr"] += bool(pl.get("ocr_used"))
        for name, rx in NOISE.items():
            hits = rx.findall(text)
            if hits:
                d["noise"][name] += 1
                if len(examples[name]) < 3:
                    line = next((l.strip() for l in text.splitlines() if rx.search(l)), hits[0])
                    examples[name].append(f"{pl.get('source')} p{pl.get('page_number')}: {line[:110]}")
        dump.append(f"--- CHUNK {i} ---\nSOURCE: {pl.get('source')} | TYPE: {pl.get('document_type')} | "
                    f"PAGE: {pl.get('page_number')} | SECTION: {pl.get('section')} | OCR: {pl.get('ocr_used')}\n"
                    f"TEXT: {text}\n")

    out = ["# BuildWise Diagnostics", "", f"Total chunks: {len(points)}", "",
           "| Source | Type | Pages | Chunks | Avg words | No section | OCR chunks | Noise hits |",
           "|---|---|---:|---:|---:|---:|---:|---|"]
    for src, d in sorted(docs.items()):
        noise = ", ".join(f"{k}:{v}" for k, v in d["noise"].items()) or "none"
        out.append(f"| {src[:55]} | {','.join(sorted(d['type']))} | {len(d['pages'])} | {d['n']} | "
                   f"{d['words'] // d['n']} | {d['no_section']} | {d['ocr']} | {noise} |")
    if examples:
        out += ["", "## Noise examples", ""]
        for name, items in examples.items():
            out += [f"**{name}**"] + [f"- {x}" for x in items]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "diagnostics.md").write_text("\n".join(out), encoding="utf-8")
    (REPORT_DIR / "chunks_dump.txt").write_text("\n".join(dump), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved: {REPORT_DIR / 'diagnostics.md'} and {REPORT_DIR / 'chunks_dump.txt'}")


if __name__ == "__main__":
    main()
