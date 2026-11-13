"""Why was a question refused? Shows retrieval rank of the answer chunk and the model's raw reply.
Usage (stop the Streamlit app first):  python debug_refusal.py
"""
import buildwise.core as core
from config import COLLECTION

CASES = [
    ("What is the revision date of BD-002 and who approved it?", "8/14/24"),
    ("What is the minimum font size for text on plans?", "at least 10-pt"),
]


def find_chunks(needle):
    client, out, offset = core.get_client(), [], None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if needle.lower() in (pl.get("text", "") or "").lower():
                out.append(pl)
        if offset is None:
            return out


raw_box = {}
_orig = core.request_json


def _spy(url, payload, *a, **k):
    data = _orig(url, payload, *a, **k)
    raw_box["raw"] = data.get("response", "")
    raw_box["prompt_chars"] = len(payload.get("prompt", ""))
    return data


core.request_json = _spy
core.health_check()
for question, needle in CASES:
    print("=" * 90, "\nQ:", question, "| looking for:", needle)
    holders = find_chunks(needle)
    print(f"Chunks in the index that contain it: {len(holders)}")
    for h in holders:
        print(f"   page {h.get('page_number')}, section {h.get('section')}, {len(h.get('text','').split())} words")
    ranked = core.hybrid_search(question, 30)
    rank = next((i for i, p in enumerate(ranked, 1)
                 if needle.lower() in ((p.payload or {}).get("text", "") or "").lower()), None)
    print("Rank of answer chunk in top 30:", rank or "NOT in top 30", "(model sees top", core.FINAL_TOP_K, ")")
    print("Top 5 retrieved:")
    for i, p in enumerate(ranked[:5], 1):
        pl = p.payload or {}
        print(f"   {i}. {pl.get('source')} p{pl.get('page_number')} | {(pl.get('text') or '')[:90]!r}")
    answer, _ = core.answer_question(question)
    print("Model raw reply:", repr(raw_box.get("raw", ""))[:400], "| prompt chars:", raw_box.get("prompt_chars"))
    print("Final answer:", answer[:200])
