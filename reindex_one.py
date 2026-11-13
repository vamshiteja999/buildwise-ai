"""Re-index one document with the current extraction code.
Usage (stop the Streamlit app first):  python reindex_one.py "BUILDING CODE SERVICES ADMINISTRATIVE GUIDELINE.pdf"
"""
import sys
from qdrant_client import models
from config import COLLECTION, DATA_DIR
from buildwise.core import get_client, ingest_all

name = sys.argv[1] if len(sys.argv) > 1 else ""
if not name or not list(DATA_DIR.rglob(name)):
    raise SystemExit(f"File not found under {DATA_DIR}: {name!r}. Files there: "
                     + ", ".join(p.name for p in DATA_DIR.rglob('*') if p.is_file()))
get_client().delete(collection_name=COLLECTION, points_selector=models.FilterSelector(
    filter=models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=name))])))
print("Old chunks removed. Re-indexing...")
print(ingest_all())
