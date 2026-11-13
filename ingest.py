import sys
from buildwise.core import ingest_all

if __name__ == "__main__":
    try:
        ingest_all()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Fix the exact issue shown above, then rerun: python ingest.py", file=sys.stderr)
        raise SystemExit(1)
