#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python 3.11 was not found. Install Python 3.11, then rerun: bash setup.sh"
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"ERROR: BuildWise requires Python 3.11; found {sys.version.split()[0]}. Set PYTHON_BIN to your Python 3.11 executable.")
PY

if [ -d .venv ]; then
  VENV_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  if [ "$VENV_VERSION" != "3.11" ]; then
    echo "Replacing existing .venv (Python ${VENV_VERSION:-unknown}) with Python 3.11..."
    rm -rf .venv
  fi
fi
if [ ! -d .venv ]; then "$PYTHON_BIN" -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip==25.0.1
python -m pip install -r requirements.txt

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: Ollama is not installed. Install Ollama for macOS, start it, then rerun bash setup.sh."
  exit 1
fi
if ! curl -fsS http://localhost:11434/api/tags >/dev/null; then
  echo "ERROR: Ollama is not responding. Open the Ollama app, then rerun bash setup.sh."
  exit 1
fi
ollama pull nomic-embed-text
ollama pull qwen3:1.7b

if [ "${OCR_ENABLED:-true}" = "true" ] && ! command -v tesseract >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing Tesseract with Homebrew..."
    brew install tesseract
  else
    echo "WARNING: Tesseract is unavailable. Text PDFs will work; scanned PDFs will be skipped. Install Tesseract to enable OCR."
  fi
fi

mkdir -p data/raw data/qdrant logs reports
[ -f .env ] || cp .env.example .env
echo
echo "BuildWise setup complete."
echo "Activate with: source .venv/bin/activate"
echo "Put source documents in: $(pwd)/data/raw"
echo "Run ingestion with: python ingest.py"
