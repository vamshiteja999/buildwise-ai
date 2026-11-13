#!/usr/bin/env bash
# Usage: bash run_model_compare.sh qwen2.5:7b-instruct
# Runs the routing check and the full answer eval for one model with TEMPERATURE=0, retry off.
# Close the Streamlit app first. Reports: reports/routing_<tag>.md-style files and reports/run_<tag>.md
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:?give the model name, e.g. qwen2.5:7b-instruct}"
TAG="$(echo "$MODEL" | tr ':/.' '___')"
source .venv/bin/activate
ollama pull "$MODEL"
ollama pull nomic-embed-text
echo "== routing check ($MODEL) =="
ANSWER_MODEL="$MODEL" ROUTER_MODEL="$MODEL" TEMPERATURE=0 python eval_routing.py
echo "== answer eval ($MODEL) =="
ANSWER_MODEL="$MODEL" ROUTER_MODEL="$MODEL" TEMPERATURE=0 AGENT_RETRY=false EVAL_REPORT="run_${TAG}.md" python eval.py > /dev/null
echo "Done. Report: reports/run_${TAG}.md"
grep -A12 "^| Metric" "reports/run_${TAG}.md" | head -14
