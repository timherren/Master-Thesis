#!/usr/bin/env bash
# ============================================================
# stop.sh — Stop the DAG Validator Agent on Linux
# ============================================================
cd "$(dirname "$0")/.."

echo ""
echo "======================================================"
echo "  DAG Validator Agent — Stopping …"
echo "======================================================"
echo ""

docker compose down

echo ""
echo "App stopped. Ollama continues running in the background"
echo "(no re-download needed on next start)."
echo ""
