#!/usr/bin/env bash
# ============================================================
# stop.command — Double-click this file on macOS to stop
#                the DAG Validator Agent
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
read -n 1 -s -r -p "Press any key to close …"
