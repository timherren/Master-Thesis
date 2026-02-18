#!/usr/bin/env bash
# ============================================================
# stop.command — Double-click this file on macOS to stop
#                the TRAM-DAG Causal Analysis Application
# ============================================================
set -e

cd "$(dirname "$0")/.."

echo ""
echo "======================================================"
echo "  TRAM-DAG Causal Analysis Application — Stopping …"
echo "======================================================"
echo ""

docker compose down 2>/dev/null

echo ""
echo "App container stopped."
echo "Experiment results remain in the ./output/ folder."
echo ""
echo "Note: Ollama is still running in the background."
echo "To stop it:  pkill ollama"
echo ""
read -n 1 -s -r -p "Press any key to close …"
