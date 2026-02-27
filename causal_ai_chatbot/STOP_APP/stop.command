#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
echo "Stopping Causal AI Chatbot containers ..."
docker compose down 2>/dev/null || true
echo "Stopped."
echo "Runtime artifacts are preserved in ./app/runtime."
echo "Note: Ollama keeps running on host. Stop manually if needed."
read -n 1 -s -r -p "Press any key to close ..."
