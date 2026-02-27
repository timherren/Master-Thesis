#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

APP_URL="http://localhost:8000"
DECISION_MODEL="llama3.2:1b"
INTERPRET_MODEL="llama3.2:latest"

echo ""
echo "======================================================"
echo "  Causal AI Chatbot — Starting ..."
echo "======================================================"
echo ""

if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker is not installed."
  echo "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
  read -n 1 -s -r -p "Press any key to close ..."
  exit 1
fi

if ! docker info &>/dev/null; then
  echo "ERROR: Docker is installed but not running."
  read -n 1 -s -r -p "Press any key to close ..."
  exit 1
fi

OLLAMA_OK=false
if command -v ollama &>/dev/null; then
  echo "Found Ollama installation."
  if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "Starting Ollama server ..."
    ollama serve &>/dev/null &
    for i in $(seq 1 15); do
      if curl -sf http://localhost:11434/api/tags &>/dev/null; then
        break
      fi
      sleep 1
    done
  fi

  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "Ollama server is running."
    if ! ollama list 2>/dev/null | grep -q "$DECISION_MODEL"; then
      echo "Pulling decision model '$DECISION_MODEL' ..."
      ollama pull "$DECISION_MODEL" || true
    fi
    if ! ollama list 2>/dev/null | grep -q "$INTERPRET_MODEL"; then
      echo "Pulling interpretation model '$INTERPRET_MODEL' ..."
      ollama pull "$INTERPRET_MODEL" || true
    fi
    OLLAMA_OK=true
  fi
fi

if [ "$OLLAMA_OK" = true ]; then
  echo "Ollama is ready."
else
  echo "WARNING: Ollama not detected/reachable. Start it manually for LLM features."
fi
echo ""

mkdir -p app/runtime
docker compose up --build -d

echo "Waiting for app at $APP_URL ..."
for i in $(seq 1 180); do
  if curl -sf "$APP_URL" >/dev/null 2>&1; then
    echo ""
    echo "App is ready. Opening browser ..."
    open "$APP_URL"
    echo "To stop: STOP_APP/stop.command"
    read -n 1 -s -r -p "Press any key to close ..."
    exit 0
  fi
  printf "."
  sleep 2
done

echo ""
echo "WARNING: App did not respond yet. Check logs with: docker compose logs -f"
read -n 1 -s -r -p "Press any key to close ..."
