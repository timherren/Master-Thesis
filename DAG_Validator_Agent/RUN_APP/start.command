#!/usr/bin/env bash
# ============================================================
# start.command — Double-click this file on macOS to launch
#                 the DAG Validator Agent
# ============================================================
set -e

cd "$(dirname "$0")/.."

APP_URL="http://localhost:3838"
OLLAMA_MODEL="llama3.2:latest"
OLLAMA_DAG_MODEL="llama3.2:latest"

echo ""
echo "======================================================"
echo "  DAG Validator Agent — Starting …"
echo "======================================================"
echo ""

# ---- Check Docker is installed ----
if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker is not installed."
  echo ""
  echo "Please install Docker Desktop from:"
  echo "  https://www.docker.com/products/docker-desktop/"
  echo ""
  echo "After installing, open Docker Desktop and wait for it"
  echo "to finish starting, then double-click this file again."
  echo ""
  read -n 1 -s -r -p "Press any key to close …"
  exit 1
fi

# ---- Check Docker daemon is running ----
if ! docker info &>/dev/null; then
  echo "ERROR: Docker is installed but not running."
  echo ""
  echo "Please open Docker Desktop and wait for the whale icon"
  echo "in the menu bar to stop animating, then try again."
  echo ""
  read -n 1 -s -r -p "Press any key to close …"
  exit 1
fi

# ---- Check Ollama is installed ----
if ! command -v ollama &>/dev/null; then
  echo "ERROR: Ollama is not installed."
  echo ""
  echo "Please install Ollama from:"
  echo "  https://ollama.com/download"
  echo ""
  echo "On Mac: download, open the .dmg, drag to Applications."
  echo "Ollama runs in the background automatically after install."
  echo ""
  read -n 1 -s -r -p "Press any key to close …"
  exit 1
fi

# ---- Ensure Ollama is running ----
if ! curl -sf http://localhost:11434/ &>/dev/null; then
  echo "Starting Ollama …"
  ollama serve &>/dev/null &
  OLLAMA_PID=$!
  for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/ &>/dev/null; then
      break
    fi
    sleep 1
  done
  if ! curl -sf http://localhost:11434/ &>/dev/null; then
    echo "ERROR: Could not start Ollama."
    echo "Try opening the Ollama app manually, then run this script again."
    echo ""
    read -n 1 -s -r -p "Press any key to close …"
    exit 1
  fi
fi
echo "Ollama is running."

# ---- Ensure models are pulled ----
if ! ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
  echo "Pulling model $OLLAMA_MODEL (~2 GB, one-time download) …"
  ollama pull "$OLLAMA_MODEL"
fi
echo "Model $OLLAMA_MODEL is ready."

if [ "$OLLAMA_DAG_MODEL" != "$OLLAMA_MODEL" ]; then
  if ! ollama list 2>/dev/null | grep -q "$OLLAMA_DAG_MODEL"; then
    echo "Pulling model $OLLAMA_DAG_MODEL (DAG proposals) …"
    ollama pull "$OLLAMA_DAG_MODEL"
  fi
  echo "Model $OLLAMA_DAG_MODEL is ready."
fi

# ---- Build and start the Shiny app ----
echo ""
echo "Building and starting the app …"
echo "(First run takes a few minutes to set up the R environment.)"
echo ""

docker compose up --build -d

# ---- Wait for the Shiny app to be reachable ----
echo ""
echo "Waiting for the app to become ready …"
for i in $(seq 1 90); do
  if curl -sf "$APP_URL" >/dev/null 2>&1; then
    echo ""
    echo "======================================================"
    echo "  App is ready!  Opening browser …"
    echo "  $APP_URL"
    echo "======================================================"
    echo ""
    open "$APP_URL"
    echo "To stop the app, double-click  STOP_APP/stop.command"
    echo "or run:  docker compose down"
    echo ""
    echo "You can close this terminal window."
    exit 0
  fi
  printf "."
  sleep 2
done

echo ""
echo "WARNING: App did not respond within 3 minutes."
echo "Check status with:  docker compose logs -f"
echo "When ready, open:   $APP_URL"
echo ""
read -n 1 -s -r -p "Press any key to close …"
