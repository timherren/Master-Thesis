#!/usr/bin/env bash
# ============================================================
# start.sh — Run this script on Linux to launch
#             the TRAM-DAG Causal Analysis Application
#
# Usage:  chmod +x start.sh && ./start.sh
# ============================================================
set -e

cd "$(dirname "$0")/.."

APP_URL="http://localhost:3838"
OLLAMA_MODEL="llama3.2"

echo ""
echo "======================================================"
echo "  TRAM-DAG Causal Analysis Application — Starting …"
echo "======================================================"
echo ""

# ---- Check Docker is installed ----
if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker is not installed."
  echo ""
  echo "Please install Docker:"
  echo "  https://docs.docker.com/engine/install/"
  echo ""
  exit 1
fi

# ---- Check Docker daemon is running ----
if ! docker info &>/dev/null; then
  echo "ERROR: Docker is installed but not running."
  echo ""
  echo "Please start Docker with:  sudo systemctl start docker"
  echo "Then try again."
  echo ""
  exit 1
fi

# ---- Start Ollama natively (GPU-accelerated) ----
OLLAMA_OK=false

if command -v ollama &>/dev/null; then
  echo "Found Ollama installation."

  # Start Ollama server if not already running
  if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "Starting Ollama server …"
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    for i in $(seq 1 15); do
      if curl -sf http://localhost:11434/api/tags &>/dev/null; then
        break
      fi
      sleep 1
    done
  fi

  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "✓ Ollama server is running."

    if ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
      echo "✓ Model '$OLLAMA_MODEL' is available."
    else
      echo "Pulling model '$OLLAMA_MODEL' (first time only, ~2 GB) …"
      if ollama pull "$OLLAMA_MODEL"; then
        echo "✓ Model '$OLLAMA_MODEL' pulled successfully."
      else
        echo "⚠ Could not pull model — LLM interpretations will be disabled."
      fi
    fi
    OLLAMA_OK=true
  else
    echo "⚠ Could not start Ollama server."
  fi
else
  echo "⚠ Ollama is not installed."
  echo "  To enable AI interpretations, install Ollama from:"
  echo "  https://ollama.com/download"
  echo "  Then run this script again."
fi

if [ "$OLLAMA_OK" = true ]; then
  echo ""
  echo "✓ Ollama is ready (GPU-accelerated). AI interpretations enabled."
else
  echo ""
  echo "Continuing without Ollama — the app works fully, AI text"
  echo "summaries are just disabled."
fi
echo ""

# ---- Create output directory ----
mkdir -p output

# ---- Build and start Docker container ----
echo "Building and starting the app container …"
echo "(First run may take 10-15 min to download all dependencies.)"
echo ""

docker compose up --build -d

# ---- Wait for the Shiny app to be reachable ----
echo ""
echo "Waiting for the app to become ready …"
for i in $(seq 1 180); do
  if curl -sf "$APP_URL" >/dev/null 2>&1; then
    echo ""
    echo "======================================================"
    echo "  App is ready!"
    echo "  $APP_URL"
    echo "======================================================"
    echo ""
    echo "  Upload your CSV data file in the app to get started."
    echo "  Experiment results are saved to the ./output/ folder."
    echo ""
    xdg-open "$APP_URL" 2>/dev/null || echo "Open $APP_URL in your browser."
    echo ""
    echo "To stop the app, run:  STOP_APP/stop.sh"
    echo "or:  docker compose down"
    exit 0
  fi
  printf "."
  sleep 2
done

echo ""
echo "WARNING: App did not respond within 6 minutes."
echo "It may still be starting — first-time build can be slow."
echo "Check status with:  docker compose logs -f"
echo "When ready, open:   $APP_URL"
echo ""
