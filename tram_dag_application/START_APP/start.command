#!/usr/bin/env bash
# ============================================================
# start.command — Double-click this file on macOS to launch
#                 the TRAM-DAG Causal Analysis Application
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

# ---- Start Ollama natively (GPU-accelerated) ----
OLLAMA_OK=false

if command -v ollama &>/dev/null; then
  echo "Found Ollama installation."

  # Start Ollama server if not already running
  if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "Starting Ollama server …"
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Wait up to 15 seconds for it to become ready
    for i in $(seq 1 15); do
      if curl -sf http://localhost:11434/api/tags &>/dev/null; then
        break
      fi
      sleep 1
    done
  fi

  # Check if the server is now reachable
  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "✓ Ollama server is running."

    # Pull the model if not already available
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
  echo "  Then double-click this file again."
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
    echo "  App is ready!  Opening browser …"
    echo "  $APP_URL"
    echo "======================================================"
    echo ""
    echo "  Upload your CSV data file in the app to get started."
    echo "  Experiment results are saved to the ./output/ folder."
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
echo "WARNING: App did not respond within 6 minutes."
echo "It may still be starting — first-time build can be slow."
echo "Check status with:  docker compose logs -f"
echo "When ready, open:   $APP_URL"
echo ""
read -n 1 -s -r -p "Press any key to close …"
