#!/usr/bin/env bash
# ============================================================
# start.sh — Run this script on Linux to launch
#             the DAG Validator Agent
#
#   chmod +x start.sh   (only needed once)
#   ./start.sh
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
  echo "Install Docker Engine:"
  echo "  https://docs.docker.com/engine/install/"
  echo ""
  echo "After installing, make sure the Docker service is running:"
  echo "  sudo systemctl start docker"
  echo ""
  exit 1
fi

# ---- Check Docker daemon is running ----
if ! docker info &>/dev/null; then
  echo "ERROR: Docker is installed but the daemon is not running."
  echo ""
  echo "Start it with:"
  echo "  sudo systemctl start docker"
  echo ""
  echo "To start Docker automatically on boot:"
  echo "  sudo systemctl enable docker"
  echo ""
  exit 1
fi

# ---- Check docker compose is available ----
if ! docker compose version &>/dev/null; then
  echo "ERROR: 'docker compose' plugin is not available."
  echo ""
  echo "Install it via your package manager or from:"
  echo "  https://docs.docker.com/compose/install/linux/"
  echo ""
  exit 1
fi

# ---- Check Ollama is installed ----
if ! command -v ollama &>/dev/null; then
  echo "ERROR: Ollama is not installed."
  echo ""
  echo "Install with one command:"
  echo "  curl -fsSL https://ollama.com/install.sh | sh"
  echo ""
  echo "Or visit: https://ollama.com/download/linux"
  echo ""
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
    echo ""
    echo "Try starting it manually:"
    echo "  ollama serve"
    echo ""
    echo "Or check the systemd service:"
    echo "  sudo systemctl start ollama"
    echo ""
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
    echo "  App is ready!"
    echo "  $APP_URL"
    echo "======================================================"
    echo ""
    # Try to open a browser (works on most desktop Linux)
    if command -v xdg-open &>/dev/null; then
      xdg-open "$APP_URL" 2>/dev/null || true
    elif command -v sensible-browser &>/dev/null; then
      sensible-browser "$APP_URL" 2>/dev/null || true
    else
      echo "Open your browser and navigate to: $APP_URL"
    fi
    echo "To stop the app, run:  ./STOP_APP/stop.sh"
    echo "or:  docker compose down"
    echo ""
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
