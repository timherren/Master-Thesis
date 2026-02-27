#!/bin/bash
# Startup script for Causal AI Agent Chatbot

echo "Starting Causal AI Agent Chatbot..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for .env file (look in multiple locations)
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found"
    echo "   Local default is Ollama (fully offline)."
    echo "   Optional .env (recommended):"
    echo "     LLM_PROVIDER=ollama"
    echo "     LLM_MODEL_DECISION=qwen2.5:3b-instruct"
    echo "     LLM_MODEL_INTERPRETATION=qwen2.5:7b-instruct"
    echo "     OLLAMA_BASE_URL=http://127.0.0.1:11434/v1"
    echo ""
fi

# Check Python dependencies
echo "Checking Python dependencies..."
python -c "import fastapi, uvicorn, pandas, numpy, torch, matplotlib, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing Python dependencies. Installing..."
    pip install -r requirements_chatbot.txt
fi

# Check TRAM-DAG
echo "Checking TRAM-DAG installation..."
python -c "from tramdag import TramDagConfig" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: TRAM-DAG import failed."
    echo "   Install the tramdag package in this environment."
    echo "   Try running: python tests/test_imports.py"
    echo "   Example: pip install tramdag"
    echo ""
    echo "   The chatbot will still start, but TRAM-DAG features won't work."
fi

# Create runtime directories via central config
python - <<'PY'
from app.config import ensure_runtime_dirs
ensure_runtime_dirs()
print("Runtime directories verified.")
PY

echo ""
echo "Starting server..."
echo "   Open http://localhost:8000 in your browser"
echo ""

# Start server
python chatbot_server.py
