@echo off
cd /d "%~dp0.."
echo Stopping Causal AI Chatbot containers ...
docker compose down >nul 2>&1
echo Stopped.
echo Runtime artifacts are preserved in .\app\runtime.
echo Note: Ollama keeps running on host. Stop manually if needed.
pause
