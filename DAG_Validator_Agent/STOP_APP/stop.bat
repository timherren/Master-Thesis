@echo off
REM ============================================================
REM stop.bat -- Double-click this file on Windows to stop
REM             the DAG Validator Agent
REM ============================================================
cd /d "%~dp0\.."

echo.
echo ======================================================
echo   DAG Validator Agent -- Stopping ...
echo ======================================================
echo.

docker compose down

echo.
echo App stopped. Ollama continues running in the background
echo -- no re-download needed on next start.
echo.
pause
