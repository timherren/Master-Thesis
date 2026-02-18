@echo off
REM ============================================================
REM stop.bat — Double-click this file on Windows to stop
REM             the TRAM-DAG Causal Analysis Application
REM ============================================================
cd /d "%~dp0.."

echo.
echo ======================================================
echo   TRAM-DAG Causal Analysis Application — Stopping ...
echo ======================================================
echo.

docker compose down >nul 2>&1

echo.
echo App container stopped.
echo Experiment results remain in the .\output\ folder.
echo.
echo Note: Ollama is still running in the background.
echo To stop it:  taskkill /IM ollama.exe /F
echo.
pause
