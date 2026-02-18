@echo off
REM ============================================================
REM start.bat — Double-click this file on Windows to launch
REM              the DAG Validator Agent
REM ============================================================

cd /d "%~dp0\.."

set APP_URL=http://localhost:3838
set OLLAMA_MODEL=llama3.2:latest
set OLLAMA_DAG_MODEL=llama3.2:latest

echo.
echo ======================================================
echo   DAG Validator Agent — Starting ...
echo ======================================================
echo.

REM ---- Check Docker is installed ----
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed.
    echo.
    echo Please install Docker Desktop from:
    echo   https://www.docker.com/products/docker-desktop/
    echo.
    echo After installing, open Docker Desktop and wait for it
    echo to finish starting, then double-click this file again.
    echo.
    pause
    exit /b 1
)

REM ---- Check Docker daemon is running ----
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is installed but not running.
    echo.
    echo Please open Docker Desktop and wait for it to finish
    echo starting, then try again.
    echo.
    pause
    exit /b 1
)

REM ---- Check Ollama is installed ----
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Ollama is not installed.
    echo.
    echo Please install Ollama from:
    echo   https://ollama.com/download
    echo.
    echo Run the installer and follow the prompts.
    echo Ollama runs in the background automatically after install.
    echo.
    pause
    exit /b 1
)

REM ---- Ensure Ollama is running ----
curl -sf http://localhost:11434/ >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Ollama ...
    start /b ollama serve >nul 2>&1
    set /a WAIT=0
    :ollama_wait
    if %WAIT% geq 30 goto ollama_fail
    curl -sf http://localhost:11434/ >nul 2>&1
    if %errorlevel% equ 0 goto ollama_ok
    set /a WAIT+=1
    timeout /t 1 /nobreak >nul
    goto ollama_wait
)
:ollama_ok
echo Ollama is running.
goto model_check

:ollama_fail
echo ERROR: Could not start Ollama.
echo Try opening the Ollama app manually, then run this script again.
echo.
pause
exit /b 1

REM ---- Ensure models are pulled ----
:model_check
ollama list 2>nul | findstr /c:"%OLLAMA_MODEL%" >nul 2>&1
if %errorlevel% neq 0 (
    echo Pulling model %OLLAMA_MODEL% (~2 GB, one-time download) ...
    ollama pull %OLLAMA_MODEL%
)
echo Model %OLLAMA_MODEL% is ready.

ollama list 2>nul | findstr /c:"%OLLAMA_DAG_MODEL%" >nul 2>&1
if %errorlevel% neq 0 (
    echo Pulling model %OLLAMA_DAG_MODEL% (DAG proposals) ...
    ollama pull %OLLAMA_DAG_MODEL%
)
echo Model %OLLAMA_DAG_MODEL% is ready.

REM ---- Build and start the Shiny app ----
echo.
echo Building and starting the app ...
echo (First run takes a few minutes to set up the R environment.)
echo.

docker compose up --build -d
if %errorlevel% neq 0 (
    echo.
    echo ERROR: docker compose failed. See messages above.
    echo.
    pause
    exit /b 1
)

REM ---- Wait for the Shiny app to be reachable ----
echo.
echo Waiting for the app to become ready ...

set /a TRIES=0
:waitloop
if %TRIES% geq 90 goto timeout
curl -sf %APP_URL% >nul 2>&1
if %errorlevel% equ 0 goto ready
set /a TRIES+=1
<nul set /p ="."
timeout /t 2 /nobreak >nul
goto waitloop

:ready
echo.
echo ======================================================
echo   App is ready!  Opening browser ...
echo   %APP_URL%
echo ======================================================
echo.
start "" %APP_URL%
echo To stop the app, double-click  STOP_APP\stop.bat
echo or run:  docker compose down
echo.
echo You can close this window.
pause
exit /b 0

:timeout
echo.
echo WARNING: App did not respond within 3 minutes.
echo Check status with:  docker compose logs -f
echo When ready, open:   %APP_URL%
echo.
pause
exit /b 0
