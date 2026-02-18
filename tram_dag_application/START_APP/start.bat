@echo off
REM ============================================================
REM start.bat — Double-click this file on Windows to launch
REM             the TRAM-DAG Causal Analysis Application
REM ============================================================
cd /d "%~dp0.."

set APP_URL=http://localhost:3838
set OLLAMA_MODEL=llama3.2

echo.
echo ======================================================
echo   TRAM-DAG Causal Analysis Application — Starting ...
echo ======================================================
echo.

REM ---- Check Docker is installed ----
where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
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
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is installed but not running.
    echo.
    echo Please open Docker Desktop and wait for it to finish
    echo starting, then try again.
    echo.
    pause
    exit /b 1
)

REM ---- Start Ollama natively (GPU-accelerated) ----
set OLLAMA_OK=false

where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found Ollama installation.

    REM Check if Ollama is already running
    curl -sf http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Starting Ollama server ...
        start /B ollama serve >nul 2>&1
        REM Wait up to 15 seconds for it to become ready
        set /a WAIT=0
        :OLLAMA_WAIT
        set /a WAIT+=1
        if %WAIT% GTR 15 goto OLLAMA_CHECK
        timeout /t 1 /nobreak >nul
        curl -sf http://localhost:11434/api/tags >nul 2>&1
        if %ERRORLEVEL% NEQ 0 goto OLLAMA_WAIT
    )

    :OLLAMA_CHECK
    curl -sf http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo Ollama server is running.

        REM Pull model if not available
        ollama list 2>nul | findstr /C:"%OLLAMA_MODEL%" >nul 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo Pulling model '%OLLAMA_MODEL%' (first time only, ~2 GB) ...
            ollama pull %OLLAMA_MODEL%
        ) else (
            echo Model '%OLLAMA_MODEL%' is available.
        )
        set OLLAMA_OK=true
    ) else (
        echo WARNING: Could not start Ollama server.
    )
) else (
    echo WARNING: Ollama is not installed.
    echo   To enable AI interpretations, install Ollama from:
    echo   https://ollama.com/download
    echo   Then double-click this file again.
)

if "%OLLAMA_OK%"=="true" (
    echo.
    echo Ollama is ready (GPU-accelerated). AI interpretations enabled.
) else (
    echo.
    echo Continuing without Ollama — the app works fully, AI text
    echo summaries are just disabled.
)
echo.

REM ---- Create output directory ----
if not exist output mkdir output

REM ---- Build and start Docker container ----
echo Building and starting the app container ...
echo (First run may take 10-15 min to download all dependencies.)
echo.

docker compose up --build -d

REM ---- Wait for the Shiny app to be reachable ----
echo.
echo Waiting for the app to become ready ...
set /a COUNTER=0

:WAIT_LOOP
set /a COUNTER+=1
if %COUNTER% GTR 180 (
    echo.
    echo WARNING: App did not respond within 6 minutes.
    echo It may still be starting — first-time build can be slow.
    echo Check status with:  docker compose logs -f
    echo When ready, open:   %APP_URL%
    echo.
    pause
    exit /b 0
)

curl -sf %APP_URL% >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================
    echo   App is ready!  Opening browser ...
    echo   %APP_URL%
    echo ======================================================
    echo.
    echo   Upload your CSV data file in the app to get started.
    echo   Experiment results are saved to the .\output\ folder.
    echo.
    start %APP_URL%
    echo To stop the app, double-click  STOP_APP\stop.bat
    echo or run:  docker compose down
    echo.
    echo You can close this window.
    pause
    exit /b 0
)

timeout /t 2 /nobreak >nul
echo|set /p="."
goto WAIT_LOOP
