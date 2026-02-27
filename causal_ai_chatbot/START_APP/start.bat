@echo off
cd /d "%~dp0.."

set APP_URL=http://localhost:8000
set DECISION_MODEL=llama3.2:1b
set INTERPRET_MODEL=llama3.2:latest

echo.
echo ======================================================
echo   Causal AI Chatbot -- Starting ...
echo ======================================================
echo.

where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not installed.
    echo Install Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is installed but not running.
    pause
    exit /b 1
)

set OLLAMA_OK=false
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found Ollama installation.
    curl -sf http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        start /B ollama serve >nul 2>&1
        timeout /t 3 /nobreak >nul
    )

    curl -sf http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        ollama list 2>nul | findstr /C:"%DECISION_MODEL%" >nul 2>&1
        if %ERRORLEVEL% NEQ 0 ollama pull %DECISION_MODEL%
        ollama list 2>nul | findstr /C:"%INTERPRET_MODEL%" >nul 2>&1
        if %ERRORLEVEL% NEQ 0 ollama pull %INTERPRET_MODEL%
        set OLLAMA_OK=true
    )
)

if "%OLLAMA_OK%"=="true" (
    echo Ollama is ready.
) else (
    echo WARNING: Ollama not detected/reachable. Start it manually for LLM features.
)
echo.

if not exist app\runtime mkdir app\runtime

docker compose up --build -d
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: docker compose failed.
    pause
    exit /b 1
)

echo Waiting for app at %APP_URL% ...
set /a COUNTER=0
:WAIT_LOOP
set /a COUNTER+=1
if %COUNTER% GTR 180 goto WAIT_TIMEOUT
curl -sf %APP_URL% >nul 2>&1
if %ERRORLEVEL% EQU 0 goto APP_READY
timeout /t 2 /nobreak >nul
<nul set /p ="."
goto WAIT_LOOP

:APP_READY
echo.
echo App is ready. Opening browser ...
start "" %APP_URL%
echo To stop: STOP_APP\stop.bat
pause
exit /b 0

:WAIT_TIMEOUT
echo.
echo WARNING: App did not respond yet. Check logs with: docker compose logs -f
pause
exit /b 0
