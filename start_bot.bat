@echo off
cd /d "%~dp0"

set DOCKER="C:\Program Files\Docker\Docker\resources\bin\docker.exe"
set DOCKER_DESKTOP="C:\Program Files\Docker\Docker\Docker Desktop.exe"

echo Stopping any existing bot instances...
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr "PID:"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr "telegram_bot" >nul && taskkill /PID %%a /F >nul 2>nul
)

echo Checking Docker daemon...
%DOCKER% info >nul 2>nul
if errorlevel 1 (
    echo Docker daemon not running, starting Docker Desktop...
    start "" %DOCKER_DESKTOP%
    echo Waiting for Docker daemon to be ready...
    :wait_docker
    timeout /t 2 /nobreak >nul
    %DOCKER% info >nul 2>nul
    if errorlevel 1 goto wait_docker
    echo Docker daemon ready.
)

echo Starting Telegram Bot API server (Docker)...
%DOCKER% compose -f docker-compose.telegram-api.yml --env-file .env.telegram-api up -d
if errorlevel 1 (
    echo ERRORE: impossibile avviare il container Bot API.
    pause
    exit /b 1
)

echo Waiting for API server to be ready...
timeout /t 3 /nobreak >nul

echo Starting Telegram bot...
call conda activate base
python telegram_bot.py
pause
