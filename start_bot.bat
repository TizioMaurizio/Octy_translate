@echo off
cd /d "%~dp0"

set DOCKER="C:\Program Files\Docker\Docker\resources\bin\docker.exe"

echo Stopping any existing bot instances...
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr "PID:"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr "telegram_bot" >nul && taskkill /PID %%a /F >nul 2>nul
)

echo Starting Telegram Bot API server (Docker)...
%DOCKER% compose -f docker-compose.telegram-api.yml --env-file .env.telegram-api up -d 2>nul
if errorlevel 1 (
    echo Docker non disponibile o container gia' attivo, proseguo...
)

echo Waiting for API server to be ready...
timeout /t 3 /nobreak >nul

echo Starting Telegram bot...
call conda activate base
python telegram_bot.py
pause
