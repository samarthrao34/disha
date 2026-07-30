@echo off
title Disha
cd /d "%~dp0"

if not exist "package.json" (
  echo.
  echo   ERROR: package.json is not in this folder.
  echo   Keep START.bat inside the Disha project folder.
  echo.
  pause
  exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
  echo.
  echo   Node.js was not found, so Disha can't start.
  echo.
  echo   Install it from https://nodejs.org/
  echo.
  pause
  exit /b 1
)

if not exist "node_modules\vite" (
  echo.
  echo   Installing Disha's voice dependencies...
  call npm install
  if errorlevel 1 (
    echo   Dependency installation failed.
    pause
    exit /b 1
  )
)

echo.
echo   Building Disha's React app...
call npm run build
if errorlevel 1 (
  echo.
  echo   Disha's frontend failed to build.
  pause
  exit /b 1
)

netstat -ano | findstr /r /c:"127.0.0.1:8777 .*LISTENING" >nul
if not errorlevel 1 (
  echo.
  echo   Disha is already running - opening her now...
  start "" http://localhost:8777/
  exit /b 0
)

echo.
echo   Starting Disha with Gemini Leda voice...
echo   http://localhost:8777/
echo.
echo   Keep this window open while you talk to her.
echo   Press Ctrl+C to stop.
echo.

rem  Open the browser a moment later, so the server is already listening.
start "" /min cmd /c "ping -n 3 127.0.0.1 >nul & start "" http://localhost:8777/"

node server.mjs
if errorlevel 1 (
  echo.
  echo   The server failed to start - port 8777 may already be in use.
  echo   Close any other Disha window and try again.
  echo.
  pause
)
