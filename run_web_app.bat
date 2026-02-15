@echo off
echo ==================================================
echo   METABOLIC DIGITAL TWIN - WEB APP LAUNCHER
echo ==================================================
echo.
echo 1. Starting Backend API (FastAPI)...
start "MDT Backend" cmd /k "cd mobile_app\backend && python api.py"

echo 2. Waiting for API to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

echo 3. Opening Frontend Dashboard...
start "" "mobile_app\frontend\index.html"

echo.
echo [SUCCESS] System is live!
echo - Backend: http://localhost:8001
echo - Frontend: Launched in Browser
echo.
pause
