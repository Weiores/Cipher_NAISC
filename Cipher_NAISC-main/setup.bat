@echo off
REM Cipher_NAISC – Windows setup script
REM Run once to install all Python and Node dependencies.

echo ============================================================
echo  Cipher_NAISC – Setup
echo ============================================================

REM ── Copy .env if needed ──────────────────────────────────────
if not exist ".env" (
    if exist ".env.example" (
        echo [setup] Copying .env.example to .env ...
        copy ".env.example" ".env" >nul
        echo [setup] .env created. Edit it and fill in your API keys before starting.
    ) else (
        echo [setup] WARNING: .env.example not found. Create .env manually.
    )
) else (
    echo [setup] .env already exists – skipping copy.
)

REM ── Python dependencies ───────────────────────────────────────
echo.
echo [setup] Installing Python dependencies ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [setup] ERROR: pip install failed. Make sure Python 3.10+ is on PATH.
    exit /b 1
)

REM ── Frontend dependencies ─────────────────────────────────────
echo.
echo [setup] Installing frontend (npm) dependencies ...
cd frontend
call npm install
if errorlevel 1 (
    echo [setup] ERROR: npm install failed. Make sure Node.js 18+ is on PATH.
    cd ..
    exit /b 1
)
cd ..

echo.
echo ============================================================
echo  Setup complete!
echo  Next steps:
echo    1. Edit .env and add your GROQ_API_KEY (and Telegram keys)
echo    2. Double-click start.bat  (or: python launcher.py)
echo ============================================================
pause
