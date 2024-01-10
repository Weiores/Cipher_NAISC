@echo off
REM Cipher_NAISC – Windows start script
REM Launches the backend API, UI layer, and frontend dashboard.

echo ============================================================
echo  Cipher_NAISC – Starting
echo ============================================================

if not exist ".env" (
    echo [start] WARNING: .env not found. Run setup.bat first.
)

python launcher.py %*
