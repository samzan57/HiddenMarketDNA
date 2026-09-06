@echo off
chcp 65001 >nul
setlocal

rem Repo root = parent of this script's own folder (scripts\..)
set PROJECT_DIR=%~dp0..
set LOG_DIR=%PROJECT_DIR%\logs

rem Uses "python" from PATH — set an absolute interpreter path here if needed.
set PYTHON=python

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set LOG_FILE=%LOG_DIR%\live_trader_%DT:~0,8%.log

echo ============================================================ >> "%LOG_FILE%"
echo Demarrage : %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

cd /d "%PROJECT_DIR%"
"%PYTHON%" -m src.live_trader >> "%LOG_FILE%" 2>&1

echo Fin : %DATE% %TIME% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

endlocal
