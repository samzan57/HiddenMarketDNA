@echo off
set TASK_NAME=HiddenMarketDNA_LiveTrader
set BATCH=%~dp0run_live_trader.bat

schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%BATCH%\"" ^
  /SC WEEKLY ^
  /D MON ^
  /ST 15:35 ^
  /RU "%USERDOMAIN%\%USERNAME%" ^
  /RL HIGHEST ^
  /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  Tache creee avec succes !
    echo  Le trader se lancera automatiquement chaque lundi a 15h35.
) else (
    echo  ERREUR : relancer en tant qu'Administrateur
)

pause
