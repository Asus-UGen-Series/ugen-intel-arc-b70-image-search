@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing Windows .venv.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launcher.py
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo Visual Search exited with code %RESULT%.
exit /b %RESULT%
