@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing Windows .venv.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launcher.py
if errorlevel 1 pause
