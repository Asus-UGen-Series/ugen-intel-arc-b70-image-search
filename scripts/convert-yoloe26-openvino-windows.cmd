@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv\Scripts\python.exe
  echo Run scripts\setup-dev.cmd first.
  exit /b 2
)

echo Converting the recommended YOLOE-26S Prompt-Free model to OpenVINO FP16.
echo Conversion dependencies are installed into an isolated artifacts folder.
".venv\Scripts\python.exe" "scripts\convert-yoloe26-openvino-wsl.py" --scale s --install-local --force
if errorlevel 1 (
  echo [ERROR] YOLOE-26 conversion failed.
  exit /b 1
)

echo.
echo Model installed. Restart Visual Search to load it.
exit /b 0
