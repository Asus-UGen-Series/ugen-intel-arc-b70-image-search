@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "PROJECT_ROOT=%~dp0"
set "PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "REPORT_DIR=%PROJECT_ROOT%.verification"
set "REPORT_LATEST=%REPORT_DIR%\quick-test-latest.txt"
set "REPORT=%REPORT_DIR%\quick-test-run-%RANDOM%-%RANDOM%.txt"

if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"
>"%REPORT%" echo Visual Search quick test - %DATE% %TIME%

if not exist "%PYTHON%" (
  echo [FAIL] 找不到 .venv\Scripts\python.exe
  echo 請先依 README 建立開發環境。
  >>"%REPORT%" echo [FAIL] Missing .venv\Scripts\python.exe
  exit /b 2
)
if not exist "%PROJECT_ROOT%tests" (
  echo [FAIL] 找不到公開測試資料夾 tests。
  >>"%REPORT%" echo [FAIL] Missing tests folder.
  exit /b 4
)
"%PYTHON%" -c "import pytest" >>"%REPORT%" 2>&1
if errorlevel 1 (
  echo [FAIL] .venv 尚未安裝 pytest。
  >>"%REPORT%" echo [FAIL] pytest is not installed in .venv.
  exit /b 3
)

if /i "%~1"=="test" goto test_only
if /i "%~1"=="diagnose" goto diagnose_only
if /i "%~1"=="repair" goto repair_only
if /i "%~1"=="app" goto app_only
if /i "%~1"=="evaluate" goto evaluate_only
if /i "%~1"=="all" goto all_checks
if not "%~1"=="" goto usage

:menu
echo.
echo Visual Search 開發者快速工具
echo   1. 執行測試
echo   2. 執行安全診斷
echo   3. 接續未完成索引
echo   4. 啟動 Debug UI
echo   5. 開啟 test100 客觀評測
echo   6. 測試、診斷後啟動 Debug UI
echo   Q. 離開
set "MENU_CHOICE="
set /p "MENU_CHOICE=請選擇 [4]："
if not defined MENU_CHOICE set "MENU_CHOICE=4"
if /i "%MENU_CHOICE%"=="1" goto test_only
if /i "%MENU_CHOICE%"=="2" goto diagnose_only
if /i "%MENU_CHOICE%"=="3" goto repair_only
if /i "%MENU_CHOICE%"=="4" goto app_only
if /i "%MENU_CHOICE%"=="5" goto evaluate_only
if /i "%MENU_CHOICE%"=="6" goto all_checks
if /i "%MENU_CHOICE%"=="Q" exit /b 0
echo 選項無效。
goto menu

:usage
echo Usage: quick-test.cmd [test^|diagnose^|repair^|app^|evaluate^|all]
exit /b 64

:all_checks
call :run_tests
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_diagnose
if errorlevel 1 exit /b %ERRORLEVEL%
goto app_only

:test_only
call :run_tests
set "RESULT=%ERRORLEVEL%"
call :publish_report
exit /b %RESULT%

:diagnose_only
call :run_diagnose
set "RESULT=%ERRORLEVEL%"
call :publish_report
exit /b %RESULT%

:repair_only
echo [REPAIR] 接續未完成的索引階段...
>>"%REPORT%" echo [REPAIR] Resuming incomplete index stages...
"%PYTHON%" "%PROJECT_ROOT%launcher.py" --resume-index
set "RESULT=%ERRORLEVEL%"
>>"%REPORT%" echo [REPAIR] Exit code %RESULT%
call :publish_report
exit /b %RESULT%

:evaluate_only
call :publish_report
"%PYTHON%" "%PROJECT_ROOT%evaluate_test100.py"
exit /b %ERRORLEVEL%

:app_only
set "VISUAL_SEARCH_DEVELOPER_MODE=1"
echo [APP] 啟動含搜尋診斷頁面的 Visual Search Debug UI...
>>"%REPORT%" echo [APP] Developer diagnostics enabled via VISUAL_SEARCH_DEVELOPER_MODE=1
call :publish_report
call "%PROJECT_ROOT%scripts\run-dev.cmd"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
  echo [FAIL] UI 結束代碼 %RESULT%。報告：%REPORT%
  type "%REPORT%"
)
call :publish_report
exit /b %RESULT%

:run_tests
echo [TEST] 執行單元與整合測試...
>>"%REPORT%" echo [TEST] Running unit and integration tests...
"%PYTHON%" -m pytest -q >>"%REPORT%" 2>&1
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
  echo [FAIL] 測試失敗，代碼 %RESULT%。報告：%REPORT%
  type "%REPORT%"
  call :publish_report
  exit /b %RESULT%
)
echo [PASS] 測試通過
>>"%REPORT%" echo [PASS] Tests
exit /b 0

:run_diagnose
echo [DIAGNOSE] 檢查模型、GPU、區域索引與資料庫...
>>"%REPORT%" echo [DIAGNOSE] Checking models, GPU, region index, and database...
"%PYTHON%" "%PROJECT_ROOT%launcher.py" --diagnose >>"%REPORT%" 2>&1
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
  echo [FAIL] 診斷失敗，代碼 %RESULT%。報告：%REPORT%
  type "%REPORT%"
  call :publish_report
  exit /b %RESULT%
)
echo [PASS] 診斷完成。報告：%REPORT%
>>"%REPORT%" echo [PASS] Diagnose
exit /b 0

:publish_report
copy /y "%REPORT%" "%REPORT_LATEST%" >nul 2>&1
exit /b 0
