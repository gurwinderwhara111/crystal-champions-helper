@echo off
:: ============================================================================
:: Crystal Champions Bot Launcher (Batch Version)
:: ============================================================================
:: LAUNCHES: Collector + Panel for PRACTICE MODE ONLY
:: 
:: ⚠️  LEGAL: WorldWinner ToS bans bots in cash tournaments.
::          Use ONLY in FREE PRACTICE MODE (no cash risk).
:: ============================================================================

:: CONFIGURATION - UPDATE THESE PATHS
set PYTHON="C:\Users\gurwi\AppData\Local\Programs\Python\Python313\pythonw.exe"
set PROJECT_ROOT="D:\random works by gs\Crystal Champions helper"

:: Verify Python exists
if not exist %PYTHON% (
    echo ERROR: Python not found at %PYTHON%
    echo Edit this file and update PYTHON path.
    pause
    exit /b 1
)

:: Verify project exists
if not exist %PROJECT_ROOT% (
    echo ERROR: Project not found at %PROJECT_ROOT%
    echo Edit this file and update PROJECT_ROOT path.
    pause
    exit /b 1
)

:: Legal warning
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  ⚠️  LEGAL WARNING - READ CAREFULLY                          ║
echo ╠══════════════════════════════════════════════════════════════╣
echo ║  WorldWinner ToS EXPLICITLY BANS automated play in cash      ║
echo ║  tournaments. Penalties include:                             ║
echo ║    • Account termination                                     ║
echo ║    • Forfeiture of all winnings/credits                      ║
echo ║    • Disgorgement (repayment) of past winnings               ║
echo ║    • Possible FBI referral                                   ║
echo ║                                                              ║
echo ║  They run 24/7 bot detection + Video Replay review.          ║
echo ║  A cash-winning bot was DOCUMENTED as BLOCKED with funds     ║
echo ║  NOT RECOVERED.                                              ║
echo ║                                                              ║
echo ║  THIS LAUNCHER IS FOR PRACTICE MODE ONLY (FREE, NO CASH).    ║
echo ╚═════════════════════════════════════════════════════════════╝
echo.

:: Confirm
set /p CONFIRM="Type 'PRACTICE' to confirm you'll only use Practice Mode: "
if /i not "%CONFIRM%"=="PRACTICE" (
    echo Launch cancelled. Only use in Practice Mode.
    pause
    exit /b 1
)

:: Change to project directory
cd /d %PROJECT_ROOT%

:: Start Collector (background data recording - 15 min @ 15fps)
echo Starting Collector (15 min @ 15fps)...
start /b %PYTHON% -m detector.collect --seconds 900 --fps 15 --wait 600
if errorlevel 1 (
    echo ERROR: Failed to start collector
    pause
    exit /b 1
)
echo Collector started.

:: Wait for collector to initialize
timeout /t 3 /nobreak >nul

:: Start Panel (overlay with auto-player)
echo Starting Panel (overlay + auto-player)...
start /b %PYTHON% -m panel.overlay
if errorlevel 1 (
    echo ERROR: Failed to start panel
    pause
    exit /b 1
)
echo Panel started.

:: Success
echo.
echo ============================================================
echo  ✅ BOT LAUNCHED SUCCESSFULLY
echo ============================================================
echo Collector: Recording 15min @ 15fps (background)
echo Panel: Overlay with auto-player (dead-zone left of board)
echo.
echo NEXT STEPS:
echo 1. Open Chrome -> Crystal Champions -> PRACTICE mode
echo 2. Wait for panel to show "READY" (board ROI detected)
echo 3. Click RECOVER -> wait for "PLAYING" state
echo 4. Click PLAY -> start match in browser
echo 5. Bot plays automatically
echo.
echo KILL SWITCH: Ctrl+Shift+X (instant stop)
echo.
echo ⚠️  REMEMBER: Practice Mode ONLY - No cash tournaments!
echo ============================================================
echo.

pause