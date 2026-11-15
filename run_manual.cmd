@echo off
rem Manual mode launcher - Crystal Champions 1 Hz assistant.
rem Double-click this file to start the compact snackbar + arrow overlay.
rem
rem Usage:
rem   1. Open Chrome -> Crystal Champions -> PRACTICE mode (no cash!)
rem   2. Double-click this file
rem   3. Click SCAN to calibrate the board
rem   4. Watch arrows update every second
rem   5. Tap the two highlighted gems manually

cd /d "%~dp0"
"C:\Users\gurwi\AppData\Local\Programs\Python\Python313\pythonw.exe" main.py manual
