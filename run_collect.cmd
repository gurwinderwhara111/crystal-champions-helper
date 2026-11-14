@echo off
rem Collector - Records live Crystal Champions board while you play.
rem Runs under pythonw.exe (NO console) so it never steals focus from Chrome.
cd /d "%~dp0"
"C:\Users\gurwi\AppData\Local\Programs\Python\Python313\pythonw.exe" -m detector.collect --seconds 900 --fps 15 --wait 600