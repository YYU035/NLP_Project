@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0context-study.ps1" %*
exit /b %ERRORLEVEL%
