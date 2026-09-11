@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stedgeai_analyze.ps1" %*
exit /b %ERRORLEVEL%
