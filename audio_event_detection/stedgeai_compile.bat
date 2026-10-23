@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stedgeai_compile.ps1" %*
exit /b %ERRORLEVEL%
