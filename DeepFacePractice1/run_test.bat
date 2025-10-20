@echo off
REM Set UTF-8 encoding for Windows console
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8

REM Run the installation test
.venv\Scripts\python.exe test_installation.py

pause
