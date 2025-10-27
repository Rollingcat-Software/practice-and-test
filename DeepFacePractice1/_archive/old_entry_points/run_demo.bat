@echo off
REM Batch script to run demos with proper UTF-8 encoding for Windows

REM Set console to UTF-8
chcp 65001 >nul 2>&1

REM Set Python encoding
set PYTHONIOENCODING=utf-8

REM Run the script passed as argument
.venv\Scripts\python.exe %*
