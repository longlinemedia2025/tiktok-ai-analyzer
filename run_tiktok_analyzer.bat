@echo off
set PYTHON_EXE="C:\Users\Administrator1\AppData\Local\Programs\Python\Python313\python.exe"
set SCRIPT_PATH="C:\Users\Administrator1\Downloads\Video Prep Tool Test\TikTok Algorithm Rater Tool\tiktok_prep_tool.py"

echo 🎬 Running TikTok Analyzer...
%PYTHON_EXE% %SCRIPT_PATH% "%~1"
pause
