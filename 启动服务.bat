@echo off
chcp 65001 >nul
title AI 面试教练
cd /d "%~dp0"
python start.py
pause
