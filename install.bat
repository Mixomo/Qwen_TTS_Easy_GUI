@echo off
setlocal
cd /d "%~dp0"
title Qwen TTS Easy GUI Installer
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
pause
