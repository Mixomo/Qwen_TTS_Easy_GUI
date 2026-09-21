@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Qwen TTS Easy GUI
set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "UV_NO_CACHE=1"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONPATH=%CD%"
set "HF_HOME=%CD%\.runtime\cache\huggingface"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "CUDA_MODULE_LOADING=LAZY"
set "GRADIO_ANALYTICS_ENABLED=False"
if not exist "%PY_EXE%" (echo [ERROR] Run install.bat first.& pause& exit /b 1)
echo Starting Qwen TTS Easy GUI with project-local Python...
"%PY_EXE%" -u app.py
if errorlevel 1 pause
