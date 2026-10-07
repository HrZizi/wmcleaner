@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%.venv\Scripts\wmclean.exe" (
  echo wmclean is not installed yet.
  echo Run setup_gpu.cmd, setup_cpu_ai.cmd, or setup_core.cmd first.
  exit /b 2
)
"%ROOT%.venv\Scripts\wmclean.exe" %*
