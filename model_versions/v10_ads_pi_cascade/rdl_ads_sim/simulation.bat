@echo off
setlocal
set "HPEESOF_DIR=C:\Keysight\ADS2026_Update1.2"
set "COMPL_DIR=%HPEESOF_DIR%"
set "SIMARCH=win32_64"
set "ADS_PYTHON_DIR=%HPEESOF_DIR%\tools\python"
set "PATH=%HPEESOF_DIR%\bin;%HPEESOF_DIR%\adsptolemy\lib.win32_64;%ADS_PYTHON_DIR%;%PATH%"
cd /d "%~dp0"

"%HPEESOF_DIR%\bin\hpeesofsim.exe" sim.net
pause
endlocal