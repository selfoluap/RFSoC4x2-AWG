@echo off
setlocal

cd /d "%~dp0"

if exist .Xil rmdir /s /q .Xil
if exist build rmdir /s /q build
del /q *.log 2>nul
del /q *.jou 2>nul

echo Cleaned generated Vivado files from %cd%

endlocal
