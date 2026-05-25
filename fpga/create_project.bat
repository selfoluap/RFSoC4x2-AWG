@echo off
setlocal

cd /d "%~dp0"

vivado -mode batch ^
  -source build.tcl ^
  -log build.log ^
  -journal build.jou

endlocal
