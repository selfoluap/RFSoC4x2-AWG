@echo off
setlocal

cd /d "%~dp0"

vivado -mode batch ^
  -source build_all.tcl ^
  -log build_all.log ^
  -journal build_all.jou

endlocal
