@echo off
setlocal

cd /d "%~dp0"

vivado -mode batch ^
  -source create_project.tcl ^
  -log create_project.log ^
  -journal create_project.jou

endlocal
