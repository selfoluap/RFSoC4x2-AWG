@echo off
setlocal

cd /d "%~dp0"

vivado -mode batch ^
  -source build_bitstream.tcl ^
  -log build_bitstream.log ^
  -journal build_bitstream.jou

endlocal
