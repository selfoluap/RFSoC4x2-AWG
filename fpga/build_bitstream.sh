#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

vivado -mode batch \
  -source build_bitstream.tcl \
  -log build_bitstream.log \
  -journal build_bitstream.jou
