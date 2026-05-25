#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

vivado -mode batch \
  -source build_all.tcl \
  -log build_all.log \
  -journal build_all.jou
