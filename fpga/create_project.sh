#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

vivado -mode batch \
  -source build.tcl \
  -log build.log \
  -journal build.jou
