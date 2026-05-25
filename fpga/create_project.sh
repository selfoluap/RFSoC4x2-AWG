#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

vivado -mode batch \
  -source create_project.tcl \
  -log create_project.log \
  -journal create_project.jou
