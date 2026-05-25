#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

rm -rf .Xil build
rm -f *.log *.jou

printf 'Cleaned generated Vivado files from %s\n' "$(pwd)"
