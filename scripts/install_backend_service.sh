#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="rfsoc-backend.service"
REPO_ROOT="/home/xilinx/RFSoC4x2-AWG"
SERVICE_SOURCE="$REPO_ROOT/deploy/systemd/$SERVICE_NAME"
SERVICE_TARGET="/etc/systemd/system/$SERVICE_NAME"

if [[ "$EUID" -ne 0 ]]; then
    printf 'Run this script with sudo.\n' >&2
    exit 1
fi

if [[ ! -f "$SERVICE_SOURCE" ]]; then
    printf 'Service file not found: %s\n' "$SERVICE_SOURCE" >&2
    exit 1
fi

/bin/bash -lc "source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && python -m pip install -r '$REPO_ROOT/backend/requirements.txt'"
install -m 0644 "$SERVICE_SOURCE" "$SERVICE_TARGET"
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"
systemctl status "$SERVICE_NAME" --no-pager
