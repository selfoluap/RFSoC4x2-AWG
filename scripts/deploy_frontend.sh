#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xilinx/RFSoC4x2-AWG"
FRONTEND_DIST="$REPO_ROOT/frontend/dist"
WEB_ROOT="/var/www/rfsoc-awg"

if [[ "$EUID" -ne 0 ]]; then
    printf 'Run this script with sudo after building frontend/dist.\n' >&2
    exit 1
fi

if [[ ! -d "$FRONTEND_DIST" ]]; then
    printf 'Frontend build not found: %s\n' "$FRONTEND_DIST" >&2
    printf 'Build it locally with: cd frontend && npm install && npm run build\n' >&2
    exit 1
fi

install -d -m 0755 "$WEB_ROOT"
rm -rf "$WEB_ROOT"/*
cp -R "$FRONTEND_DIST"/. "$WEB_ROOT"/
find "$WEB_ROOT" -type d -exec chmod 0755 {} +
find "$WEB_ROOT" -type f -exec chmod 0644 {} +
