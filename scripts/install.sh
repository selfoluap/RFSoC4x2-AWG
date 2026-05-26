#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_PATH="${PYNQ_JUPYTER_NOTEBOOKS:-$HOME/jupyter_notebooks}"

python3 -m pip install -e "$REPO_ROOT"

if command -v pynq >/dev/null 2>&1; then
    if ! pynq get-notebooks --ignore-overlays --path "$NOTEBOOK_PATH" rfsoc4x2-awg; then
        printf 'Warning: pynq get-notebooks failed; copying notebooks directly.\n' >&2
        mkdir -p "$NOTEBOOK_PATH/rfsoc4x2-awg"
        cp -R "$REPO_ROOT/firmware/notebooks/." "$NOTEBOOK_PATH/rfsoc4x2-awg/"
    fi
else
    printf 'Warning: pynq command not found; skipping notebook delivery.\n' >&2
    printf 'Install notebooks later with: pynq get-notebooks --ignore-overlays --path "%s" rfsoc4x2-awg\n' "$NOTEBOOK_PATH" >&2
fi
