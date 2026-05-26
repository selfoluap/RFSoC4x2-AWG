#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_PATH="${PYNQ_JUPYTER_NOTEBOOKS:-$HOME/jupyter_notebooks}"

python3 -m pip install -e "$REPO_ROOT"

if command -v pynq >/dev/null 2>&1; then
    pynq get-notebooks rfsoc4x2-awg --path "$NOTEBOOK_PATH"
else
    printf 'Warning: pynq command not found; skipping notebook delivery.\n' >&2
    printf 'Install notebooks later with: pynq get-notebooks rfsoc4x2-awg --path "%s"\n' "$NOTEBOOK_PATH" >&2
fi
