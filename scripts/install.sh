#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_PATH="${PYNQ_JUPYTER_NOTEBOOKS:-$HOME/jupyter_notebooks}"
NOTEBOOK_DIR="$NOTEBOOK_PATH/rfsoc4x2-awg"

if [[ "${RFSOC_AWG_WHEEL:-0}" == "1" ]]; then
    python3 -m pip install "$REPO_ROOT"
else
    python3 -m pip install -e "$REPO_ROOT"
fi

USER_SITE="$(python3 -m site --user-site)"
mkdir -p "$USER_SITE"
printf '%s\n' "$REPO_ROOT" > "$USER_SITE/rfsoc4x2_awg_repo.pth"

copy_notebooks() {
    mkdir -p "$NOTEBOOK_DIR"
    cp -R "$REPO_ROOT/firmware/notebooks/." "$NOTEBOOK_DIR/"
}

copy_runtime_files() {
    mkdir -p "$NOTEBOOK_DIR"
    rm -rf "$NOTEBOOK_DIR/firmware" "$NOTEBOOK_DIR/overlays"
    cp -R "$REPO_ROOT/firmware" "$NOTEBOOK_DIR/firmware"
    cp -R "$REPO_ROOT/overlays" "$NOTEBOOK_DIR/overlays"
}

if command -v pynq >/dev/null 2>&1; then
    if ! pynq get-notebooks --force --ignore-overlays --path "$NOTEBOOK_PATH" rfsoc4x2-awg; then
        printf 'Warning: pynq get-notebooks failed; copying notebooks directly.\n' >&2
        if ! copy_notebooks; then
            printf 'Error: could not write notebooks to "%s".\n' "$NOTEBOOK_PATH" >&2
            printf 'If files were created with sudo, fix ownership with: sudo chown -R "$USER:$USER" "%s"\n' "$NOTEBOOK_PATH" >&2
            exit 1
        fi
    fi
else
    printf 'Warning: pynq command not found; copying notebooks directly.\n' >&2
    if ! copy_notebooks; then
        printf 'Error: could not write notebooks to "%s".\n' "$NOTEBOOK_PATH" >&2
        exit 1
    fi
fi

if ! copy_runtime_files; then
    printf 'Error: could not copy runtime files to "%s".\n' "$NOTEBOOK_DIR" >&2
    printf 'If files were created with sudo, fix ownership with: sudo chown -R "$USER:$USER" "%s"\n' "$NOTEBOOK_PATH" >&2
    exit 1
fi

PYTHONPATH="$NOTEBOOK_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import firmware

print(f"Notebook runtime import OK: {firmware.__file__}")
PY

printf 'Installed notebooks and runtime files in: %s\n' "$NOTEBOOK_DIR"
