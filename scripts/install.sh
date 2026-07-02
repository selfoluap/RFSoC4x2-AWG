#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_PATH="${PYNQ_JUPYTER_NOTEBOOKS:-$HOME/jupyter_notebooks}"
NOTEBOOK_DIR="$NOTEBOOK_PATH/rfsoc4x2-awg"

# Default to a normal install: PYNQ/Ubuntu 22.04 often ship setuptools <64,
# which cannot do PEP 660 editable installs for pyproject-only projects.
if [[ "${RFSOC_AWG_EDITABLE:-0}" == "1" ]]; then
    PIP_INSTALL_ARGS=(-e "$REPO_ROOT")
else
    PIP_INSTALL_ARGS=("$REPO_ROOT")
fi

# Try a normal pip install first. If it fails (e.g. no internet for build
# isolation), retry with --no-build-isolation to use the system setuptools.
# If pip fails entirely, the .pth file and runtime copy below still make
# `import firmware` work — pip install is mainly for dependency resolution
# and the pynq.notebooks entry point.
if ! python3 -m pip install "${PIP_INSTALL_ARGS[@]}"; then
    printf 'Warning: pip install failed. Trying --no-build-isolation...\n' >&2
    if ! python3 -m pip install --no-build-isolation "${PIP_INSTALL_ARGS[@]}"; then
        printf 'Warning: pip install failed. Relying on .pth file for import.\n' >&2
        printf 'Ensure numpy and scipy are installed separately.\n' >&2
    fi
fi

USER_SITE="$(python3 -m site --user-site)"
mkdir -p "$USER_SITE"
printf '%s\n' "$REPO_ROOT" > "$USER_SITE/rfsoc4x2_awg_repo.pth"

# Copy overlays next to the installed firmware package so the bitfile is
# findable even when importing from site-packages without the .pth file.
FIRMWARE_DIR="$(python3 -c "import firmware, pathlib; print(pathlib.Path(firmware.__file__).resolve().parent)" 2>/dev/null || true)"
if [[ -n "$FIRMWARE_DIR" && -d "$FIRMWARE_DIR" ]]; then
    mkdir -p "$FIRMWARE_DIR/overlays"
    cp "$REPO_ROOT/overlays/rfsocawg.bit" "$FIRMWARE_DIR/overlays/" 2>/dev/null || true
    cp "$REPO_ROOT/overlays/rfsocawg.hwh" "$FIRMWARE_DIR/overlays/" 2>/dev/null || true
fi

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
