#!/usr/bin/env bash
# Generates the Python gRPC stubs for the device server into device_server/gen/.
# Needs: pip install grpcio-tools
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/device_server/gen"
mkdir -p "$OUT"
python3 -m grpc_tools.protoc \
    -I "$REPO_ROOT/device_server/protos" \
    --python_out="$OUT" --grpc_python_out="$OUT" \
    "$REPO_ROOT/device_server/protos/h2pcontrol/rfsocawg/v1/rfsoc_awg.proto"
echo "stubs written to $OUT"
