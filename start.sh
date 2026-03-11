#!/bin/sh
set -e

BUNDLED_DATA_DIR="${BUNDLED_DATA_DIR:-/app/data}"
RUNTIME_DATA_DIR="${DEW_DATA_DIR:-/data}"

mkdir -p "$RUNTIME_DATA_DIR"
if [ -d "$BUNDLED_DATA_DIR" ]; then
  cp -rn "$BUNDLED_DATA_DIR"/* "$RUNTIME_DATA_DIR"/ 2>/dev/null || true
fi

export DEW_DATA_DIR="$RUNTIME_DATA_DIR"
exec uvicorn server.api:app --host 0.0.0.0 --port "${PORT:-8080}"
