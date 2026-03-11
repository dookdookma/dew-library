#!/bin/sh
set -e

export DEW_DATA_DIR=/app/data
exec uvicorn server.api:app --host 0.0.0.0 --port "${PORT:-8080}"
