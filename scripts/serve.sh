#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
