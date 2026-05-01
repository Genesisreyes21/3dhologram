#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
uvicorn app:app --reload --host 127.0.0.1 --port 8000
