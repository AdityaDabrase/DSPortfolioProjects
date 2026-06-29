#!/usr/bin/env bash
# Download all portfolio datasets. Prefer running from repo root:
#   ./scripts/download_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 scripts/download_data.py
