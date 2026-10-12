#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv-atlas/bin/python"
if [[ ! -x "$PY" ]]; then
  python3 -m venv .venv-atlas
fi
if ! "$PY" -c 'import numpy, pandas, sklearn, joblib, yfinance' >/dev/null 2>&1; then
  "$PY" -m pip install -r requirements-atlas.txt
fi
exec "$PY" sakura_atlas.py "$@"
