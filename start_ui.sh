#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -e '.[dashboard]'
python -m streamlit run app.py
