#!/usr/bin/env python3
"""Thin launcher for the read-only Adaptive AI Signal Terminal."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_trading_lab.dashboard_terminal import main


if __name__ == "__main__":
    main()
