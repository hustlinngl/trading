#!/usr/bin/env python3
"""Thin executable/import shim for the Adaptive AI Signal Terminal."""

from __future__ import annotations

import sys

from ai_trading_lab import dashboard_terminal as _impl

if __name__ == "__main__":
    _impl.main()
else:
    # Preserve legacy imports and monkeypatch semantics: importing signal_dashboard
    # returns the actual implementation module instead of a copy of its symbols.
    sys.modules[__name__] = _impl
