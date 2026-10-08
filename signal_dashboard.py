#!/usr/bin/env python3
"""Import/executable shim for the read-only Adaptive AI Signal Terminal."""

from __future__ import annotations

import sys

from ai_trading_lab import dashboard_terminal as _impl

if __name__ == "__main__":
    _impl.main()
else:
    # Preserve legacy imports and monkeypatch semantics: importing this module exposes
    # the real implementation object used by the dashboard and its tests.
    sys.modules[__name__] = _impl
