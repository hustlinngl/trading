from __future__ import annotations

from pathlib import Path


def asset_slug(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def asset_bundle_dir(root: str | Path, symbol: str) -> Path:
    return Path(root) / "models" / "assets" / asset_slug(symbol)


def resolve_signal_bundle(settings, root: str | Path, symbol: str) -> Path:
    """Resolve an asset-specific bundle first; never reuse another asset's global champion."""
    root = Path(root)
    asset = asset_bundle_dir(root, symbol)
    if (asset / "signal_model.joblib").exists():
        return asset
    global_dir = root / getattr(settings, "model_dir", "models/champion")
    if symbol == str(getattr(settings, "symbol", "")) and (global_dir / "signal_model.joblib").exists():
        return global_dir
    return asset


def resolve_trade_window_model(settings, root: str | Path, symbol: str) -> Path:
    asset = asset_bundle_dir(root, symbol) / "trade_window_specialist.joblib"
    if asset.exists():
        return asset
    configured = Path(root) / getattr(
        settings, "trade_window_model_path", "models/champion/trade_window_specialist.joblib"
    )
    if symbol == str(getattr(settings, "symbol", "")) and configured.exists():
        return configured
    return asset
