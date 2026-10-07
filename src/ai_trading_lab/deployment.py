from __future__ import annotations

from pathlib import Path



def model_semantics_fingerprint(settings) -> str:
    import hashlib
    import json

    fields = {
        "timeframe": str(getattr(settings, "timeframe", "")),
        "horizon_bars": int(getattr(settings, "horizon_bars", 8)),
        "pt_atr": float(getattr(settings, "pt_atr", 1.6)),
        "sl_atr": float(getattr(settings, "sl_atr", 1.0)),
        "memory_k": int(getattr(settings, "memory_k", 32)),
        "memory_information_weighted": bool(getattr(settings, "memory_information_weighted", False)),
        "xgb_estimators": int(getattr(settings, "xgb_estimators", 240)),
        "lgbm_estimators": int(getattr(settings, "lgbm_estimators", 240)),
        "hist_max_iter": int(getattr(settings, "hist_max_iter", 260)),
        "regime_n_init": int(getattr(settings, "regime_n_init", 5)),
        "external_feature_lag_bars": int(getattr(settings, "external_feature_lag_bars", 1)),
        "conformal_level": float(getattr(settings, "conformal_level", 0.90)),
        "efficiency_corr_threshold": float(getattr(settings, "efficiency_corr_threshold", 0.995)),
        "validation_purge_bars": int(getattr(settings, "validation_purge_bars", 12)),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()[:24]

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


def bundle_compatibility(settings, bundle: str | Path, symbol: str) -> tuple[bool, str]:
    """Validate lightweight persisted identity metadata before live/paper inference."""
    bundle = Path(bundle)
    meta_path = bundle / "base_training_meta.json"
    if not meta_path.exists() and bool(getattr(settings, "signal_only_mode", True)):
        return False, "model_metadata_missing"
    if meta_path.exists():
        try:
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("symbol")) != str(symbol):
                return False, "model_symbol_mismatch"
            if str(meta.get("timeframe")) != str(getattr(settings, "timeframe", "")):
                return False, "model_timeframe_mismatch"
            recorded_semantics=meta.get("model_semantics_fingerprint")
            if recorded_semantics and str(recorded_semantics)!=model_semantics_fingerprint(settings):
                return False, "model_semantics_mismatch"
        except Exception as exc:
            return False, f"model_metadata_error:{type(exc).__name__}"
    if bool(getattr(settings, "require_deployment_manifest_for_signal", False)):
        manifest_path = bundle / "deployment_manifest.json"
        if not manifest_path.exists():
            return False, "deployment_manifest_missing"
        try:
            import json
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not bool(manifest.get("ready", False)):
                return False, "deployment_manifest_not_ready"
        except Exception as exc:
            return False, f"deployment_manifest_error:{type(exc).__name__}"
    return True, "ok"
