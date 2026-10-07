from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from .objectives import robust_performance_utility



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

def deployment_semantics_fingerprint(settings) -> str:
    """Fingerprint every runtime/economic rule that can change signal eligibility."""
    import hashlib
    fields = {
        "model_semantics": model_semantics_fingerprint(settings),
        "exchange": str(getattr(settings, "exchange", "binance")),
        "probability_threshold": float(getattr(settings, "probability_threshold", 0.57)),
        "min_expected_return": float(getattr(settings, "min_expected_return", 0.0015)),
        "min_edge_after_cost_bps": float(getattr(settings, "min_edge_after_cost_bps", 5.0)),
        "decision_threshold": float(getattr(settings, "decision_threshold", 0.16)),
        "meta_threshold": float(getattr(settings, "meta_threshold", 0.53)),
        "conformal_blend": float(getattr(settings, "conformal_blend", 0.60)),
        "uncertainty_penalty_mult": float(getattr(settings, "uncertainty_penalty_mult", 2.0)),
        "regime_weight": float(getattr(settings, "regime_weight", 0.08)),
        "memory_weight": float(getattr(settings, "memory_weight", 0.16)),
        "meta_weight": float(getattr(settings, "meta_weight", 0.18)),
        "conviction_weight": float(getattr(settings, "conviction_weight", 0.34)),
        "edge_weight": float(getattr(settings, "edge_weight", 0.30)),
        "max_holding_bars": int(getattr(settings, "max_holding_bars", 96)),
        "risk_per_trade": float(getattr(settings, "risk_per_trade", 0.005)),
        "max_position_pct": float(getattr(settings, "max_position_pct", 0.25)),
        "max_daily_loss_pct": float(getattr(settings, "max_daily_loss_pct", 0.02)),
        "stop_atr_mult": float(getattr(settings, "stop_atr_mult", 1.8)),
        "take_profit_rr": float(getattr(settings, "take_profit_rr", 2.2)),
        "fee_bps": float(getattr(settings, "fee_bps", 7.0)),
        "slippage_bps": float(getattr(settings, "slippage_bps", 5.0)),
        "impact_bps_per_sqrt": float(getattr(settings, "impact_bps_per_sqrt", 1.5)),
        "max_participation_pct": float(getattr(settings, "max_participation_pct", 0.10)),
        "short_borrow_bps_per_bar": float(getattr(settings, "short_borrow_bps_per_bar", 0.0)),
        "force_daily_loss_exit": bool(getattr(settings, "force_daily_loss_exit", True)),
        "trade_window_enabled": bool(getattr(settings, "trade_window_enabled", True)),
        "trade_window_required_for_signal": bool(getattr(settings, "trade_window_required_for_signal", True)),
        "trade_window_target_precision": float(getattr(settings, "trade_window_target_precision", 0.80)),
        "trade_window_min_holdout_wilson": float(getattr(settings, "trade_window_min_holdout_wilson", 0.60)),
        "trade_window_min_holdout_trades": int(getattr(settings, "trade_window_min_holdout_trades", 12)),
        "trade_window_min_confidence": float(getattr(settings, "trade_window_min_confidence", 0.80)),
        "trade_window_min_net_return": float(getattr(settings, "trade_window_min_net_return", 0.0005)),
        "trade_window_require_positive_holdout_backtest": bool(getattr(settings, "trade_window_require_positive_holdout_backtest", True)),
        "base_min_holdout_trades": int(getattr(settings, "base_min_holdout_trades", 20)),
        "base_require_positive_holdout_return": bool(getattr(settings, "base_require_positive_holdout_return", True)),
        "base_min_holdout_profit_factor": float(getattr(settings, "base_min_holdout_profit_factor", 1.0)),
        "base_max_holdout_drawdown": float(getattr(settings, "base_max_holdout_drawdown", -0.25)),
        "base_min_holdout_utility": float(getattr(settings, "base_min_holdout_utility", 0.0)),
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]

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
    meta = {}
    if not meta_path.exists() and bool(getattr(settings, "signal_only_mode", True)):
        return False, "model_metadata_missing"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("symbol")) != str(symbol):
                return False, "model_symbol_mismatch"
            if str(meta.get("timeframe")) != str(getattr(settings, "timeframe", "")):
                return False, "model_timeframe_mismatch"
            recorded_semantics=meta.get("model_semantics_fingerprint")
            if recorded_semantics and str(recorded_semantics)!=model_semantics_fingerprint(settings):
                return False, "model_semantics_mismatch"
            if bool(getattr(settings, "require_deployment_manifest_for_signal", False)):
                if not meta.get("data_fingerprint"):
                    return False, "model_data_provenance_missing"
                if not meta.get("deployment_semantics_fingerprint"):
                    return False, "deployment_semantics_provenance_missing"
                if str(meta.get("deployment_semantics_fingerprint")) != deployment_semantics_fingerprint(settings):
                    return False, "deployment_semantics_mismatch"
        except Exception as exc:
            return False, f"model_metadata_error:{type(exc).__name__}"
    if bool(getattr(settings, "require_deployment_manifest_for_signal", False)):
        manifest_path = bundle / "deployment_manifest.json"
        if not manifest_path.exists():
            return False, "deployment_manifest_missing"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not bool(manifest.get("ready", False)):
                return False, "deployment_manifest_not_ready"
            if str(manifest.get("symbol")) != str(symbol):
                return False, "deployment_manifest_symbol_mismatch"
            if str(manifest.get("timeframe")) != str(getattr(settings, "timeframe", "")):
                return False, "deployment_manifest_timeframe_mismatch"
            if str(manifest.get("data_fingerprint")) != str(meta.get("data_fingerprint")):
                return False, "deployment_manifest_data_mismatch"
            if str(manifest.get("model_semantics_fingerprint")) != str(meta.get("model_semantics_fingerprint")):
                return False, "deployment_manifest_semantics_mismatch"
            if str(manifest.get("deployment_semantics_fingerprint")) != deployment_semantics_fingerprint(settings):
                return False, "deployment_manifest_runtime_mismatch"
        except Exception as exc:
            return False, f"deployment_manifest_error:{type(exc).__name__}"
    return True, "ok"


def refresh_deployment_manifest(settings, root: str | Path = ".") -> dict:
    """Rebuild deployment evidence from the current asset bundle, binding it to provenance."""
    root=Path(root)
    asset_dir=asset_bundle_dir(root,getattr(settings,"symbol",""))
    meta_path=asset_dir/"base_training_meta.json"
    base_path=asset_dir/"base_holdout_report.json"
    duration_path=asset_dir/"trade_window_training_report.json"
    meta=json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    base=json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    duration=json.loads(duration_path.read_text(encoding="utf-8")) if duration_path.exists() else {}
    h=base.get("holdout",{}) if isinstance(base,dict) else {}
    base_checks={
        "model_artifact_present":(asset_dir/"signal_model.joblib").exists(),
        "model_provenance_present":bool(meta.get("data_fingerprint")) and bool(meta.get("model_semantics_fingerprint")) and bool(meta.get("deployment_semantics_fingerprint")),
        "deployment_semantics_match":str(meta.get("deployment_semantics_fingerprint"))==deployment_semantics_fingerprint(settings),
        "holdout_report_present":bool(base),
        "holdout_trade_support":int(h.get("trades_taken",h.get("trades",0)))>=int(getattr(settings,"base_min_holdout_trades",20)),
        "positive_holdout_return":float(h.get("net_compounded_return",h.get("total_return",h.get("return",-1.0))))>0.0 if bool(getattr(settings,"base_require_positive_holdout_return",True)) else True,
        "profit_factor":float(h.get("profit_factor",0.0))>=float(getattr(settings,"base_min_holdout_profit_factor",1.0)),
        "holdout_drawdown":float(h.get("max_drawdown",-1.0))>=float(getattr(settings,"base_max_holdout_drawdown",-0.25)),
    }
    base_utility=robust_performance_utility(h,min_trades=int(getattr(settings,"base_min_holdout_trades",20)),max_drawdown=float(getattr(settings,"base_max_holdout_drawdown",-0.25))) if h else -1.0
    base_checks["risk_adjusted_utility"]=base_utility>=float(getattr(settings,"base_min_holdout_utility",0.0))
    duration_bt=duration.get("holdout",{}).get("backtest",{}) if isinstance(duration,dict) else {}
    current_fp=meta.get("data_fingerprint") or base.get("data_fingerprint")
    duration_provenance_match=(
        bool(duration)
        and str(duration.get("data_fingerprint"))==str(current_fp)
        and str(duration.get("symbol"))==str(getattr(settings,"symbol",""))
        and str(duration.get("timeframe"))==str(getattr(settings,"timeframe",""))
    )
    duration_utility=robust_performance_utility(duration_bt,min_trades=int(getattr(settings,"trade_window_min_holdout_trades",12)),max_drawdown=float(getattr(settings,"trade_window_max_holdout_drawdown",-0.25))) if duration_bt else -1.0
    duration_checks={
        "duration_artifact_present":(asset_dir/"trade_window_specialist.joblib").exists(),
        "duration_report_present":bool(duration),
        "duration_provenance_match":duration_provenance_match,
        "duration_production_ready":bool(duration.get("production_ready",False)),
        "risk_adjusted_utility":(not bool(duration_bt)) or duration_utility>=float(getattr(settings,"trade_window_min_holdout_utility",0.0)),
    }
    base_provenance_match=(
        bool(base)
        and str(base.get("data_fingerprint"))==str(meta.get("data_fingerprint"))
        and str(base.get("symbol"))==str(getattr(settings,"symbol",""))
    )
    base_checks["holdout_provenance_match"]=base_provenance_match
    if bool(getattr(settings,"trade_window_enabled",True)):
        checks={**base_checks,**duration_checks}
    else:
        checks=base_checks
    ready=all(bool(v) for v in checks.values())
    return_manifest={
        "symbol":str(getattr(settings,"symbol","")),
        "timeframe":str(getattr(settings,"timeframe","")),
        "updated_at":datetime.now(timezone.utc).isoformat(),
        "ready":bool(ready),
        "data_fingerprint":meta.get("data_fingerprint") or base.get("data_fingerprint"),
        "model_semantics_fingerprint":meta.get("model_semantics_fingerprint"),
        "deployment_semantics_fingerprint":meta.get("deployment_semantics_fingerprint"),
        "checks":checks,
        "base_holdout_report":str(base_path),
        "duration_report":str(duration_path),
        "objective":{"base_risk_adjusted_utility":float(base_utility),"duration_risk_adjusted_utility":float(duration_utility)},
    }
    asset_dir.mkdir(parents=True,exist_ok=True)
    tmp=asset_dir/"deployment_manifest.json.tmp"
    tmp.write_text(json.dumps(return_manifest,indent=2,default=str),encoding="utf-8")
    tmp.replace(asset_dir/"deployment_manifest.json")
    return return_manifest
