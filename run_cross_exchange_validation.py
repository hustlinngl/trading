from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

from ai_trading_lab.binance_vision import download_range, merge_archives
from ai_trading_lab.config import load_settings
from ai_trading_lab.data_quality import audit_market_data
from ai_trading_lab.engine import AdaptiveEngine
from ai_trading_lab.features import make_oos_features
from ai_trading_lab.master_tuner import master_tune
from ai_trading_lab.policy import make_actions
from ai_trading_lab.evaluation import run_configured_backtest

def fetch_kraken_ohlc(pair="XBTUSD", interval=240, timeout=30):
    r=requests.get("https://api.kraken.com/0/public/OHLC",params={"pair":pair,"interval":interval},timeout=timeout)
    r.raise_for_status(); payload=r.json()
    if payload.get("error"): raise RuntimeError(f"Kraken API error: {payload['error']}")
    result=payload.get("result",{}); key=next((k for k in result if k!="last"),None)
    if not key: raise RuntimeError("Kraken OHLC response has no pair data")
    frame=pd.DataFrame(result[key],columns=["timestamp","open","high","low","close","vwap","volume","trades"])
    frame["timestamp"]=pd.to_datetime(pd.to_numeric(frame["timestamp"]),unit="s",utc=True)
    for c in ["open","high","low","close","vwap","volume","trades"]: frame[c]=pd.to_numeric(frame[c],errors="coerce")
    frame=frame.set_index("timestamp").sort_index()
    if len(frame)>1: frame=frame.iloc[:-1]
    return frame[["open","high","low","close","volume"]].dropna()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binance-start",default="2022-01-01"); p.add_argument("--binance-end",default="2026-06-08")
    p.add_argument("--trials",type=int,default=36); p.add_argument("--holdout-frac",type=float,default=0.15)
    p.add_argument("--config",default="config.yaml"); p.add_argument("--data-dir",default="data/cross_exchange")
    p.add_argument("--log-dir",default="logs/intensive_real"); p.add_argument("--timeout",type=int,default=30)
    a=p.parse_args(); settings=load_settings(a.config); settings.symbol="BTC/USDT"; settings.timeframe="4h"; settings.research_timeframe="4h"; settings.horizon_bars=2; settings.max_holding_bars=6
    raw=Path(a.data_dir)/"binance_4h"; paths=download_range("BTC/USDT","4h",a.binance_start,a.binance_end,"spot",raw,timeout=a.timeout)
    binance=merge_archives(paths,Path(a.data_dir)/"merged/BTCUSDT_4h_cross_exchange.parquet")
    if binance.empty: raise RuntimeError("No Binance 4h history")
    if "timestamp" in binance.columns: binance=binance.set_index("timestamp")
    binance=binance.sort_index(); qb=audit_market_data(binance,"4h")
    if not qb.passed: raise RuntimeError(f"Binance data failed: {qb.reasons}")
    tuning=master_tune(binance,settings,trials=a.trials,final_holdout_frac=a.holdout_frac,save_path=Path(a.log_dir)/"cross_exchange_binance_tuning.json")
    best=dict(tuning["best_params"])
    kraken=fetch_kraken_ohlc(timeout=a.timeout); qk=audit_market_data(kraken,"4h")
    if not qk.passed: raise RuntimeError(f"Kraken data failed: {qk.reasons}")
    if len(binance.index) and len(kraken.index) and binance.index.max() >= kraken.index.min():
        raise RuntimeError(
            "Cross-exchange validation requires non-overlapping time ranges; "
            f"Binance ends at {binance.index.max()} while Kraken starts at {kraken.index.min()}"
        )
    engine=AdaptiveEngine(settings); engine.fit(binance)
    features=make_oos_features(binance,kraken,settings.horizon_bars,external_feature_lag_bars=getattr(settings,"external_feature_lag_bars",1))
    tuned=settings
    for key,value in best.items():
        if hasattr(tuned,key): setattr(tuned,key,value)
    actions=make_actions(engine,features,tuned); bt=kraken.copy(); bt["atr_14"]=features["atr_14"].reindex(kraken.index).ffill()
    result=run_configured_backtest(bt,actions,tuned,stop_atr_mult=float(best.get("stop_atr_mult",tuned.stop_atr_mult)),take_profit_rr=float(best.get("take_profit_rr",tuned.take_profit_rr)),max_holding_bars=int(best.get("max_holding_bars",tuned.max_holding_bars)))
    payload={"run_at":datetime.now(timezone.utc).isoformat(),"source":{"training_exchange":"binance","validation_exchange":"kraken","training_symbol":"BTC/USDT","validation_pair":"XBTUSD","training_timeframe":"4h","kraken_interval_minutes":240,"binance_start":str(binance.index.min()),"binance_end":str(binance.index.max()),"kraken_start":str(kraken.index.min()),"kraken_end":str(kraken.index.max())},"data_quality":{"binance":qb.to_dict(),"kraken":qk.to_dict()},"tuned_parameters":best,"kraken_validation":result.stats,"kraken_trade_count":int(len(result.trades)),"tuning_final_holdout":tuning.get("final_holdout_summary"),"pbo":tuning.get("probability_of_backtest_overfitting"),"deflated_sharpe":tuning.get("final_statistical_evidence",{}).get("deflated_sharpe_ratio"),"promotion":"NOT_PERFORMED"}
    out=Path(a.log_dir)/"cross_exchange_validation.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); print(json.dumps(payload,indent=2,default=str))
if __name__=="__main__": main()
