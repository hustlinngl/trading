"""Example scheduled retraining hook.
Run this from cron/Task Scheduler rather than embedding an uncontrolled self-modifying loop.
"""
from ai_trading_lab.config import load_settings
from ai_trading_lab.data import exchange_client,fetch_ohlcv
from ai_trading_lab.autolearn import auto_update
from ai_trading_lab.deployment import asset_bundle_dir
s=load_settings(); ex=exchange_client(s.exchange,sandbox=False); df=fetch_ohlcv(ex,s.symbol,s.timeframe,s.lookback_bars); print(auto_update(df,s,model_dir=asset_bundle_dir("models",s.symbol)))
