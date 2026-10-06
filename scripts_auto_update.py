"""Example scheduled retraining hook.
Run this from cron/Task Scheduler rather than embedding an uncontrolled self-modifying loop.
"""
from ai_trading_lab.config import load_settings
from ai_trading_lab.data import exchange_client,fetch_ohlcv
from ai_trading_lab.autolearn import auto_update
s=load_settings(); ex=exchange_client("binance",sandbox=False); df=fetch_ohlcv(ex,s.symbol,s.timeframe,s.lookback_bars); print(auto_update(df,s,model_dir="models"))
