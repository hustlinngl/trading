from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import pandas as pd


class DecisionLedger:
    """Logs executed and skipped decisions so the learner gets counterfactual data."""
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS decisions(
                id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT, symbol TEXT,
                price REAL, action TEXT, score REAL, p_up REAL, expected_return REAL,
                regime TEXT, features_json TEXT, status TEXT DEFAULT 'open',
                realized_return REAL, bars_forward INTEGER
            )""")
            cx.commit()

    def append(self, *, observed_at, symbol, price, action, score, p_up,
               expected_return, regime, features=None, bars_forward=8):
        with sqlite3.connect(self.path) as cx:
            cur=cx.execute("INSERT INTO decisions(observed_at,symbol,price,action,score,p_up,expected_return,regime,features_json,bars_forward) VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (str(observed_at),symbol,float(price),action,float(score),float(p_up),float(expected_return),regime,json.dumps(features or {},default=str),int(bars_forward)))
            cx.commit(); return int(cur.lastrowid)

    def settle(self, prices: pd.Series | pd.DataFrame, default_horizon: int = 8) -> int:
        frame = prices.sort_index() if isinstance(prices, pd.DataFrame) else prices.sort_index()
        close = frame['close'] if isinstance(frame, pd.DataFrame) and 'close' in frame.columns else frame
        open_ = frame['open'] if isinstance(frame, pd.DataFrame) and 'open' in frame.columns else None
        n=0
        with sqlite3.connect(self.path) as cx:
            rows=cx.execute("SELECT id,observed_at,price,action,bars_forward FROM decisions WHERE status='open'").fetchall()
            for rid,ts,base,action,h in rows:
                t=pd.Timestamp(ts); t=t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
                entry_pos=close.index.searchsorted(t,side='right'); h=int(h or default_horizon)
                if entry_pos>=len(close) or entry_pos+h>=len(close): continue
                if open_ is not None:
                    entry=float(open_.iloc[entry_pos]); fut=float(close.iloc[entry_pos+h]); ret=fut/entry-1.0
                else:
                    entry=float(close.iloc[entry_pos-1]) if entry_pos > 0 else float(base)
                    fut=float(close.iloc[entry_pos+h]); ret=fut/entry-1.0
                if action=='SHORT': ret=-ret
                cx.execute("UPDATE decisions SET realized_return=?,status='settled' WHERE id=?",(ret,rid)); n+=1
            cx.commit()
        return n
