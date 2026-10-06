import time
from ai_trading_lab.config import load_settings
from ai_trading_lab.paper import one_iteration

s=load_settings()
while True:
    try:
        print(one_iteration(s))
    except Exception as exc:
        print(f"paper loop error: {exc}")
    time.sleep(max(5,int(getattr(s,'poll_seconds',60))))
