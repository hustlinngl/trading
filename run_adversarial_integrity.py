from __future__ import annotations
import numpy as np
import pandas as pd
from ai_trading_lab.data_quality import audit_market_data
from ai_trading_lab.features import make_features, make_oos_features
from ai_trading_lab.statistical_evidence import deflated_sharpe_ratio, combinatorial_pbo
from ai_trading_lab.backtest import run_backtest
from ai_trading_lab.risk import RiskEngine

def market_frame(n=400):
    idx=pd.date_range("2024-01-01",periods=n,freq="15min",tz="UTC")
    rng=np.random.default_rng(123); ret=rng.normal(0,0.002,n); close=100*np.exp(np.cumsum(ret)); open_=np.r_[close[0],close[:-1]]
    high=np.maximum(open_,close)*(1+rng.uniform(0,0.002,n)); low=np.minimum(open_,close)*(1-rng.uniform(0,0.002,n)); vol=rng.uniform(1000,2000,n)
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol},index=idx)

def main():
    df=market_frame(); q=audit_market_data(df,"15m"); assert q.passed
    broken=df.drop(df.index[120:125]); qb=audit_market_data(broken,"15m"); assert not qb.passed and (qb.max_gap_bars>=4 or qb.gap_ratio>0)
    X,y,fr=make_features(df,8); forbidden=[c for c in X.columns if any(k in c.lower() for k in ("future","target","label","forward","lead_"))]; assert not forbidden
    oos=make_oos_features(df.iloc[:300],df.iloc[300:],8); assert len(oos)==100
    r=np.linspace(-0.01,0.02,200); ds1=deflated_sharpe_ratio(r,1); ds60=deflated_sharpe_ratio(r,60); assert ds60<=ds1+1e-12
    mat=np.array([[0.1,0.2,0.3],[0.0,0.1,0.2],[0.2,0.1,0.05],[0.1,0.3,0.15],[0.4,0.2,0.1],[0.1,0.05,0.2],[0.3,0.15,0.1],[0.2,0.25,0.05]],float); pbo=combinatorial_pbo(mat,partitions=8); assert pbo["available"]
    # Event timing invariant: a LONG signal on t must not open until t+1.
    signal=pd.Series("FLAT",index=df.index,dtype=object); signal.iloc[10]="LONG"
    class TinyRisk(RiskEngine):
        pass
    risk=RiskEngine(10000,0.01,0.5,0.20,1.5,2.0,fee_bps=0,slippage_bps=0,max_participation_pct=1.0,impact_bps_per_sqrt=0)
    result=run_backtest(df,signal,risk,10000,fee_bps=0,slippage_bps=0,impact_bps_per_sqrt=0,intrabar_barriers=True,max_holding_bars=2)
    if not result.trades.empty: assert pd.Timestamp(result.trades.iloc[0]["entry_timestamp"])==df.index[11]
    print("adversarial-integrity-ok",{"rows":len(df),"gap_max":qb.max_gap_bars,"dsr_1":ds1,"dsr_60":ds60,"pbo":pbo["pbo"],"trades":len(result.trades)})
if __name__=="__main__": main()
