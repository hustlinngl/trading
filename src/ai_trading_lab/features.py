from __future__ import annotations
import numpy as np
import pandas as pd

def _rsi(close,n=14):
    delta=close.diff(); up=delta.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); down=(-delta.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); rs=up/down.replace(0,np.nan); return 100-100/(1+rs)
def _zscore(s,n):
    m=s.rolling(n).mean(); sd=s.rolling(n).std(); return (s-m)/sd.replace(0,np.nan)
def _entropy_binary(close,n=48):
    sign=(close.pct_change()>0).astype(float); p=sign.rolling(n).mean().clip(1e-6,1-1e-6); return -(p*np.log2(p)+(1-p)*np.log2(1-p))
def _efficiency_ratio(close,n=24):
    return close.diff(n).abs()/close.diff().abs().rolling(n).sum().replace(0,np.nan)
def _obv(close,volume):
    return (np.sign(close.diff()).fillna(0)*volume).cumsum()
def _wilder_adx(high,low,close,n=14):
    up=high.diff(); down=-low.diff(); plus_dm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=high.index); minus_dm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=high.index)
    tr=pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/n,adjust=False).mean()
    plus_di=100*plus_dm.ewm(alpha=1/n,adjust=False).mean()/atr.replace(0,np.nan); minus_di=100*minus_dm.ewm(alpha=1/n,adjust=False).mean()/atr.replace(0,np.nan)
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)).fillna(0); return dx.ewm(alpha=1/n,adjust=False).mean()
def _higher_tf_features(df,rule,prefix):
    htf=df[['open','high','low','close','volume']].resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(); htf_ret=htf.close.pct_change(4); htf_vol=htf.close.pct_change().rolling(12).std(); htf_ema=htf.close.ewm(span=12,adjust=False).mean()
    out=pd.DataFrame(index=htf.index); out[f'{prefix}_ret']=htf_ret; out[f'{prefix}_vol']=htf_vol; out[f'{prefix}_ema_gap']=htf.close/htf_ema-1; out[f'{prefix}_rsi']=_rsi(htf.close,14)/100; out[f'{prefix}_range']=(htf.high-htf.low)/htf.close
    out.index=out.index+pd.tseries.frequencies.to_offset(rule); return out.reindex(df.index,method='ffill')
def make_features(df,horizon=8,external_feature_lag_bars=1):
    required={'open','high','low','close','volume'}; missing=required.difference(df.columns)
    if missing: raise ValueError(f'Missing OHLCV columns: {sorted(missing)}')
    x=df.copy().sort_index(); c,h,l,v=x.close,x.high,x.low,x.volume; ret1=c.pct_change(); x['ret_1']=ret1
    for n in [3,6,12,24,48,96,192]: x[f'ret_{n}']=c.pct_change(n); x[f'vol_{n}']=ret1.rolling(n).std(); x[f'ema_gap_{n}']=c/c.ewm(span=n,adjust=False).mean()-1; x[f'volume_z_{n}']=_zscore(v,n)
    prev_c=c.shift(); tr=pd.concat([(h-l),(h-prev_c).abs(),(l-prev_c).abs()],axis=1).max(axis=1); x['tr_pct']=tr/c; x['atr_14']=tr.rolling(14).mean(); x['atr_50']=tr.rolling(50).mean(); x['atr_pct']=x.atr_14/c; x['atr_ratio']=x.atr_14/x.atr_50.replace(0,np.nan)
    x['rsi_14']=_rsi(c,14); ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean(); x['macd']=ema12-ema26; x['macd_signal']=x.macd.ewm(span=9,adjust=False).mean(); x['macd_hist']=x.macd-x.macd_signal; x['macd_hist_z']=_zscore(x.macd_hist,48)
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); x['bb_z']=(c-mid)/sd.replace(0,np.nan); x['bb_width']=2*sd/mid.replace(0,np.nan); x['adx_14']=_wilder_adx(h,l,c,14); x['range_pct']=(h-l)/c; x['close_location']=(c-l)/(h-l).replace(0,np.nan)
    x['vwap_48']=(c*v).rolling(48).sum()/v.rolling(48).sum().replace(0,np.nan); x['vwap_gap']=c/x.vwap_48-1; x['dist_high_48']=c/c.rolling(48).max()-1; x['dist_low_48']=c/c.rolling(48).min()-1
    x['obv_z']=_zscore(_obv(c,v),48); x['efficiency_24']=_efficiency_ratio(c,24); x['sign_entropy_48']=_entropy_binary(c,48); x['autocorr_24']=ret1.rolling(24).corr(ret1.shift(1)); x['skew_48']=ret1.rolling(48).skew(); x['kurt_48']=ret1.rolling(48).kurt()
    for ccol in ['quote_volume','trades','taker_buy_base_volume','taker_buy_quote_volume']:
        if ccol in df.columns: s0=pd.to_numeric(df[ccol],errors='coerce'); x[ccol]=s0; x[f'{ccol}_z_48']=_zscore(s0,48)
    if 'quote_volume' in df.columns and 'taker_buy_quote_volume' in df.columns:
        qv=pd.to_numeric(df.quote_volume,errors='coerce').replace(0,np.nan); tbq=pd.to_numeric(df.taker_buy_quote_volume,errors='coerce'); x['taker_buy_quote_ratio']=tbq/qv; x['taker_buy_imbalance']=2*x.taker_buy_quote_ratio-1
    async_prefixes=('micro_','deriv_','cross_','macro_','ext_','state_')
    for ccol in df.columns:
        if ccol in x.columns or not str(ccol).startswith(async_prefixes): continue
        s0=pd.to_numeric(df[ccol],errors='coerce'); lagged=s0.shift(max(0,int(external_feature_lag_bars))); x[ccol]=lagged; x[f'{ccol}_delta_8']=lagged-lagged.shift(8); x[f'{ccol}_z_48']=_zscore(lagged,48)
    x['body_pct']=(c-x.open)/x.open.replace(0,np.nan); x['upper_wick_pct']=(h-pd.concat([x.open,c],axis=1).max(axis=1))/c; x['lower_wick_pct']=(pd.concat([x.open,c],axis=1).min(axis=1)-l)/c; x['range_z_48']=_zscore(x.range_pct,48); x['volume_trend']=v/v.ewm(span=24,adjust=False).mean()-1
    for rule,prefix in [('1h','htf1h'),('4h','htf4h')]: x=pd.concat([x,_higher_tf_features(df,rule,prefix)],axis=1)
    idx=x.index
    if isinstance(idx,pd.DatetimeIndex):
        x['hour_sin']=np.sin(2*np.pi*idx.hour/24); x['hour_cos']=np.cos(2*np.pi*idx.hour/24); x['dow_sin']=np.sin(2*np.pi*idx.dayofweek/7); x['dow_cos']=np.cos(2*np.pi*idx.dayofweek/7)
    future_entry=x.open.shift(-1); future_exit=x.open.shift(-(int(horizon)+1)); future_ret=future_exit/future_entry-1; y=(future_ret>0).astype(float); y[future_ret.isna()]=np.nan
    return x.replace([np.inf,-np.inf],np.nan),y,future_ret
def make_oos_features(history,future,horizon=8,warmup_bars=None,external_feature_lag_bars=1):
    warmup=int(warmup_bars or max(256,int(horizon)*4))
    context=history.tail(warmup).copy()
    future=future.copy()
    if not isinstance(context.index,pd.DatetimeIndex) or not isinstance(future.index,pd.DatetimeIndex):
        raise TypeError("history and future must use DatetimeIndex")
    if context.index.has_duplicates or future.index.has_duplicates:
        raise ValueError("history and future must have unique timestamps")
    overlap=context.index.intersection(future.index)
    if len(overlap):
        cols=[c for c in context.columns.intersection(future.columns) if c in {'open','high','low','close','volume','quote_volume','trades','taker_buy_base_volume','taker_buy_quote_volume'}]
        for col in cols:
            a=pd.to_numeric(context.loc[overlap,col],errors='coerce')
            b=pd.to_numeric(future.loc[overlap,col],errors='coerce')
            equal=(a.isna() & b.isna()) | np.isclose(a.to_numpy(float),b.to_numpy(float),rtol=1e-10,atol=1e-12,equal_nan=True)
            if not bool(np.all(equal)):
                raise ValueError(f"Conflicting overlapping OOS data at column {col}")
    combined=pd.concat([context,future]).sort_index()
    combined=combined[~combined.index.duplicated(keep='last')]
    features,_,_=make_features(combined,horizon,external_feature_lag_bars=external_feature_lag_bars)
    return features.reindex(future.index)
