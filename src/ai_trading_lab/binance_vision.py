from __future__ import annotations
from pathlib import Path
from datetime import date,timedelta,datetime,timezone
import hashlib,io,json,zipfile,requests,pandas as pd
BASE='https://data.binance.vision/data'


def rolling_closed_history_window(now=None, months=24):
    """Return UTC date bounds for full monthly archives plus the latest closed days.

    Monthly Binance Vision archives are only complete after a month closes. The
    latest partial month must therefore be downloaded from daily archives through
    yesterday (UTC), not from the not-yet-published monthly archive.
    """
    months = int(months)
    if months < 1:
        raise ValueError("months must be >= 1")

    stamp = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    today = stamp.normalize()
    recent_start = today.replace(day=1)
    monthly_end = recent_start - pd.Timedelta(days=1)
    start = recent_start - pd.DateOffset(months=months)
    end = today - pd.Timedelta(days=1)
    return {
        "start": start.strftime("%Y-%m-%d"),
        "monthly_end": monthly_end.strftime("%Y-%m-%d"),
        "recent_start": recent_start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
    }
def _months(start,end):
    s=pd.Timestamp(start).to_period('M'); e=pd.Timestamp(end).to_period('M')
    for p in pd.period_range(s,e,freq='M'): yield p.year,p.month
def _timestamp_unit(v):
    return 'us' if abs(float(v))>=1e14 else 'ms'
def _sha256_bytes(raw): return hashlib.sha256(raw).hexdigest()
def _write_provenance(path,url,checksum,checksum_status,bytes_size):
    meta={'url':url,'sha256':_sha256_bytes(path.read_bytes()),'bytes':int(bytes_size),'checksum':checksum,'checksum_status':checksum_status,'retrieved_at':datetime.now(timezone.utc).isoformat()}
    path.with_suffix(path.suffix+'.provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
def clip_history(df,start,end):
    if df.empty:return df
    if not isinstance(df.index,pd.DatetimeIndex):
        if 'timestamp' in df.columns: df=df.set_index('timestamp')
        else: raise TypeError('History must use a DatetimeIndex or timestamp column')
    raw_start=pd.Timestamp(start); raw_end=pd.Timestamp(end)
    start_ts=raw_start.tz_localize('UTC') if raw_start.tzinfo is None else raw_start.tz_convert('UTC')
    end_ts=raw_end.tz_localize('UTC') if raw_end.tzinfo is None else raw_end.tz_convert('UTC')
    if isinstance(start,str) and len(start.strip())==10:start_ts=start_ts.normalize()
    if isinstance(end,str) and len(end.strip())==10:end_ts=end_ts.normalize()+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)
    if end_ts<start_ts:raise ValueError('end must be >= start')
    return df.sort_index().loc[start_ts:end_ts].copy()
def archive_provenance(paths):
    out=[]
    for raw_path in paths:
        path=Path(raw_path); sha=_sha256_bytes(path.read_bytes()); meta_path=path.with_suffix(path.suffix+'.provenance.json'); meta={}
        if meta_path.exists():
            try: meta=json.loads(meta_path.read_text(encoding='utf-8'))
            except Exception: meta={}
        out.append({'path':str(path),'bytes':int(path.stat().st_size),'sha256':sha,'url':meta.get('url'),'checksum':meta.get('checksum'),'checksum_status':meta.get('checksum_status','unknown'),'retrieved_at':meta.get('retrieved_at')})
    return out
def _download(url,name,path,timeout,verify_checksum):
    if path.exists() and path.stat().st_size>0:return path
    r=requests.get(url,timeout=timeout)
    if r.status_code==404:raise FileNotFoundError(f'Binance Vision archive not available: {name}')
    r.raise_for_status(); checksum=None; status='not_requested'
    if verify_checksum:
        try:
            c=requests.get(url+'.CHECKSUM',timeout=timeout)
            if c.ok:
                checksum=c.text.strip().split()[0] if c.text.strip() else None; status='verified' if checksum==_sha256_bytes(r.content) else 'mismatch'
                if status=='mismatch':raise ValueError(f'Checksum mismatch for {name}')
            else:status='missing'
        except requests.RequestException:status='unavailable'
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(r.content); _write_provenance(path,url,checksum,status,len(r.content)); return path
def download_month(symbol,interval='15m',year=None,month=None,market='spot',out_dir='data/raw/binance',timeout=60,verify_checksum=True):
    if year is None or month is None:raise ValueError('year and month are required')
    sym=symbol.replace('/','').upper(); prefix='spot' if market=='spot' else 'futures/um'; folder=Path(out_dir)/market/f'{sym}_{interval}'; name=f'{sym}-{interval}-{year:04d}-{month:02d}.zip'; url=f'{BASE}/{prefix}/monthly/klines/{sym}/{interval}/{name}'
    return _download(url,name,folder/name,timeout,verify_checksum)
def download_daily(symbol,interval='15m',day=None,market='spot',out_dir='data/raw/binance',timeout=60,verify_checksum=True):
    if day is None:raise ValueError('day is required')
    d=pd.Timestamp(day).date(); sym=symbol.replace('/','').upper(); prefix='spot' if market=='spot' else 'futures/um'; folder=Path(out_dir)/market/f'{sym}_{interval}'/'daily'; name=f'{sym}-{interval}-{d.isoformat()}.zip'; url=f'{BASE}/{prefix}/daily/klines/{sym}/{interval}/{name}'
    return _download(url,name,folder/name,timeout,verify_checksum)
def download_range(symbol,interval,start,end,market='spot',out_dir='data/raw/binance',timeout=60,verify_checksum=True,exact=True):
    raw_start=pd.Timestamp(start); raw_end=pd.Timestamp(end)
    start_ts=raw_start.tz_localize('UTC') if raw_start.tzinfo is None else raw_start.tz_convert('UTC')
    end_ts=raw_end.tz_localize('UTC') if raw_end.tzinfo is None else raw_end.tz_convert('UTC')
    if isinstance(start,str) and len(start.strip())==10: start_ts=start_ts.normalize()
    if isinstance(end,str) and len(end.strip())==10: end_ts=end_ts.normalize()+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)
    if end_ts<start_ts: raise ValueError('end must be >= start')
    out=[]; boundary_months=set()
    if exact:
        if start_ts.day!=1: boundary_months.add((start_ts.year,start_ts.month))
        if end_ts.date()!=(end_ts.normalize()+pd.offsets.MonthEnd(0)).date(): boundary_months.add((end_ts.year,end_ts.month))
    for y,m in _months(start,end):
        if (y,m) in boundary_months: continue
        try: out.append(download_month(symbol,interval,y,m,market,out_dir,timeout,verify_checksum))
        except FileNotFoundError: continue
    if exact:
        for y,m in sorted(boundary_months):
            first=max(start_ts.date(),date(y,m,1)); last=min(end_ts.date(),(pd.Timestamp(year=y, month=m, day=1)+pd.offsets.MonthEnd(0)).date())
            d=first
            while d<=last:
                try: out.append(download_daily(symbol,interval,d,market,out_dir,timeout,verify_checksum))
                except FileNotFoundError: pass
                d+=timedelta(days=1)
    return list(dict.fromkeys(out))

def _normalize_vision_csv(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names:raise ValueError('No CSV found in Binance Vision archive')
        with z.open(names[0]) as fh:df=pd.read_csv(fh,header=None)
    cols=['timestamp','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
    if df.shape[1]==11:cols=cols[:-1]
    df.columns=cols[:df.shape[1]]
    df['timestamp']=pd.to_datetime(df['timestamp'],unit=_timestamp_unit(df['timestamp'].iloc[0]),utc=True)
    for c in [c for c in cols if c!='timestamp']:
        if c in df:df[c]=pd.to_numeric(df[c],errors='coerce')
    required=['timestamp','open','high','low','close','volume']
    missing=int(df[required].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"Binance archive contains {missing} rows with missing required OHLCV fields")
    df=df.sort_values('timestamp')
    dup=df[df['timestamp'].duplicated(keep=False)]
    if not dup.empty:
        numeric=[c for c in ('open','high','low','close','volume','quote_volume','trades','taker_buy_base_volume','taker_buy_quote_volume') if c in dup.columns]
        conflicting=dup.groupby('timestamp')[numeric].nunique(dropna=False).max(axis=1)
        if bool((conflicting>1).any()):
            bad=str(conflicting[conflicting>1].index[0])
            raise ValueError(f"Conflicting duplicate market bar in archive at {bad}")
        df=df.drop_duplicates('timestamp',keep='first')
    return df
def merge_archives(paths,out_path=None):
    frames=[_normalize_vision_csv(Path(p).read_bytes()) for p in paths]
    if not frames:return pd.DataFrame()
    df=pd.concat(frames,ignore_index=True).sort_values('timestamp')
    dup=df[df['timestamp'].duplicated(keep=False)]
    if not dup.empty:
        numeric=[c for c in ('open','high','low','close','volume') if c in dup.columns]
        conflicting=dup.groupby('timestamp')[numeric].nunique(dropna=False).max(axis=1)
        if bool((conflicting>1).any()):
            bad=str(conflicting[conflicting>1].index[0])
            raise ValueError(f"Conflicting duplicate market bars across archives at {bad}")
        df=df.drop_duplicates('timestamp',keep='first')
    if out_path:
        p=Path(out_path);p.parent.mkdir(parents=True,exist_ok=True)
        try:df.to_parquet(p,index=False)
        except Exception:df.to_csv(p.with_suffix('.csv'),index=False)
    return df
