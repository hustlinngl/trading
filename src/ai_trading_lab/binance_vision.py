from __future__ import annotations

from pathlib import Path
import hashlib
import io
import zipfile
import requests
import pandas as pd

BASE='https://data.binance.vision/data'


def _months(start: str, end: str):
    s=pd.Timestamp(start).to_period('M'); e=pd.Timestamp(end).to_period('M')
    for p in pd.period_range(s,e,freq='M'):
        yield p.year,p.month


def _timestamp_unit(v: float) -> str:
    return 'us' if abs(float(v)) >= 1e14 else 'ms'


def _normalize_vision_csv(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names: raise ValueError('No CSV found in Binance Vision archive')
        with z.open(names[0]) as fh:
            df=pd.read_csv(fh,header=None)
    cols=['timestamp','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
    if df.shape[1] == 11:
        cols=cols[:-1]
    df.columns=cols[:df.shape[1]]
    unit=_timestamp_unit(df['timestamp'].iloc[0])
    df['timestamp']=pd.to_datetime(df['timestamp'],unit=unit,utc=True)
    for c in [c for c in cols if c!='timestamp']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp').drop_duplicates('timestamp')


def download_month(symbol: str, interval: str='15m', year: int|None=None, month: int|None=None,
                   market: str='spot', out_dir: str|Path='data/raw/binance', timeout: int=60,
                   verify_checksum: bool=True) -> Path:
    if year is None or month is None: raise ValueError('year and month are required')
    sym=symbol.replace('/','').upper()
    prefix='spot' if market=='spot' else 'futures/um'
    folder=Path(out_dir)/market/f'{sym}_{interval}'
    folder.mkdir(parents=True,exist_ok=True)
    name=f'{sym}-{interval}-{year:04d}-{month:02d}.zip'
    url=f'{BASE}/{prefix}/monthly/klines/{sym}/{interval}/{name}'
    path=folder/name
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        r=requests.get(url,timeout=timeout)
        if r.status_code == 404:
            raise FileNotFoundError(f'Binance Vision archive not available: {name}')
        r.raise_for_status()
    except requests.HTTPError as exc:
        if getattr(exc.response, 'status_code', None) == 404:
            raise FileNotFoundError(f'Binance Vision archive not available: {name}') from exc
        raise
    if verify_checksum:
        try:
            c=requests.get(url+'.CHECKSUM',timeout=timeout); c.raise_for_status()
            checksum=c.text.strip().split()[0]
            got=hashlib.sha256(r.content).hexdigest()
            if checksum != got: raise ValueError(f'Checksum mismatch for {name}')
        except requests.RequestException:
            pass
    path.write_bytes(r.content); return path


def download_range(symbol: str, interval: str, start: str, end: str, market: str='spot',
                   out_dir: str|Path='data/raw/binance', timeout: int=60,
                   verify_checksum: bool=True) -> list[Path]:
    out=[]
    for y,m in _months(start,end):
        try:
            out.append(download_month(symbol,interval,y,m,market,out_dir,timeout,verify_checksum))
        except FileNotFoundError:
            continue
    return out


def merge_archives(paths: list[str|Path], out_path: str|Path|None=None) -> pd.DataFrame:
    frames=[_normalize_vision_csv(Path(p).read_bytes()) for p in paths]
    if not frames: return pd.DataFrame()
    df=pd.concat(frames,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp')
    if out_path:
        p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True)
        try: df.to_parquet(p,index=False)
        except Exception: df.to_csv(p.with_suffix('.csv'),index=False)
    return df
